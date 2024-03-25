import os
import cv2
import lightning as L
import segmentation_models_pytorch as smp
import torch
from pprint import pprint
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as albu
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from osgeo import gdal
import torchvision.transforms.functional as TF
import logging
import queue
import threading
from tqdm import tqdm
gdal.UseExceptions()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
class SegModel(L.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, learning_rate=1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # Define transformations for normalization (only for images)
        self.image_transform = transforms.Compose([
            transforms.Normalize(mean=smp.encoders.get_preprocessing_params(encoder_name)['mean'],
                                 std=smp.encoders.get_preprocessing_params(encoder_name)['std'])
        ])
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def preprocess(self, batch):
        images, masks = batch
        if self.image_transform is not None:
            images = torch.stack([self.image_transform(image) for image in images])
        return images, masks

    def shared_step(self, batch, stage):
        images, masks = self.preprocess(batch)
        logits_mask = self(images)
        assert images.ndim == 4
        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Calculate loss
        loss = self.loss_fn(logits_mask, masks)

        logits_mask = torch.sigmoid(logits_mask).float()
        masks = masks.long()
        tp, fp, fn, tn = smp.metrics.get_stats(logits_mask, masks, mode='multilabel', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        # Returning the loss and metrics
        return {
            "loss": loss,
            "iou": iou,
            "f1": f1,
        }

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, "train")
        self.log('train_loss', metrics['loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('train_iou', metrics['iou'], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_f1', metrics['f1'], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, "val")
        self.log('val_loss', metrics['loss'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', metrics['iou'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', metrics['f1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return metrics

    def on_train_epoch_end(self, unused=None):
        # Access the logged metrics
        metrics = self.trainer.callback_metrics
        print(
            f"/n[Epoch {self.current_epoch} Training] Loss: {metrics['train_loss']:.4f}, IoU: {metrics['train_iou']:.4f}, F1: {metrics['train_f1']:.4f}")

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(
            f"[Epoch {self.current_epoch} Validation] Loss: {metrics['val_loss']:.4f}, IoU: {metrics['val_iou']:.4f}, F1: {metrics['val_f1']:.4f}")

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, "test")
        self.log('test_loss', metrics['loss'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_iou', metrics['iou'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', metrics['f1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10),
            'monitor': 'val_iou',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def resize_and_pad_image(img, split_size, pad_color=(255, 255, 255), new_size=1024):
    h, w = img.shape[:2]
    padded_img = np.full((split_size, split_size, 3), pad_color, dtype=np.uint8)
    padded_img[:h, :w, :] = img
    padded_img = cv2.resize(padded_img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    return padded_img

def apply_clahe_to_rgb_image(rgb_image, clip_limit=3, tile_grid_size=(4, 4)):
    # Convert the BGR image to Lab color space
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    rgb_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2RGB)
    return rgb_image_clahe

def cut_image(picture, num_bands, w0, h0, w1, h1):
    data = [picture.GetRasterBand(band + 1).ReadAsArray(w0, h0, w1-w0, h1-h0) for band in range(num_bands)]
    pic = np.stack(data, axis=-1)
    return pic

def generate_cuts(pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max, cut_width, cut_height, offsets=(0,0)):
    for w in range(pixel_x_min + offsets[0], pixel_x_max, cut_width):
        for h in range(pixel_y_min + offsets[1], pixel_y_max, cut_height):
            w0, h0 = w, h
            w1, h1 = min(w0 + cut_width, pixel_x_max), min(h0 + cut_height, pixel_y_max)
            yield w0, h0, w1, h1

# def split_image_large(image_paths, split_arr, input_img_size, augment=False):
#     logging.info('开始切割图片')
#     split_images_dict = {}
#     for image_path in tqdm(image_paths):
#         if image_path not in split_images_dict:
#             split_images_dict[image_path] = {'split_images': [], 'results': []}
#         picture = gdal.Open(image_path)
#         ratio = pixel_2_meter(image_path)
#         width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
#         adfGeoTransform = picture.GetGeoTransform()
#         split_images_dict[image_path]['parameters'] = {'ratio': ratio, 'width': width, 'height': height, 'num_bands': num_bands, 'adfGeoTransform': adfGeoTransform}
#         for cut_width, cut_height in split_arr:
#             cut_width, cut_height = int(cut_width * ratio), int(cut_height * ratio)
#             offsets = [(0, 0), (int(cut_width / 2), 0), (0, int(cut_height / 2)), (int(cut_width / 2), int(cut_height / 2))]
#             for offset in offsets:
#                 for w0, h0, w1, h1 in generate_cuts(0, width, 0, height, cut_width, cut_height, offset):
#                     if w1 > w0 and h1 > h0:  # Check if the cut dimensions are valid
#                         pic = cut_image(picture, num_bands, w0, h0, w1, h1) # RGB Image
#                         if augment:
#                             pic = apply_clahe_to_rgb_image(pic)
#                         # pic = resize_and_pad_image(pic, pad_color=(0, 0, 0), new_size=input_img_size)
#                         pic = resize_and_pad_image(pic, cut_width, pad_color=(255, 255, 255), new_size=input_img_size)
#                         image_pil = Image.fromarray(pic)
#                         # image_pil.save('test.png')
#                         split_images_dict[image_path]['split_images'].append({'image_path': image_path, 'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': [cut_width, cut_height]})
#     logging.info('切割图片完成')
#     return split_images_dict

def process_image_cut(image_path, num_bands, w0, h0, w1, h1, augment, cut_width, cut_height, input_img_size, pad_color=(255, 255, 255)):
    picture = gdal.Open(image_path)
    pic = cut_image(picture, num_bands, w0, h0, w1, h1)
    if augment:
        pic = apply_clahe_to_rgb_image(pic)
    pic = resize_and_pad_image(pic, cut_width, pad_color, new_size=input_img_size)
    image_pil = Image.fromarray(pic)
    return {'image_path': image_path, 'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': [cut_width, cut_height]}


def split_image_large(image_paths, split_arr, input_img_size, augment=False):
    logging.info('开始切割图片')
    split_images_dict = {}

    with ThreadPoolExecutor() as executor:
        future_to_cut = {}
        tasks = []  # Keep track of all submitted tasks

        for image_path in image_paths:
            if image_path not in split_images_dict:
                split_images_dict[image_path] = {'split_images': [], 'results': []}
            picture = gdal.Open(image_path)
            ratio = pixel_2_meter(image_path)
            width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
            adfGeoTransform = picture.GetGeoTransform()
            split_images_dict[image_path]['parameters'] = {'ratio': ratio, 'width': width, 'height': height,
                                                           'num_bands': num_bands, 'adfGeoTransform': adfGeoTransform}
            for cut_width, cut_height in split_arr:
                cut_width, cut_height = int(cut_width * ratio), int(cut_height * ratio)
                offsets = [(0, 0), (int(cut_width / 2), 0), (0, int(cut_height / 2)),
                           (int(cut_width / 2), int(cut_height / 2))]
                for offset in offsets:
                    for w0, h0, w1, h1 in generate_cuts(0, width, 0, height, cut_width, cut_height, offset):
                        if w1 > w0 and h1 > h0:  # Check if the cut dimensions are valid
                            future = executor.submit(process_image_cut, image_path, num_bands, w0, h0, w1, h1, augment,
                                                     cut_width, cut_height, input_img_size, pad_color=(255, 255, 255))
                            future_to_cut[future] = image_path
                            tasks.append(future)

        # Use tqdm to show progress
        for future in tqdm(as_completed(future_to_cut), total=len(tasks), desc="Processing Cuts"):
            image_path = future_to_cut[future]
            try:
                result = future.result()
                split_images_dict[image_path]['split_images'].append(result)
            except Exception as exc:
                logging.error('%r generated an exception: %s' % (image_path, exc))

    logging.info('切割图片完成')
    return split_images_dict

def model_predict(model_path, split_images_dict, gpu_ids):
    logging.info('开始模型预测')
    split_images_all = []
    for value in split_images_dict.values():
        split_images_all.extend(value['split_images'])
    subsets = [split_images_all[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    # Calculate the total number of images to be processed
    total_images = sum(len(subset) for subset in subsets)
    # Initialize the shared tqdm progress bar
    progress_bar = tqdm(total=total_images, desc="Processing Images")
    # A thread function to process a subset of images on a specific GPU
    def process_images_batch(gpu_id, images_subset, output_queue, progress_bar):
        device = f'cuda:{gpu_id}'
        model = SegModel.load_from_checkpoint(model_path).to(device)
        model.eval()  # Set the model to evaluation mode

        batch_size = 16
        num_images = len(images_subset)

        with torch.no_grad():
            # Process images in batches
            for batch_start in range(0, num_images, batch_size):
                batch_end = min(batch_start + batch_size, num_images)
                batch_images = [images_subset[i]['pic'].convert('RGB') for i in range(batch_start, batch_end)]

                # Preprocess images as a batch
                batch_tensors = torch.stack([TF.to_tensor(img) for img in batch_images]).to(device)
                batch_transformed = model.image_transform(batch_tensors)  # Assuming this can handle a batch

                # Perform inference on the whole batch
                predictions = model(batch_transformed)
                predictions = torch.sigmoid(predictions).squeeze().cpu()

                # Post-process predictions to generate masks
                for i, prediction in enumerate(predictions):
                    threshold = 0.5
                    mask = prediction > threshold
                    mask_uint8 = mask.to(torch.uint8) * 255
                    mask_pil = TF.to_pil_image(mask_uint8)

                    # Resize mask to original size
                    original_size = images_subset[batch_start + i]['size']
                    mask_pil_resized = mask_pil.resize(original_size, resample=Image.NEAREST)
                    resized_mask_tensor = TF.to_tensor(mask_pil_resized).to(torch.bool).squeeze()

                    # Save the mask in the subset
                    images_subset[batch_start + i]['mask'] = resized_mask_tensor

                    # Update the progress bar
                    progress_bar.update(1)

        # Safely put the processed images back into the output queue
        output_queue.put(images_subset)
    def process_images(gpu_id, images_subset, output_queue, progress_bar):
        device = f'cuda:{gpu_id}'
        model = SegModel.load_from_checkpoint(model_path).to(device)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for i in range(len(images_subset)):
                image = images_subset[i]['pic'].convert('RGB')
                split_arr = images_subset[i]['size']
                image = TF.to_tensor(image).unsqueeze(0).to(model.device)
                image = model.image_transform(image)
                # Perform inference
                prediction = model(image)
                prediction = torch.sigmoid(prediction).squeeze().cpu()
                # determine = np.any(prediction.numpy()>2)
                # print(determine)

                # Convert to binary mask
                threshold = 0.5
                mask = prediction > threshold
                # Save the mask
                mask_uint8 = mask.to(torch.uint8) * 255  # Convert bool to uint8 and scale to 0-255
                mask_pil = TF.to_pil_image(mask_uint8)
                mask_pil = mask_pil.resize((split_arr[0], split_arr[1]), resample=Image.NEAREST)
                resized_mask_tensor = TF.to_tensor(mask_pil).to(torch.bool).squeeze()

                images_subset[i]['mask'] = resized_mask_tensor
                # torch.cuda.empty_cache()
                progress_bar.update(1)
        output_queue.put(images_subset)  # Safely put the result into the queue

    # Create a queue to hold the results from each thread
    results_queue = queue.Queue()

    # Create and start a thread for each GPU/subset
    threads = []
    for i in range(len(gpu_ids)):
        if not i < len(subsets):
            continue
        gpu_id = gpu_ids[i]
        thread = threading.Thread(target=process_images_batch, args=(gpu_id, subsets[i], results_queue, progress_bar))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Close the progress bar
    progress_bar.close()

    # Collect results from the queue
    all_results = []
    while not results_queue.empty():
        all_results.extend(results_queue.get())
    # {'image_path': image_path, 'location': [w0, h0, w1, h1], 'pic': pic, 'result': ___}

    for j in range(len(all_results)):

        image_path = all_results[j]['image_path']
        split_images_dict[image_path]['results'].append(all_results[j])
    logging.info('模型预测完成')
    return split_images_dict

def smooth_and_threshold_bool_mask(mask, ksize=(5, 5)):
    """
    Apply Gaussian Blur and thresholding to a boolean mask.

    Parameters:
    - mask: A NumPy array representing the input boolean mask.

    Returns:
    - smooth_mask: A boolean NumPy array where the mask has been smoothed and thresholded.
    """
    # Convert boolean mask to uint8 [0, 255]
    mask_uint8 = mask.astype(np.uint8) * 255

    # Apply Gaussian Blur
    blurred_mask = cv2.GaussianBlur(mask_uint8, ksize, 0)  # (49, 49) is the kernel size, 0 is sigmaX

    # Threshold the blurred image to binarize it back to 0 and 255
    _, smooth_mask_uint8 = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    # Convert uint8 mask back to boolean
    smooth_mask_bool = smooth_mask_uint8.astype(bool)

    return smooth_mask_bool


def overlay_masks(split_images_dict):
    logging.info('开始后处理')
    for image_path, split_image_dict in tqdm(split_images_dict.items(), desc="Processing images"):

        parameters = split_image_dict['parameters']
        width, height = parameters['width'], parameters['height']
        large_mask = np.zeros((height, width), dtype=bool)

        for result in split_image_dict['results']:
            location = result['location']
            mask = result['mask'].numpy()
            y_start, x_start = location[1], location[0]
            y_end, x_end = y_start + mask.shape[0], x_start + mask.shape[1]
            # Calculate the bounds for cutting if necessary
            y_end_cut = min(y_end, large_mask.shape[0])
            x_end_cut = min(x_end, large_mask.shape[1])
            mask = mask[:y_end_cut - y_start, :x_end_cut - x_start]

            # Extract the region of the larger mask that corresponds to the size of the smaller mask
            region = large_mask[y_start:y_start + mask.shape[0], x_start:x_start + mask.shape[1]]

            # Overlay the smaller mask onto the larger mask
            # Use logical OR to combine masks, ensuring 1s in the smaller mask overwrite the corresponding area
            large_mask[y_start:y_start + mask.shape[0], x_start:x_start + mask.shape[1]] = np.logical_or(region,
                                                                                                         mask).astype(
                int)

        large_mask = smooth_and_threshold_bool_mask(large_mask)
        split_images_dict[image_path]['mask'] = large_mask
    logging.info('后处理完成')
    return split_images_dict

def pixel_2_meter(img_path):
    # Open the raster file using GDAL
    ds = gdal.Open(img_path)

    # Get raster size (width and height)
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Get georeferencing information
    geoTransform = ds.GetGeoTransform()
    pixel_size_x = geoTransform[1]  # Pixel width
    pixel_size_y = abs(geoTransform[5])  # Pixel height (absolute value)

    # Get the top latitude from the geotransform and the height
    # geoTransform[3] is the top left y, which gives the latitude
    latitude = geoTransform[3] - pixel_size_y * height
    # Close the dataset
    ds = None

    # Convert road width from meters to pixels
    # road_width_meters = line_width
    meters_per_degree = 111139 * math.cos(math.radians(latitude))
    thickness_pixels_ratio = 1 / (pixel_size_x * meters_per_degree)
    return thickness_pixels_ratio

def save_masks(split_images_dict, out_dir):
    for image_path, split_image_dict in split_images_dict.items():
        file_name = os.path.split(image_path)[1]

        parameters = split_image_dict['parameters']
        result_mask = split_image_dict['mask']
        uint8_mask = (result_mask * 255).astype(np.uint8)
        # Convert the uint8 NumPy array to a PIL Image
        pil_image = Image.fromarray(uint8_mask, mode='L')  # 'L' mode for grayscale
        pil_image.save(os.path.join(out_dir, file_name))

    return split_images_dict

def batch_predict(image_paths, split_arr, model_path, out_dir, gpu_ids, input_img_size):

    split_images_dict = split_image_large(image_paths, split_arr, input_img_size, augment=False)
    split_images_dict = model_predict(model_path, split_images_dict, gpu_ids)

    # Overlay the smaller mask onto the larger one
    split_images_dict = overlay_masks(split_images_dict)
    split_images_dict = save_masks(split_images_dict, out_dir)

def change_img_path(image_paths):
    # 如果是文件就输出文件；如果是路径就输出路径下图片文件
    if os.path.isfile(image_paths):
        out_image_paths = [image_paths]
    else:
        out_image_paths = []
        image_names = os.listdir(image_paths)
        for image_name in image_names:
            if image_name.endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(image_paths, image_name)
                out_image_paths.append(image_path)
    return out_image_paths

def main():
    img_path = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/test/test_img"
    split_arr = [[500, 500], [700, 700], [900, 900]]
    gpu_ids = [0, 1, 2, 3]
    input_img_size = 1024
    out_dir = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/out"
    model_path = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/model_checkpoints/best-checkpoint-v1.ckpt"
    image_paths = change_img_path(img_path)
    batch_predict(image_paths, split_arr, model_path, out_dir, gpu_ids, input_img_size)

if __name__ == '__main__':
    main()
