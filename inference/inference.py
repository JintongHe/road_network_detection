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
import math
from osgeo import gdal
import torchvision.transforms.functional as TF
import logging
from tqdm import tqdm
gdal.UseExceptions()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
        #
        # # Convert logits to binary predictions
        # probs = torch.sigmoid(logits_mask)
        # preds = (probs > 0.5).float()
        #
        # # Calculate IoU
        # intersection = (preds * masks).sum()
        # union = (preds + masks).sum() - intersection
        # iou = (intersection + 1e-6) / (union + 1e-6)  # Adding a small constant to avoid division by zero
        #
        # # Calculate F1 Score
        # tp = intersection
        # fp = (preds * (1 - masks)).sum()
        # fn = ((1 - preds) * masks).sum()
        # f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)  # Adding a small constant to avoid division by zero
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

def lon_lat_to_pixel(image_path, coordinates):
    dataset = gdal.Open(image_path)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()

    pixel_arr = []
    if not coordinates or len(coordinates) <= 1:
        pixel_arr.append([0, 0])
        pixel_arr.append([width, height])
    else:
        for coordinate in coordinates:
            coordinate_x = coordinate[0]
            coordinate_y = coordinate[1]
            location = [int((coordinate_x - geo_transform[0]) / geo_transform[1]),
                        int((coordinate_y - geo_transform[3]) / geo_transform[5])]
            pixel_arr.append(location)
    return pixel_arr

def resize_and_pad_image(img, pad_color=(255, 255, 255), new_size=640):
    h, w = img.shape[:2]

    # Determine the scaling factor and resize
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a new image with the desired size and white background
    # padded_img = np.full((split_size[0], split_size[1], 3), pad_color, dtype=np.uint8)
    padded_img = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)

    # Place the resized image onto the center of the square image
    padded_img[:new_h, :new_w, :] = img

    return padded_img

# 求像素点坐标最大值，最小值
def pixel_max_min(pixel_arr):
    pixel_x_min = -1
    pixel_x_max = -1
    pixel_y_min = -1
    pixel_y_max = -1
    for pixel in pixel_arr:
        pixel_x = pixel[0]
        pixel_y = pixel[1]
        if pixel_x_min == -1:
            pixel_x_min = pixel_x
        if pixel_y_min == -1:
            pixel_y_min = pixel_y
        if pixel_x_max == -1:
            pixel_x_max = pixel_x
        if pixel_y_max == -1:
            pixel_y_max = pixel_y

        if pixel_x > pixel_x_max:
            pixel_x_max = pixel_x
        if pixel_x < pixel_x_min:
            pixel_x_min = pixel_x
        if pixel_y > pixel_y_max:
            pixel_y_max = pixel_y
        if pixel_y < pixel_y_max:
            pixel_y_max = pixel_y
    return pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max

def apply_clahe_to_rgb_image(rgb_image, clip_limit=2, tile_grid_size=(8, 8)):
    # Convert the BGR image to Lab color space
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    rgb_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2RGB)
    return rgb_image_clahe

def split_image(image_path, split_arr, pixel_arr):
    logging.info('开始切割图片')
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        # picture = cv2.imread(image_path)
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount

        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                # pic = picture[h0: h1, w0: w1, :]
                data = []
                for band in range(num_bands):
                    b = picture.GetRasterBand(band + 1)
                    data.append(b.ReadAsArray(w0, h0, w1-w0, h1-h0))
                # Assuming the image is RGB
                pic = np.zeros((h1-h0, w1-w0, num_bands), dtype=np.uint8)
                for b in range(num_bands):
                    pic[:, :, 0] = data[0]  # Red channel
                    pic[:, :, 1] = data[1]  # Green channel
                    pic[:, :, 2] = data[2]  # Blue channel
                pic = resize_and_pad_image(pic, pad_color=(0, 0, 0))
                image_pil = Image.fromarray(pic)
                image_pil.save('test.png')
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': split_data}
                num += 1
    logging.info('切割图片完成')
    return split_images_dict

def split_image_large(image_path, split_arr, pixel_arr, ratio, augment=True):
    logging.info('开始切割图片')
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = int(split_data[0] * ratio)
        cut_height = int(split_data[1] * ratio)
        # 读取要分割的图片，以及其尺寸等数据
        # picture = cv2.imread(image_path)
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount

        # 计算可以划分的横纵的个数
        for w in tqdm(range(pixel_x_min, pixel_x_max - 1, cut_width)):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                # pic = picture[h0: h1, w0: w1, :]
                data = []
                for band in range(num_bands):
                    b = picture.GetRasterBand(band + 1)
                    data.append(b.ReadAsArray(w0, h0, w1-w0, h1-h0))
                # Assuming the image is RGB
                pic = np.zeros((h1-h0, w1-w0, num_bands), dtype=np.uint8)
                for b in range(num_bands):
                    pic[:, :, 0] = data[0]  # Red channel
                    pic[:, :, 1] = data[1]  # Green channel
                    pic[:, :, 2] = data[2]  # Blue channel
                if augment:
                    pic = apply_clahe_to_rgb_image(pic)
                pic = resize_and_pad_image(pic, pad_color=(0, 0, 0))
                image_pil = Image.fromarray(pic)
                image_pil.save('test.png')
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': [cut_width, cut_height]}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[0]  # Red channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[2]  # Blue channel
                    if augment:
                        pic = apply_clahe_to_rgb_image(pic)
                    pic = resize_and_pad_image(pic, pad_color=(0, 0, 0))
                    image_pil = Image.fromarray(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': [cut_width, cut_height]}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[0]  # Red channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[2]  # Blue channel
                    if augment:
                        pic = apply_clahe_to_rgb_image(pic)
                    pic = resize_and_pad_image(pic, pad_color=(0, 0, 0))
                    image_pil = Image.fromarray(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': [cut_width, cut_height]}
                    num += 1
                # 情况4
                w0 = w + int(cut_width / 2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[0]  # Red channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[2]  # Blue channel
                    if augment:
                        pic = apply_clahe_to_rgb_image(pic)
                    pic = resize_and_pad_image(pic, pad_color=(0, 0, 0))
                    image_pil = Image.fromarray(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': image_pil, 'size': [cut_width, cut_height]}
                    num += 1
    logging.info('切割图片完成')
    return split_images_dict


def model_predict(split_images_dict, model_path):
    logging.info('开始模型预测')
    # Load the model from checkpoint
    model = SegModel.load_from_checkpoint(model_path)
    model = model.to('cuda')
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for key in tqdm(split_images_dict.keys()):
            image = split_images_dict[key]['pic'].convert('RGB')
            split_arr = split_images_dict[key]['size']
            image = TF.to_tensor(image).unsqueeze(0).to(model.device)  # Convert to tensor and add batch dimension
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
            split_images_dict[key]['mask'] = resized_mask_tensor
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

def overlay_masks(split_images_dict, img_path):
    picture = gdal.Open(img_path)
    width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
    large_mask = np.zeros((height, width), dtype=bool)
    for key in split_images_dict.keys():
        location = split_images_dict[key]['location']
        mask = split_images_dict[key]['mask'].numpy()
        y_start, x_start = location[1], location[0]
        y_end, x_end = y_start + mask.shape[0], x_start + mask.shape[1]
        # Calculate the bounds for cutting if necessary
        y_end_cut = min(y_end, large_mask.shape[0])
        x_end_cut = min(x_end, large_mask.shape[1])
        mask = mask[:y_end_cut - y_start, :x_end_cut - x_start]

        # Extract the region of the larger mask that corresponds to the size of the smaller mask
        region = large_mask[y_start:y_start+mask.shape[0], x_start:x_start+mask.shape[1]]

        # Overlay the smaller mask onto the larger mask
        # Use logical OR to combine masks, ensuring 1s in the smaller mask overwrite the corresponding area
        large_mask[y_start:y_start+mask.shape[0], x_start:x_start+mask.shape[1]] = np.logical_or(region, mask).astype(int)
    large_mask = smooth_and_threshold_bool_mask(large_mask)
    return large_mask

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

def single_predict(img_path, split_arr, model_path, out_dir, coordinates):
    file_name = os.path.split(img_path)[1]
    pixel_arr = lon_lat_to_pixel(img_path, coordinates)
    ratio = pixel_2_meter(img_path)
    split_images_dict = split_image_large(img_path, split_arr, pixel_arr, ratio, augment=True)
    split_images_dict = model_predict(split_images_dict, model_path)

    # Overlay the smaller mask onto the larger one
    result_mask = overlay_masks(split_images_dict, img_path)

    uint8_mask = (result_mask * 255).astype(np.uint8)

    # Convert the uint8 NumPy array to a PIL Image
    pil_image = Image.fromarray(uint8_mask, mode='L')  # 'L' mode for grayscale
    pil_image.save(os.path.join(out_dir, file_name))

def batch_predict(image_file_path, split_arr, model_path, out_dir, coordinates):
    image_names = os.listdir(image_file_path)
    for image_name in image_names:
        if image_name.endswith(IMAGE_EXTENSIONS):
            print('预测，', image_name)
            image_path = image_file_path + '/' + image_name
            single_predict(image_path, split_arr, model_path, out_dir, coordinates)

def main():
    img_path = "/home/zkxq/project/hjt/smp/test_images/access_road_fengji12_20171228.tif"
    img_dir = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/test/test_img"
    split_arr = [[300, 300], [500, 500], [700, 700]]
    coordinates = []
    out_dir = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/out"
    model_path = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/model_checkpoints/best-checkpoint.ckpt"
    # single_predict(img_path, split_arr, model_path, out_dir, coordinates)
    batch_predict(img_dir, split_arr, model_path, out_dir, coordinates)

if __name__ == '__main__':
    main()
