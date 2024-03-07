import os

import lightning as L
import segmentation_models_pytorch as smp
import torch
from pprint import pprint
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import ModelCheckpoint
import albumentations as albu
from lightning.pytorch.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from lightning.pytorch.tuner import Tuner

# Set matrix multiplication precision to 'medium' to utilize Tensor Cores
torch.set_float32_matmul_precision('medium')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')


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


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None, augmentation=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = None
        mask = None

        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            mask = Image.open(self.mask_paths[idx])  # Assuming mask is a PIL image
            if self.image_transform is not None:
                image = self.image_transform(image)

            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            image = np.array(image)
            mask = np.array(mask).squeeze()

            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            image = transforms.ToTensor()(image).float()
            mask = torch.from_numpy(mask).bool().unsqueeze(0).long().float()
        except Exception as e:
            print(f"Error opening image or mask at index {idx}: {e}")
            return

        return image, mask


def convert_trimap_to_binary(mask):
    mask_array = np.array(mask)

    # Convert to torch tensor and perform the binary conversion
    mask = mask_array.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    # mask[mask != 1.0] = 0.0
    # convert to torch tensor
    mask = torch.from_numpy(mask)
    return mask


def convert_grayscale_to_binary(mask):
    mask_array = np.array(mask)

    # Set a threshold
    threshold = 128  # This can vary depending on your image
    # Convert to binary format
    mask = (mask_array > threshold).astype(np.uint8)
    # convert to PIL
    mask = Image.fromarray(mask)
    return mask


def get_training_augmentation(img_size):
    train_transform = [
        # Horizontal flipping doesn't change road network structure
        albu.HorizontalFlip(p=0.5),

        # Mild scaling and rotating. Keeping rotate_limit low to avoid extreme angles
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),

        # Padding and random cropping can help the model generalize to different sizes and aspect ratios
        albu.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
        albu.RandomCrop(height=img_size, width=img_size, always_apply=True),

        # Mild Gaussian noise can help with generalization but should not be too strong to avoid blurring road details
        albu.GaussNoise(p=0.1, var_limit=(10.0, 50.0)),

        # Mild random brightness and contrast adjustments
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        # Adjust hue, saturation, value to simulate different lighting conditions and camera settings
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

        # Mild sharpening can help enhance road edges
        albu.Sharpen(p=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(img_size, img_size)
    ]
    return albu.Compose(test_transform)


class SegDataModule(L.LightningDataModule):
    def __init__(self, train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, test_image_paths,
                 test_mask_paths, img_size=512, batch_size=8):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.train_image_paths = train_image_paths
        self.train_mask_paths = train_mask_paths
        self.val_image_paths = val_image_paths
        self.val_mask_paths = val_mask_paths
        self.test_image_paths = test_image_paths
        self.test_mask_paths = test_mask_paths
        self.batch_size = batch_size
        self.n_cpu = os.cpu_count()
        self.img_size = img_size

        # Define transformations
        self.transform = transforms.Compose([

            # Resize
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ToTensor(),
            # Add any other transformations you need
        ])

        self.mask_transform = transforms.Compose([
            # Add any mask transformations (usually spatial) you need
            # transforms.Lambda(convert_grayscale_to_binary),

            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomSegmentationDataset(
                self.train_image_paths, self.train_mask_paths, self.transform, self.mask_transform,
                augmentation=get_training_augmentation(self.img_size))
            self.val_dataset = CustomSegmentationDataset(
                self.val_image_paths, self.val_mask_paths, self.transform, self.mask_transform,
                augmentation=get_validation_augmentation(self.img_size))
        if stage == 'test' or stage is None:
            self.test_dataset = CustomSegmentationDataset(
                self.test_image_paths, self.test_mask_paths, self.transform, self.mask_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpu,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_cpu, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.n_cpu,
                          persistent_workers=True)


def pair_images_with_labels(image_dir, label_dir):
    paired_paths = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith(IMAGE_EXTENSIONS):
            img_file_suf = os.path.splitext(image_file)[1]
            image_path = os.path.join(image_dir, image_file)
            for suf in IMAGE_EXTENSIONS:
                label_file = image_file.replace(img_file_suf, suf)
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    paired_paths.append((image_path, label_path))
                    break
            else:
                print(f"Label file missing for {image_file}")
    return paired_paths


def setup_data_module(base_dir, img_size=512, batch_size=8):
    paths = {
        "train_images": f"{base_dir}/train/images",
        "train_labels": f"{base_dir}/train/labels",
        "val_images": f"{base_dir}/val/images",
        "val_labels": f"{base_dir}/val/labels",
        "test_images": f"{base_dir}/val/images",
        "test_labels": f"{base_dir}/val/labels"
    }

    pairs = {key: pair_images_with_labels(paths[key], paths[key.replace("images", "labels")])
             for key in paths if "images" in key}

    train_image_paths, train_mask_paths = zip(*pairs["train_images"])
    val_image_paths, val_mask_paths = zip(*pairs["val_images"])
    test_image_paths, test_mask_paths = zip(*pairs["test_images"])
    return SegDataModule(train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, test_image_paths,
                         test_mask_paths, img_size, batch_size)


def setup_model_and_trainer(arch='unet', encoder_name='resnet34', save_model_every_n_epochs=10, max_epochs=50,
                            checkpoint_callback_dir='model_checkpoints/', learning_rate=1e-4, patience=10,
                            log_dir="tb_logs", name="road network detection", device=1):
    model = SegModel(arch, encoder_name, in_channels=3, out_classes=1, learning_rate=learning_rate)

    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_iou', dirpath=checkpoint_callback_dir,
        filename='best-checkpoint', save_top_k=1, mode='max')

    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=checkpoint_callback_dir, filename='model-{epoch:02d}',
        every_n_epochs=save_model_every_n_epochs, save_top_k=-1)

    early_stop_callback = EarlyStopping(
        monitor='val_iou',  # Metric to monitor
        min_delta=0.00,  # Minimum change in the monitored quantity to qualify as an improvement
        patience=patience,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # If True, prints a message whenever the early stop condition is met
        mode='max'  # 'min' for minimizing the metric, 'max' for maximizing
    )

    logger = TensorBoardLogger(log_dir, name=name)
    return model, L.Trainer(devices=device, logger=logger, max_epochs=max_epochs,
                             callbacks=[checkpoint_callback_best, checkpoint_callback_periodic, early_stop_callback])


def show_images(seg_data_module, trainer, model_path):
    # Load the model from checkpoint
    model = SegModel.load_from_checkpoint(model_path)
    model = model.to('cuda')
    model.eval()  # Set the model to evaluation mode

    # Set up the data module for testing
    seg_data_module.setup(stage='test')

    # Run the validation
    valid_metrics = trainer.test(model, seg_data_module, verbose=False)
    pprint(valid_metrics)

    def visualize_segmentation(image, ground_truth, prediction, ax):
        image_np = image.cpu().permute(1, 2, 0).numpy()  # Assuming image is a torch tensor
        ground_truth_np = ground_truth.cpu().numpy()  # Assuming ground_truth is a torch tensor
        prediction_np = prediction.cpu().numpy()  # Assuming prediction is also a torch tensor
        ax[0].imshow(image_np)
        ax[1].imshow(ground_truth_np.squeeze())  # Use squeeze() in case ground_truth has an extra dimension
        ax[2].imshow(prediction_np.squeeze())  # Use squeeze() in case prediction has an extra dimension

    with torch.no_grad():  # Disable gradient computation
        for batch in seg_data_module.test_dataloader():  # Assuming test_dataloader is set up
            images, ground_truths = batch  # Get images and ground truth masks
            images = images.to(model.device)
            predictions = model(images)
            predictions = predictions.sigmoid()
            threshold = 0.5
            predictions = (predictions > threshold).float()  # Convert to binary (0 and 1)

            # Visualize first image, ground truth, and prediction
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figure size as needed
            visualize_segmentation(images[0], ground_truths[0], predictions[0], ax)
            plt.show()
            plt.close(fig)  # Close the figure

            # break  # Exit after processing the first batch


def save_images(seg_data_module, trainer, model_path, save_path):
    # Load the model from checkpoint
    model = SegModel.load_from_checkpoint(model_path)
    model = model.to('cuda')
    model.eval()  # Set the model to evaluation mode

    # Set up the data module for testing
    seg_data_module.setup(stage='test')

    # Run the validation
    valid_metrics = trainer.test(model, seg_data_module, verbose=False)
    pprint(valid_metrics)

    # Define the directory to save images
    save_dir = save_path
    os.makedirs(save_dir, exist_ok=True)

    def save_segmentation(image, ground_truth, prediction, idx):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        image_np = image.cpu().permute(1, 2, 0).numpy()
        ground_truth_np = ground_truth.cpu().numpy()
        prediction_np = prediction.cpu().numpy()
        ax[0].imshow(image_np)
        ax[1].imshow(ground_truth_np.squeeze())
        ax[2].imshow(prediction_np.squeeze())
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"segmentation_{idx}.png"))
        plt.close(fig)

    with torch.no_grad():
        for i, batch in enumerate(seg_data_module.test_dataloader()):
            images, ground_truths = batch
            images = images.to(model.device)
            predictions = model(images)
            predictions = predictions.sigmoid()
            threshold = 0.5
            predictions = (predictions > threshold).float()

            for j, (image, ground_truth, prediction) in enumerate(zip(images, ground_truths, predictions)):
                save_segmentation(image, ground_truth, prediction, f"batch_{i}_image_{j}")


def save_inference_masks(image_dir, save_dir, model_path, img_size):
    # Load the model from checkpoint
    model = SegModel.load_from_checkpoint(model_path)
    model = model.to('cuda')
    model.eval()  # Set the model to evaluation mode

    # Define the resize transform
    resize_transform = transforms.Resize((img_size, img_size))

    # Create the output directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # List all .jpg, .jpeg, .png, .bmp, .tif, and .tiff images in the directory, case-insensitive
    image_paths = []
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF',
                '*.TIFF']
    for pattern in patterns:
        image_paths.extend(Path(image_dir).rglob(pattern))
    with torch.no_grad():
        for image_path in image_paths:
            # Read the image
            image = Image.open(image_path).convert('RGB')

            # Resize and preprocess the image
            image = resize_transform(image)
            image = TF.to_tensor(image).unsqueeze(0).to(model.device)  # Convert to tensor and add batch dimension
            image = model.image_transform(image)

            # Perform inference
            prediction = model(image)
            prediction = torch.sigmoid(prediction).squeeze().cpu()

            # Convert to binary mask
            threshold = 0.5
            mask = (prediction > threshold).float()
            # Save the mask
            mask_pil = TF.to_pil_image(mask)
            mask_pil = mask_pil.resize((img_size, img_size), resample=Image.NEAREST)
            mask_save_path = Path(save_dir) / image_path.name
            mask_pil.save(mask_save_path)

    print(f"Inference masks saved in {save_dir}")


def main():
    base_dir = '/home/zkxq/project/hjt/smp/datasets/Wind_Turbine'
    # ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan']
    # https://github.com/qubvel/segmentation_models.pytorch
    architecture = 'deeplabv3plus'

    encoder = 'resnet50'
    device = [0, 1, 2, 3]
    save_model_every_n_epochs = 25
    max_epochs = 200
    checkpoint_callback_dir = os.path.join(base_dir, 'model_checkpoints/')
    img_size = 640
    batch_size = 8
    initial_learning_rate = 1e-3
    patience = 20
    log_dir = os.path.join(base_dir, 'tb_logs')
    # tensorboard --logdir=log_dir
    # 添加本地映射命令：ssh -L 6006:localhost:6006 zkxq@192.168.50.6
    name = 'access_road'

    seg_data_module = setup_data_module(base_dir, img_size, batch_size)
    model, trainer = setup_model_and_trainer(architecture, encoder, save_model_every_n_epochs, max_epochs,
                                             checkpoint_callback_dir, initial_learning_rate, patience,
                                             log_dir, name, device)
    tuner = Tuner(trainer)
    # Auto-scale batch size by growing it exponentially (default)
    # tuner.scale_batch_size(model, mode="power")
    tuner.lr_find(model, datamodule=seg_data_module)
    trainer.fit(model, datamodule=seg_data_module)
    valid_metrics = trainer.validate(model, seg_data_module, verbose=False)
    test_metrics = trainer.test(model, seg_data_module, verbose=False)

    pprint(valid_metrics)
    pprint(test_metrics)

    # model_path = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/model_checkpoints/best-checkpoint.ckpt"
    # save_path = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/out"
    # image_dir = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/val/images"
    # # save_inference_masks(image_dir, save_path, model_path, img_size)
    # save_images(seg_data_module, trainer, model_path, save_path)


if __name__ == '__main__':
    # main()
    pass
