import os
import cv2
import numpy as np

def draw_mask_on_image(image, mask, mask_color=(0, 255, 0), alpha=0.5):
    mask = mask.astype(bool)
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask] = mask_color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def process_images(images_folder, masks_folder, output_folder, mask_color=(0, 255, 0), alpha=0.5):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)  # Assuming mask has same filename as image
        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image is not None and mask is not None:
                # Draw mask on image
                result = draw_mask_on_image(image, mask, mask_color, alpha)

                # Save the result
                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, result)
if __name__ == '__main__':
    # Example usage
    images_folder = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/train/split_images"
    masks_folder = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/train/split_masks"
    output_folder = os.path.join(os.path.dirname(masks_folder), 'out')

    process_images(images_folder, masks_folder, output_folder)
