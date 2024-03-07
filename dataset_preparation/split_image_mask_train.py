import os
import numpy as np
import gdal
import cv2
import math
from shapely.geometry import box, Polygon, MultiPolygon
from multiprocessing import Pool

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')


def resize_and_pad_image(img, split_size, pad_color=(255, 255, 255), new_size=1024):
    h, w = img.shape[:2]

    # Determine the scaling factor and resize
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a new image with the desired size and white background
    # padded_img = np.full((split_size[0], split_size[1], 3), pad_color, dtype=np.uint8)
    padded_img = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)
    # Place the resized image onto the center of the square image
    padded_img[:new_h, :new_w, :] = resized_img

    return padded_img

def resize_and_pad_mask(mask, split_size, pad_value=0, new_size=1024):
    h, w = mask.shape[:2]
    # Determine the scaling factor and resize
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Use INTER_NEAREST for masks

    # Create a new mask with the desired size and default background value
    # padded_mask = np.full((split_size[0], split_size[1]), pad_value, dtype=mask.dtype)
    padded_mask = np.full((new_size, new_size), pad_value, dtype=mask.dtype)
    # Place the resized mask onto the top-left of the square mask
    padded_mask[:new_h, :new_w] = resized_mask

    return padded_mask

def process_image_segment_v1(filename, IMAGE_FOLDER, split_sizes, training_image_folder_path,
                                 training_mask_folder_path, MASK_FOLDER, num_bands, input_img_size):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = gdal.Open(image_path)
    basename = os.path.splitext(filename)[0]
    mask_path = os.path.join(MASK_FOLDER, basename + '.tif')
    if not os.path.exists(mask_path):
        print(f"{mask_path} does not exist!")
        return
    mask = gdal.Open(mask_path)
    width, height = image.RasterXSize, image.RasterYSize
    ratio = pixel_2_meter(image_path)
    # Image segmentation
    for split_size in split_sizes:
        split_size = [int(i * ratio) for i in split_size]
        cut_width, cut_height = split_size
        for w in range(0, width, cut_width):
            for h in range(0, height, cut_height):
                w0, h0 = w, h
                if w0 >= width or h0 >= height:
                    continue
                # w1, h1 = min(w0 + cut_width, width), min(h0 + cut_height, height)
                w1, h1 = w0 + cut_width, h0 + cut_height
                if w1 > width or h1 > height:
                    continue
                local_image_width, local_image_height = w1 - w0, h1 - h0
                cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}.png"

                mask_file_path = os.path.join(training_mask_folder_path, os.path.splitext(cropped_filename)[0] + '.png')
                if not os.path.exists(mask_file_path):
                    # Read the mask as a single-channel grayscale image
                    mask_array = mask.GetRasterBand(1).ReadAsArray(w0, h0, local_image_width, local_image_height)

                    # Check if there are any non-zero pixels in the mask
                    if len(mask_array[mask_array > 0]) < 10000:
                        continue
                    mask_array = resize_and_pad_mask(mask_array, split_size, pad_value=0, new_size=input_img_size)
                    # Write the mask to a file
                    cv2.imwrite(mask_file_path, mask_array)

                training_image_file_path = os.path.join(training_image_folder_path, cropped_filename)
                if not os.path.exists(training_image_file_path):
                    data = [image.GetRasterBand(band + 1).ReadAsArray(w0, h0, local_image_width,
                                                                      local_image_height) for band in
                            range(num_bands)]
                    pic = np.dstack(data[::-1])  # Assuming RGB order
                    pic = resize_and_pad_image(pic, split_size, pad_color=(255, 255, 255), new_size=input_img_size)
                    cv2.imwrite(training_image_file_path, pic)

def process_image_segment_v2(filename, IMAGE_FOLDER, split_sizes, training_image_folder_path,
                                 training_mask_folder_path, MASK_FOLDER, num_bands, input_img_size):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = gdal.Open(image_path)
    basename = os.path.splitext(filename)[0]
    mask_path = os.path.join(MASK_FOLDER, basename + '.tif')
    if not os.path.exists(mask_path):
        print(f"{mask_path} does not exist!")
        return
    mask = gdal.Open(mask_path)
    width, height = image.RasterXSize, image.RasterYSize
    ratio = pixel_2_meter(image_path)

    # Image segmentation
    for split_size in split_sizes:
        split_size = [int(i * ratio) for i in split_size]
        cut_width, cut_height = split_size
        for w in range(0, width, cut_width):
            for h in range(0, height, cut_height):
                for dx in [0, int(cut_width / 2)]:
                    for dy in [0, int(cut_height / 2)]:
                        w0, h0 = w + dx, h + dy
                        if w0 >= width or h0 >= height:
                            continue
                        # w1, h1 = min(w0 + cut_width, width), min(h0 + cut_height, height)
                        w1, h1 = w0 + cut_width, h0 + cut_height
                        if w1 > width or h1 > height:
                            continue
                        local_image_width, local_image_height = w1 - w0, h1 - h0
                        cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}_{dx // int(cut_width / 2)}_{dy // int(cut_height / 2)}.png"

                        mask_file_path = os.path.join(training_mask_folder_path, os.path.splitext(cropped_filename)[0] + '.png')
                        if not os.path.exists(mask_file_path):
                            # Read the mask as a single-channel grayscale image
                            mask_array = mask.GetRasterBand(1).ReadAsArray(w0, h0, local_image_width, local_image_height)

                            # Check if there are any non-zero pixels in the mask
                            if len(mask_array[mask_array > 0]) < 1000:
                                continue
                            mask_array = resize_and_pad_mask(mask_array, split_size, pad_value=0, new_size=input_img_size)
                            # Write the mask to a file
                            cv2.imwrite(mask_file_path, mask_array)

                        training_image_file_path = os.path.join(training_image_folder_path, cropped_filename)
                        if not os.path.exists(training_image_file_path):
                            data = [image.GetRasterBand(band + 1).ReadAsArray(w0, h0, local_image_width,
                                                                              local_image_height) for band in
                                    range(num_bands)]
                            pic = np.dstack(data[::-1])  # Assuming RGB order
                            pic = resize_and_pad_image(pic, split_size, pad_color=(255, 255, 255), new_size=input_img_size)
                            cv2.imwrite(training_image_file_path, pic)


def split_images_segment_v1(IMAGE_FOLDER, split_size, LABEL_FOLDER, input_img_size):
    training_image_folder_path = os.path.join(os.path.dirname(IMAGE_FOLDER), 'split_images')
    training_mask_folder_path = os.path.join(os.path.dirname(IMAGE_FOLDER), 'split_masks')
    os.makedirs(training_image_folder_path, exist_ok=True)
    os.makedirs(training_mask_folder_path, exist_ok=True)

    filenames = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(IMAGE_EXTENSIONS)]
    num_bands = gdal.Open(
        os.path.join(IMAGE_FOLDER, filenames[0])).RasterCount if filenames else 3  # Default to 3 if no files


    args = [(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_mask_folder_path,
             LABEL_FOLDER, num_bands, input_img_size)
            for filename in filenames]

    with Pool() as pool:
        pool.starmap(process_image_segment_v1, args)

    print('Image segmentation completed!')

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

def main():
    split_sizes = [[500, 500], [700, 700], [900, 900]]
    images_folder_path = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/train/training_img"
    masks_folder_path = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/train/training_label"
    input_img_size = 1024
    split_images_segment_v1(images_folder_path, split_sizes, masks_folder_path, input_img_size)


if __name__ == '__main__':
    main()
