import geopandas as gpd
from osgeo import gdal
import subprocess
import numpy as np
import cv2
import math
import os
import rasterio
from rasterio.enums import Resampling
import shutil
from shapely.affinity import scale

def convert_to_8Bit(inputRaster, outputRaster,
                    outputPixType='Byte',
                    outputFormat='GTiff',
                    rescale_type='rescale',
                    percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit.
    rescale_type = ['clip', 'rescale']
    '''
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat]

    for bandId in range(srcRaster.RasterCount):
        band = srcRaster.GetRasterBand(bandId + 1)
        if rescale_type == 'rescale':
            # Compute percentiles for rescaling
            band_arr = band.ReadAsArray()
            bmin = np.percentile(band_arr.flatten(), percentiles[0])
            bmax = np.percentile(band_arr.flatten(), percentiles[1])
        else:
            # Use the full 16-bit range
            bmin, bmax = 0, 65535

        cmd.extend(['-scale_{}'.format(bandId + 1), str(bmin), str(bmax), '0', '255'])

    cmd.extend([inputRaster, outputRaster])
    print("Conversion command:", ' '.join(cmd))
    subprocess.call(cmd)

def apply_clahe_to_rgb_image(bgr_image, clip_limit=3, tile_grid_size=(4, 4)):
    # Convert the BGR image to Lab color space
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    bgr_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2BGR)
    return bgr_image_clahe

def process_image_with_rasterio(input_path, output_path):
    with rasterio.open(input_path) as src:
        # Read the image data and metadata
        img_data = src.read([1, 2, 3], resampling=Resampling.bilinear) # Reading RGB bands
        transform = src.transform
        crs = src.crs
        img_data = np.moveaxis(img_data, 0, -1)  # Move the channels to the last dimension for OpenCV
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV processing

        # Apply CLAHE
        enhanced_img = apply_clahe_to_rgb_image(img_data)

        # Convert back to RGB and to rasterio's channel-first format
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        enhanced_img = np.moveaxis(enhanced_img, -1, 0)  # Move channels back to first dimension

        # Write the image using the original's metadata
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=enhanced_img.shape[1],
            width=enhanced_img.shape[2],
            count=3,
            dtype=enhanced_img.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(enhanced_img)
def linestring_to_mask(img_path, geojson_path, out_dir, line_width=2):
    # Function to convert LineString to pixel coordinates
    def linestring_to_pixels(geometry, image_shape, geoTransform):
        def transform_coords(linestring):
            # x_min, y_min, x_max, y_max = bounds
            # x_scale = image_shape[1] / (x_max - x_min)
            # y_scale = image_shape[0] / (y_max - y_min)

            coords = np.array(linestring.coords)
            coords[:, 0] = (coords[:, 0] - geoTransform[0]) / geoTransform[1]
            coords[:, 1] = (coords[:, 1] - geoTransform[3]) / geoTransform[5]
            # coords[:, 1] = image_shape[0] - coords[:, 1]  # Flip y-axis

            return np.array(coords, dtype=np.int32)

        if geometry.type == 'MultiLineString':
            return [transform_coords(part) for part in geometry.geoms]
        else:
            return [transform_coords(geometry)]

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
    road_width_meters = line_width
    meters_per_degree = 111139 * math.cos(math.radians(latitude))
    thickness_pixels = int(road_width_meters / (pixel_size_x * meters_per_degree))
    # Load GeoJSON file
    gdf = gpd.read_file(geojson_path)
    gdf_projected = gdf.to_crs("EPSG:4326")  # Make sure it matches the TIFF CRS

    # Get bounds of the projected GeoDataFrame
    bounds = gdf_projected.total_bounds

    # Define the size of your mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw each LineString on the mask
    for geom in gdf_projected['geometry']:
        pixel_arrays = linestring_to_pixels(geom, mask.shape, geoTransform)
        for pixels in pixel_arrays:
            cv2.polylines(mask, [pixels], isClosed=False, color=1, thickness=thickness_pixels)

    img_name = os.path.splitext(img_path.split('\\')[-1])[0]
    out_file_path = os.path.join(out_dir, img_name) + '.tif'
    # Save or use the mask
    cv2.imwrite(out_file_path, mask * 255)

def linestring_to_mask_batch(img_dir, geojson_dir, line_width=2):
    root_dir = os.path.dirname(img_dir)
    out_dir = os.path.join(root_dir, 'out_mask')
    out_img_dir = os.path.join(root_dir, 'out_image')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    img_names_suf = os.listdir(img_dir)
    for img_name_suf in img_names_suf:
        if not img_name_suf.endswith('.tif'):
            print(f'error, wrong image type with {img_name_suf}')
            continue
        img_name = os.path.splitext(img_name_suf)[0]
        img_path = os.path.join(img_dir, img_name_suf)
        # geojson_name = '_'.join(img_name.split('_')[:6] + ['geojson_roads'] + [img_name.split('_')[-1]])
        geojson_name = img_name
        geojson_path = os.path.join(geojson_dir, geojson_name+'.geojson')
        if not os.path.exists(geojson_path):
            print(f'{img_name_suf} has no corresponding labels!')
            continue
        # shutil.copy2(img_path, out_img_dir)
        # convert_to_8Bit(img_path, os.path.join(out_img_dir, img_name_suf))
        process_image_with_rasterio(img_path, os.path.join(out_img_dir, img_name_suf))
        linestring_to_mask(img_path, geojson_path, out_dir, line_width)


if __name__ == '__main__':
    img_dir = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/images"
    geojson_dir = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/label_txt"
    linestring_to_mask_batch(img_dir, geojson_dir, line_width=4)
