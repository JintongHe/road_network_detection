import logging
import os.path
import yaml
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import sys
import json
import inference
import skeleton
import shutil
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')

def get_config():
    config_path = 'config/road_network.yaml'
    config_file = open(config_path, 'r', encoding='utf-8')
    file_info = config_file.read()
    config_dict = yaml.safe_load(file_info)
    return config_dict

def model_predict(config_dict):
    image_paths = config_dict['image_paths']
    split_arr = config_dict['split_arr']
    model_path = config_dict['model_path']
    img_sz = config_dict['image_size']
    gpu_ids = config_dict['gpu_ids']
    out_dir = config_dict['mid_path']
    inference.batch_predict(image_paths, split_arr, model_path, out_dir, gpu_ids, img_sz)

def after_handle(config_dict):
    logging.info('开始生成geojson')
    image_dir = config_dict['image_path']
    mask_dir = config_dict['mid_path']
    out_dir = config_dict['out_dir']
    skeleton.main(image_dir, mask_dir, out_dir)
    shutil.rmtree(mask_dir)
    logging.info('生成geojson完成')

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
    logging.info('开始！！！')
    # start_time = time.time()
    # 1，参数处理
    config_dict = get_config()
    is_test = True
    if is_test:
        image_path = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/test/test_img"
        out_dir = "/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/out"
        # config_dict['out_flag'] = True
    else:
        request_data = ''
        for i in range(len(sys.argv) - 1):
            request_data += sys.argv[i + 1]
        request_data = json.loads(request_data.replace('\'', '\"'))

        image_path = request_data['image_path']
        out_dir = request_data['out_dir']

    # config_dict['start_time'] = start_time
    mid_path = os.path.join(image_path, 'mid_result')
    os.makedirs(mid_path, exist_ok=True)
    config_dict['mid_path'] = mid_path
    config_dict['image_path'] = image_path
    config_dict['image_paths'] = change_img_path(image_path)
    config_dict['out_dir'] = out_dir

    model_predict(config_dict)
    after_handle(config_dict)
    print(f'结果已存储在：{out_dir}')


if __name__ == '__main__':
    main()