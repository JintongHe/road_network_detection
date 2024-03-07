import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')
def split_train_val_test(img_path, label_path, val_percentage, test_percentage):
    root_path = os.path.dirname(img_path)
    paths = {
        "train_img": os.path.join(root_path, 'train', 'training_img'),
        "train_label": os.path.join(root_path, 'train', 'training_label'),
        "val_img": os.path.join(root_path, 'val', 'val_img'),
        "val_label": os.path.join(root_path, 'val', 'val_label'),
        "test_img": os.path.join(root_path, 'test', 'test_img'),
        "test_label": os.path.join(root_path, 'test', 'test_label')
    }

    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    img_names = [f for f in os.listdir(img_path) if f.endswith(tuple(IMAGE_EXTENSIONS))]

    # Splitting for test set
    val_img_names = set(random.sample(img_names, int(len(img_names) * val_percentage)))
    remaining_imgs = list(set(img_names) - val_img_names)

    # Splitting for validation set
    test_img_names = set(random.sample(remaining_imgs, int(len(remaining_imgs) * test_percentage)))

    def copy_files(img_name):
        img_name_no_ext = os.path.splitext(img_name)[0]
        old_img_path = os.path.join(img_path, img_name)
        old_label_path = os.path.join(label_path, img_name_no_ext + '.tif')

        if img_name in test_img_names:
            img_folder, label_folder = "test_img", "test_label"
        elif img_name in val_img_names:
            img_folder, label_folder = "val_img", "val_label"
        else:
            img_folder, label_folder = "train_img", "train_label"

        new_img_path = os.path.join(paths[img_folder], img_name)
        new_label_path = os.path.join(paths[label_folder], img_name_no_ext + '.tif')
        shutil.copy2(old_label_path, new_label_path)
        shutil.copy2(old_img_path, new_img_path)

    with ThreadPoolExecutor() as executor:
        executor.map(copy_files, img_names)

if __name__ == '__main__':
    img_path = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/out_image"
    label_path = "D:/Code/Datasets/wind_turbine/dataset20240202/access_road/out_mask"
    split_train_val_test(img_path, label_path, 0.1, 0.1)