from utils import Utils
import os
import numpy as np
import cv2
from PIL import Image
import shutil
import pandas as pd

# Set what you want to do:
get_scan_slices = False
masks = False
gen_df = False
connected_comps = False
gen_train_val = False
bin_age = True

# 1. Generating 2D slices from 3D scans
if get_scan_slices:
    data_path_3d = '/Users/gzilbar/msc/courses/medical_dl/project/vertebrae/xVertSeg/xVertSeg-1.v1/Data2'
    img_path_3d = os.path.join(data_path_3d, 'images')
    out_2d_dir_imgs = 'images'

    msk_path_3d = os.path.join(data_path_3d, 'masks')
    out_2d_dir_masks = 'masks'

    Utils.extract_2d_images(img_path_3d, msk_path_3d, out_2d_dir_imgs, out_2d_dir_masks, masks=masks)

# 2. Generating dataframe to match age, gender to the new slices
if gen_df:
    orig_self_data_df = 'self_data.csv'
    new_imgs_path = 'images'
    Utils.gen_self_df(orig_self_data_df, new_imgs_path)

# 3. Applying connected components to fix the pixel labeling problem
# TODO: Apply this to all images - and pass them to adele
if connected_comps:
    msk_imgs = os.listdir('masks')
    for msk_file_name in msk_imgs:
        # img = cv2.imread(os.path.join('masks', msk_file_name), 0)  # 0 means read grayscale
        # out_msk = np.zeros(shape=img.shape, dtype=np.uint8)
        # thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        # num_labels, labels_img = cv2.connectedComponents(thresh_img, connectivity=4)
        # for label in range(1, 11):
        #     out_msk[labels_img == label] = np.mean(img[labels_img == label])

        # msk = cv2.imread(os.path.join('masks', msk_file_name), 0)  # 0 means read grayscale
        # out_msk = msk.copy()
        # out_msk[msk <= 195] = 0
        # ranges = [(100, 205), (205, 215), (215, 225), (225, 235), (235, 256)]
        # for strt, end in ranges:
        #   out_msk[(out_msk >= strt) & (out_msk < end)] = int((strt + end) / 2)

        """Decided to go for a binary mask!!"""
        msk = cv2.imread(os.path.join('masks', msk_file_name), 0)  # 0 means read grayscale
        thresh_msk = cv2.threshold(msk, 150, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        thresh_msk = Image.fromarray(thresh_msk)
        # new_msk_arr = (((thresh_msk - min_val) / (thresh_msk - min_val)) * 255).astype(np.uint8)
        msk_save_path = os.path.join('binary_masks', msk_file_name)
        thresh_msk.save(msk_save_path)

        # Sanity check:
        # if len(np.unique(out_msk)) != 6:
        #     print(msk_file_name, np.unique(out_msk), len(np.unique(out_msk)))
        # min_val, max_val = np.min(out_msk), np.max(out_msk)
        # new_msk_arr = (((out_msk - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
        # new_msk = Image.fromarray(new_msk_arr)
        # msk_save_path = os.path.join('fixed_masks', msk_file_name)
        # new_msk.save(msk_save_path)

# 4. Create train and validation from mid-slices only
if gen_train_val:
    patients_num_1 = ['00' + str(idx) for idx in range(1, 10)]
    patients_num_2 = ['0' + str(idx) for idx in range(10, 16)]
    patients_num = patients_num_1 + patients_num_2

    image_names = os.listdir('images')
    mid_images_list = []
    for patient_num in patients_num:
        # 1. Find middle slice:
        patient_images = [int(img.split('_')[1].split('.')[0]) for img in image_names if img.startswith(patient_num)]
        mid_slice = np.round(np.median(patient_images)).astype('int')

        # 2. Get middle image name
        mid_img_name = patient_num + '_' + str(mid_slice) + '.jpg'
        mid_images_list.append(mid_img_name)

    # 3. move them to train and validation
    try:
        os.mkdir('mid_img_train')
        os.mkdir('mid_msk_train')
        os.mkdir('mid_img_val')
        os.mkdir('mid_msk_val')
    except FileExistsError:
        pass

    np.random.shuffle(mid_images_list)
    for idx, img_name in enumerate(mid_images_list):
        if idx < 3:
            shutil.copyfile(os.path.join('images', img_name), os.path.join('mid_img_val', img_name))
            shutil.copyfile(os.path.join('binary_masks', img_name), os.path.join('mid_msk_val', img_name))
        else:
            shutil.copyfile(os.path.join('images', img_name), os.path.join('mid_img_train', img_name))
            shutil.copyfile(os.path.join('binary_masks', img_name), os.path.join('mid_msk_train', img_name))

if bin_age:
    df = pd.read_csv('scans_self_data.csv')

    def age_binning(row):
        if row['age'] <= 50:
            return 0
        elif 50 < row['age'] <= 60:
            return 1
        elif 60 < row['age'] <= 70:
            return 2
        elif 70 < row['age'] <= 80:
            return 3
        elif 80 < row['age']:
            return 4

    df['age_binned'] = df.apply(age_binning, axis=1)
    df.to_csv('scans_self_data.csv')

