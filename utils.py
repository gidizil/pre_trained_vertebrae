import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import pandas as pd

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def run_length_decode(rle, height=1024, width=1024, fill_value=1):
        component = np.zeros((height, width), np.float32)
        component = component.reshape(-1)
        rle = np.array([int(s) for s in rle.strip().split(' ')])
        rle = rle.reshape(-1, 2)
        start = 0
        for index, length in rle:
            start = start + index
            end = start + length
            component[start: end] = fill_value
            start = end
        component = component.reshape(width, height).T
        return component

    @staticmethod
    def run_length_encode(component):
        component = component.T.flatten()
        start = np.where(component[1:] > component[:-1])[0] + 1
        end = np.where(component[:-1] > component[1:])[0] + 1
        length = end - start
        rle = []
        for i in range(len(length)):
            if i == 0:
                rle.extend([start[0], length[0]])
            else:
                rle.extend([start[i] - end[i - 1], length[i]])
        rle = ' '.join([str(r) for r in rle])
        return rle

    @staticmethod
    def extract_2d_images(img_path_3d, msk_path_3d, out_2d_dir_imgs, out_2d_dir_masks, masks):
        """
        Extract relevant 2D slices from a 3D scan
        :param img_path_3d: str. path to original 3D slices
        :param msk_path_3d: str. path to original 3D masks
        :param out_2d_dir_imgs: str. path to output 2D slices of scans
        :param out_2d_dir_masks: str. path to original 2D masks of slices
        :return: None. Saves to 2d slices to out_dir
        """
        img_3d_files = os.listdir(img_path_3d)
        mhd_3d_imgs = [img_file for img_file in img_3d_files if img_file.endswith('.mhd')]
        mhd_3d_imgs.sort()

        if masks:
            msk_3d_files = os.listdir(msk_path_3d)
            mhd_3d_msks = [msk_file for msk_file in msk_3d_files if msk_file.endswith('.mhd')]
            mhd_3d_msks.sort()

        # Dummy data so for loop will work
        if not masks:
            mhd_3d_msks = mhd_3d_imgs

        for img, msk in zip(mhd_3d_imgs, mhd_3d_msks):
            if img == 'image010.mhd':
                a = 1  # Why did I did that?:)

            itk_img = sitk.ReadImage(os.path.join(img_path_3d, img))
            arr_img = sitk.GetArrayFromImage(itk_img)

            if masks:
                itk_msk = sitk.ReadImage(os.path.join(msk_path_3d, msk))
                arr_msk = sitk.GetArrayFromImage(itk_msk)

            img_dims = arr_img.shape
            mid_x = int(img_dims[2] / 2)
            for idx in range(mid_x-20, mid_x+20):
                tmp_2d_img_name = img[0:-4] + '_' + str(idx) + '.jpg'
                if masks:
                    tmp_2d_msk_name = msk[0:-4] + '_' + str(idx) + '.jpg'
                try:
                    slice_img = arr_img[:, :, idx]  #.astype(np.uint8)
                    if masks:
                        slice_msk = arr_msk[:, :, idx]  #.astype(np.uint8)
                except Exception:
                    raise ValueError('Image {0} has dimension problem'.format(img))

                # Make sure it's gray scale
                max_slice_img = np.max(slice_img)
                min_slice_img = np.min(slice_img)

                new_img = (((slice_img - min_slice_img) / (max_slice_img - min_slice_img)) * 255).astype(np.uint8)
                tmp_img = Image.fromarray(new_img)
                save_img_path = os.path.join(out_2d_dir_imgs, tmp_2d_img_name)
                tmp_img.save(save_img_path)

                if masks:
                    tmp_msk = Image.fromarray(slice_msk)
                    save_msk_path = os.path.join(out_2d_dir_masks, tmp_2d_msk_name)
                    tmp_msk.save(save_msk_path)

    @staticmethod
    def gen_self_df(orig_self_df_path, new_imgs_path):
        """
        Generate new DataFrame for the multiple 2D images"
        :param orig_self_df_path:  str. path to the age and gender
        :param new_imgs_path: str. path to the scan slices imgs directory
        :return: None. Generate a .csv files holding age & gender for each slice
        """
        orig_df = pd.read_csv(orig_self_df_path)
        full_imgs_list = os.listdir(new_imgs_path)
        orig_imgs_list = [img.split('_')[0] for img in full_imgs_list]
        data = {'orig_img': orig_imgs_list, 'new_slice': full_imgs_list}
        imgs_df = pd.DataFrame(data)

        # Join to age & gender
        final_df = orig_df.merge(imgs_df, how='inner', left_on='name', right_on='orig_img',)
        final_df['gender'] = np.where(final_df['gender'] == 'F', 1, 0)
        final_df.to_csv('scans_self_data.csv')





