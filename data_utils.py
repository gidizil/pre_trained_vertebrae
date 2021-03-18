import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from collections import OrderedDict
import shutil
import numpy as np


class VertebraeDataset(Dataset):
    """Dataloader that can used for both segmentation ans SSL training"""

    def __init__(self, img_path, msk_path=None, self_data_path=None,
                 self_data_type=None, img_transforms=None, msk_transforms=None):
        self.img_path = img_path
        self.msk_path = msk_path
        self.img_transforms = img_transforms
        self.msk_transforms = msk_transforms
        self.self_data_path = self_data_path
        self.self_data_type = self_data_type

        self.imgs_list = os.listdir(self.img_path)
        self.imgs_list.sort()

        self.msks_list = None
        self.self_df = None
        self.flags_dict = {'mask': False, 'age': False, 'gender': False}
        self.values_dict = {'mask': None, 'age': None, 'gender': None}

        if isinstance(msk_path, str):
            self.msks_list = os.listdir(msk_path)
            self.msks_list.sort()

        # self_data can be: None, 'gender', 'age', 'both'
        if isinstance(self.self_data_path, str):
            self.self_df = pd.read_csv(self.self_data_path)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        img_pil = Image.open(os.path.join(self.img_path, img_name))
        img_arr = np.array(img_pil)

        if self.img_transforms is not None:
            img = self.img_transforms(img_pil)
        else:
            img = torch.Tensor(img_arr)

        # When using Masks
        if self.msks_list is not None:
            msk_name = self.msks_list[idx]
            msk_pil = Image.open(os.path.join(self.msk_path, msk_name))
            msk_arr = np.array(msk_pil)

            if self.msk_transforms is not None:
                msk = self.msk_transforms(msk_pil)
            else:
                msk = torch.Tensor(msk_arr)

            self.flags_dict['mask'] = True
            self.values_dict['mask'] = msk

        # When using age or gender or both
        if self.self_df:
            self.self_df.sort_values('new_slice', axis='index')
            if self.self_data_type == 'age':
                age_list = self.self_df.age.values.tolist()
                age = torch.Tensor([age_list[idx]])
                self.values_dict['age'] = age

                self.flags_dict['age'] = True
            elif self.self_data_type == 'gender':
                gender_list = self.self_df.gender.values.tolist()
                gender = torch.Tensor([gender_list[idx]])
                self.values_dict['gender'] = gender

                self.flags_dict['gender'] = True
            elif self.self_data_type == 'age_gender':
                age_list = self.self_df.age.values.tolist()
                gender_list = self.self_df.gender.values.tolist()

                age = torch.Tensor([age_list[idx]])
                gender = torch.Tensor([gender_list[idx]])

                self.values_dict['age'] = age
                self.values_dict['gender'] = gender

                self.flags_dict['age'] = True
                self.flags_dict['gender'] = True
            elif self.self_data_type is None:
                pass
            else:
                raise ValueError("Invalid input. Should be 'age', 'gender' or 'age_gender'")

        out = (img, )
        for key in self.flags_dict.keys():
            if self.flags_dict[key]:
                out += (self.values_dict[key][0], )

        return out

class VertebraeDataset2(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']

    CLASSES = ['verte']

    def __init__(
            self,
            images_dir,
            masks_dir,
            age,
            gender,
            decoder,
            classes=None,
            augmentation=None,
            preprocessing=None,

    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.images_fps.sort()
        self.masks_dir = masks_dir
        if self.masks_dir is not None:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
            self.masks_fps.sort()

        self.age = age
        self.gender = gender
        self.decoder = decoder

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):

        # read data
        image = cv2.imread(self.images_fps[idx], 0)  # '0' for grayscale
        if self.masks_dir is not None:
            mask = cv2.imread(self.masks_fps[idx], 0)
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # resize it:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
        if self.masks_dir is not None:
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
            mask = mask / 255.0

        # Make it in the proper format:
        image = np.expand_dims(image, 0)
        if self.masks_dir is not None:
            mask = np.expand_dims(mask, 0)

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        # print(mask.shape)
        # print(image.shape)
        # print(mask.squeeze().shape)

        # apply augmentations
        # TODO: Not active at the moment
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        # TODO: Not active at the moment
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # support for image and/ors gender if required
        if self.age or self.gender:  # load dataframe only once
            df = pd.read_csv('scans_self_data.csv')
            patient_id = self.images_fps[idx].split('/')[-1].split('.')[0].split('_')[0]
        if self.age:
            age = [df[df.name == patient_id]['age_binned'].tolist()[0]]

        if self.gender:
            gender = [df[df.name == patient_id]['gender'].tolist()[0]]

        # TODO: Make this nicer using flags dict
        if self.gender and self.age and self.decoder:
            return torch.from_numpy(image).float(), torch.from_numpy(mask).float(), \
                   torch.Tensor(gender).long(), torch.Tensor(age).long()
        elif self.gender and self.age and not self.decoder:
            return torch.from_numpy(image).float(), torch.Tensor(gender).long(), torch.Tensor(age).long()
        elif self.gender and not self.age:
            return torch.from_numpy(image).float(), torch.Tensor(gender).float()
        elif self.age and not self.gender:
            return torch.from_numpy(image).float(), torch.Tensor(age)
        elif not(self.age or self.gender):
            return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.ids)


# Check that VertebraeDataset works works
if False:
    test = VertebraeDataset('images', 'masks')
    data_test = DataLoader(test, batch_size=4)

    batch = next(iter(data_test))
    print('a')
    # /Users/gzilbar/Library/Caches/PyCharm2019.2/remote_sources/-201990402/-997824191/segmentation_models_pytorch/unet/decoder.py

if False:
    CLASSES = ['verte']
    test = VertebraeDataset2('mid_img_train', 'mid_msk_train',
                             age=False, gender=True, classes=CLASSES, )
    data_test = DataLoader(test, batch_size=4)

    batch = next(iter(data_test))
    print('a')

class Utils:
    """Misc class that holds utility data functions"""
    def __init__(self):
        pass

    @staticmethod
    def update_state_dict(state_dict, params_type=None):
        if not isinstance(state_dict, dict):
            raise ValueError('state_dict must be an OrderedDict')

        encoder_weights = state_dict.copy()
        if params_type not in ('cls', 'multi', 'dec_multi'):
            raise ValueError("params_type must be 'cls' or 'multi'")
        elif params_type == 'cls':
            custom_enc_weights = OrderedDict(
                [(k.replace('module.encoder.', ''), v) for k, v in encoder_weights.items()])
            custom_enc_weights = OrderedDict(
                [(k.replace('module.cls.3', 'fc'), v) for k, v in custom_enc_weights.items()])

        elif params_type == 'multi':
            custom_enc_weights = OrderedDict([(k.replace('encoder.', ''), v) for k, v in encoder_weights.items()])
            custom_enc_weights = OrderedDict(
                [(k.replace('cls_gender.3', 'fc'), v) for k, v in custom_enc_weights.items()])

        elif params_type == 'dec_multi':
            custom_enc_weights = OrderedDict([(k.replace('encoder.', ''), v) for k, v in encoder_weights.items()])
            custom_enc_weights = OrderedDict([(k.replace('cls_gender.3', 'fc'), v) for k, v in custom_enc_weights.items()])
            custom_enc_weights.pop('cls_age.3.weight')
            custom_enc_weights.pop('cls_age.3.bias')
            custom_enc_weights.pop('seg_head.0.weight')
            custom_enc_weights.pop('seg_head.0.bias')

            dict_keys = list(encoder_weights)
            for k in dict_keys:
                if k.startswith('decoder'):
                    custom_enc_weights.pop(k)

        return custom_enc_weights

    @staticmethod
    def gen_train_val_test_dirs(train_type, k=None):
        """
        :param train_type: str. if 'cls_only' can use all train data. else dependent on k.
        :param k: int. number of images to use for segmentation
        :return: None. generate necassery folders
        """
        # Images 13,14,15 are constantly test data. 1-12, 16-25 plays
        TEST_IMAGES = ('image013', 'image014', 'image015')
        TEST_MASKS = ('mask013', 'mask014', 'mask015')
        # 1. Set Test data to always evaluate against - keep images 13,14,15 (40 slices each) for test
        if not os.path.exists('test_images'):
            os.makedirs('test_images')
            for img in os.listdir('images'):
                if img.startswith(TEST_IMAGES):
                    src = 'images/' + img
                    dst = 'test_images/' + img
                    shutil.copyfile(src, dst)

        if not os.path.exists('test_masks'):
            os.makedirs('test_masks')
            for msk in os.listdir('binary_masks'):
                if msk.startswith(TEST_MASKS):
                    src = 'binary_masks/' + msk
                    dst = 'test_masks/' + msk
                    shutil.copyfile(src, dst)

        # TODO: generate the proper train and validation data with folders
        if not os.path.exists('train_images'):
            os.makedirs('train_images')
            if train_type == 'segmentation':
                os.makedirs('train_masks')
        else:
            raise Exception('train_images directory already exists. please delete it')

        if not os.path.exists('val_images'):
            os.makedirs('val_images')
            if train_type == 'segmentation':
                os.makedirs('val_masks')
        else:
            raise Exception('val_images directory already exists. please delete it')

        if train_type == 'cls_only':
            # Randomly split images to train and validation
            img_strt_names = ['image00' + str(idx) for idx in range(1, 10)] + \
                             ['image0' + str(idx) for idx in range(10, 13)] + \
                             ['image0' + str(idx) for idx in range(16, 26)]

            img_strt_names = np.array(img_strt_names)  # To extract multiple indices
            images_indices = range(len(img_strt_names))
            train_indices = np.random.choice(images_indices, size=17, replace=False)
            train_img_strt = img_strt_names[train_indices]
            val_img_strt = np.delete(img_strt_names, train_indices)

            cls_images = [img_name for img_name in os.listdir('images')
                          if not img_name.startswith(TEST_IMAGES)]
            for img in cls_images:
                if img.startswith(tuple(train_img_strt)):
                    src = 'images/' + img
                    dst = 'train_images/' + img
                    shutil.copyfile(src, dst)
                elif img.startswith(tuple(val_img_strt)):
                    src = 'images/' + img
                    dst = 'val_images/' + img
                    shutil.copyfile(src, dst)
                else:
                    raise ValueError('img name - {} is deformed'.format(img))

        elif train_type == 'segmentation':
            img_strt_names = ['image00' + str(idx) for idx in range(1, 10)] + \
                             ['image0' + str(idx) for idx in range(10, 13)]

            msk_strt_names = ['mask00' + str(idx) for idx in range(1, 10)] + \
                             ['mask0' + str(idx) for idx in range(10, 13)]

            img_strt_names = np.array(img_strt_names)  # To extract multiple indices
            msk_strt_names = np.array(msk_strt_names)
            image_indices = range(len(img_strt_names))
            train_indices = np.random.choice(image_indices, size=k, replace=False)
            train_img_strt = img_strt_names[train_indices]
            val_img_strt = np.delete(img_strt_names, train_indices)

            train_msk_strt = msk_strt_names[train_indices]
            val_msk_strt = np.delete(msk_strt_names, train_indices)

            seg_images = [img_name for img_name in os.listdir('images')
                          if img_name.startswith(tuple(img_strt_names))]
            seg_masks = [msk_name for msk_name in os.listdir('binary_masks')
                         if msk_name.startswith(tuple(msk_strt_names))]

            for img in seg_images:
                if img.startswith(tuple(train_img_strt)):
                    src = 'images/' + img
                    dst = 'train_images/' + img
                    shutil.copyfile(src, dst)
                elif img.startswith(tuple(val_img_strt)):
                    src = 'images/' + img
                    dst = 'val_images/' + img
                    shutil.copyfile(src, dst)
                else:
                    raise ValueError('img name - {} is deformed'.format(img))

            for msk in seg_masks:
                if msk.startswith(tuple(train_msk_strt)):
                    src = 'binary_masks/' + msk
                    dst = 'train_masks/' + msk
                    shutil.copyfile(src, dst)
                elif msk.startswith(tuple(val_msk_strt)):
                    src = 'binary_masks/' + msk
                    dst = 'val_masks/' + msk
                    shutil.copyfile(src, dst)
                else:
                    raise ValueError('msk name - {} is deformed'.format(msk))

        else:
            raise ValueError("train_type must be 'cls_only' or 'segmentation")

    @staticmethod
    def rm_train_val_dirs(train_type):
        """
        remove training and validation folders at the end of training
        :param train_type: str. 'cls_only' or 'segmentation
        :return: None. Removes the train and validation data directories after training
        """
        if os.path.exists('train_images'):
            shutil.rmtree('train_images')
        if os.path.exists('val_images'):
            shutil.rmtree('val_images')
        # TODO: probably unnecessary - consider removing
        if train_type == 'cls_only':
            pass
        elif train_type == 'segmentation':
            if os.path.exists('train_masks'):
                shutil.rmtree('train_masks')
            if os.path.exists('val_masks'):
                shutil.rmtree('val_masks')
        else:
            raise ValueError("train_type should be 'cls_only' or 'segmentation'")

