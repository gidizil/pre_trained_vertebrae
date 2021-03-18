import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from collections import OrderedDict

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
        # TODO: Do not take mask if not necessary
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.images_fps.sort()
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
        mask = cv2.imread(self.masks_fps[idx], 0)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # resize it:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
        mask = mask / 255.0

        # Make it in the proper format:
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        # print(mask.shape)
        # print(image.shape)
        # print(mask.squeeze().shape)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # support for image and/ors gender if required
        if self.age or self.gender:  # load dataframe only once
            df = pd.read_csv('scans_self_data.csv')
            patient_id = self.images_fps[idx].split('/')[-1].split('.')[0].split('_')[0]
        if self.age:
            age = [df[df.name == 'image' + patient_id]['age_binned'].tolist()[0]]

        if self.gender:
            gender = [df[df.name == 'image' + patient_id]['gender'].tolist()[0]]

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

    def gen_train_val_dirs(self, train_type, k):
        """

        :param train_type: str. if 'cls_only' can use all train data. else dependent on k.
        :param k: int. number of images to use for segmentation
        :return: None. generate necassery folders
        """
        # TODO: generate the proper train and validation data with folders
        # Images 13,14,15 are constantly test data. 1-12, 16-25 plays
        if train_type == 'cls_only':
            # generate the following data folders:
            # cls_train, cls_validation, seg_train, seg_validation
            pass

        elif train_type == 'segmentation':
            pass
        else:
            raise ValueError("train_type must be 'cls_only' or 'segmentation")



