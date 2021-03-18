import torch
from train_utils import UNetModel, UnetEncoderClassiferModel, TrainEncoderClassifier
from train_utils import UnetEncoderMultiTaskModel, TrainEncoderMultiTask
from train_utils import UnetEncoderDecoderMultiTaskModel, TrainEncoderDecoderMultiTask
from data_utils import VertebraeDataset, VertebraeDataset2, Utils as data_utils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import segmentation_models_pytorch as smp

"""General Preps Configuration"""
# 0. Set what you want to train:
train_plain_unet = False
plain_unet_weights = None  # Can be None or 'imagenet'

train_unet_cls = False
train_unet_multi = False

set_new_encoder = True
cls_encoder = False
multi_encoder = False
dec_multi_encoder = True

train_unet_multi_w_decoder = False

gender = False
age = False
decoder = False

# 1. Device:
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Data:
# TODO: Support for age and gender - currently manually changed according to needs
train_data = VertebraeDataset2(images_dir='mid_img_train', masks_dir='mid_msk_train',
                               age=age, gender=gender, decoder=decoder, classes=["verte"])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)

val_data = VertebraeDataset2(images_dir='mid_img_val', masks_dir='mid_msk_val',
                             age=age, gender=gender, decoder=decoder, classes=["verte"])
val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)


""" 1. Train Unet Model and eval after each epoch """
if train_plain_unet:
    unet_model = UNetModel(classes=1, mode='train', device=device, loss='dice',
                           encoder_weights=plain_unet_weights, encoder_name='resnet34')

    unet_model.set_epoch(mode='train')
    train_epoch = unet_model.epoch
    unet_model.set_epoch(mode='eval')
    val_epoch = unet_model.epoch
    for idx in range(500):
        print('\nEpoch: {}'.format(idx))
        train_logs = train_epoch.run(train_loader)
        val_logs = val_epoch.run(val_loader)


""" 2. Train UNet encoder with cls head for gender detection"""
if train_unet_cls:
    # 1. Define Model and pass to GPU:
    model = UnetEncoderClassiferModel(cls_params=None, encoder_params=None)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 2. Train and eval
    tr_encoder_cls = TrainEncoderClassifier(model_name='best_cls_params.pth')
    tr_encoder_cls.set_model(model)
    params_dict = {'num_epochs': 50, 'lr': 0.0005, 'optimizer': 'adam'}
    tr_encoder_cls.train_and_eval(train_loader, val_loader, params_dict=params_dict)

"""2. Train UNet encoder with multi-task head for gender&age detection"""
if train_unet_multi:
    # 1. Define Model and pass it to GPU
    model = UnetEncoderMultiTaskModel()
    # model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Train and Eval
    tr_encoder_multi = TrainEncoderMultiTask(model_name='best_multi_params.pth')
    tr_encoder_multi.set_model(model)
    params_dict = {'num_epochs': 250, 'lr': 0.0005, 'optimizer': 'adam', 'criterion': 'ce'}
    tr_encoder_multi.train_and_eval(train_loader, val_loader, params_dict=params_dict)

if train_unet_multi_w_decoder:
    # 1. Define model and pass it to GPU
    model = UnetEncoderDecoderMultiTaskModel()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 2. Train and Eval
    tr_enc_dec_multi = TrainEncoderDecoderMultiTask(model_name='best_dec_multi_params.pth')
    tr_enc_dec_multi.set_model(model)
    params_dict = {'cls_criterion': 'ce', 'seg_criterion': 'dice', 'num_epochs': 250, 'lr': 0.0005}
    tr_enc_dec_multi.train_and_eval(train_loader, val_loader, params_dict=params_dict)

""" 4. Use encoder results from previous pre-training"""
if set_new_encoder:
    # Adjust encoder weights
    if cls_encoder:
        encoder_cls_params = torch.load('best_cls_params.pth')
        custom_encoder_weights = data_utils.update_state_dict(encoder_cls_params, params_type='cls')
    elif multi_encoder:
        encoder_multi_params = torch.load('best_multi_params.pth')
        custom_encoder_weights = data_utils.update_state_dict(encoder_multi_params, params_type='multi')
    elif dec_multi_encoder:
        encoder_dec_multi_params = torch.load('best_dec_multi_params.pth')
        custom_encoder_weights = data_utils.update_state_dict(encoder_dec_multi_params, params_type='dec_multi')

    # Plug encoder weights to decoder.
    unet_model = UNetModel(classes=1, mode='train', device=device, loss='dice',
                           encoder_weights=None, encoder_name='resnet34',
                           custom_encoder_weights=custom_encoder_weights)

    # Train and evaluate:
    # TODO: consider encapsulate this into a method in UnetModel
    unet_model.set_epoch(mode='train')
    train_epoch = unet_model.epoch
    unet_model.set_epoch(mode='eval')
    val_epoch = unet_model.epoch
    for idx in range(500):
        print('\nEpoch: {}'.format(idx))
        train_logs = train_epoch.run(train_loader)
        val_logs = val_epoch.run(val_loader)








