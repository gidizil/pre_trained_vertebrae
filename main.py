import torch
from train_utils import UNetModel, UnetEncoderClassiferModel, TrainEncoderClassifier
from train_utils import UnetEncoderMultiTaskModel, TrainEncoderMultiTask
from train_utils import UnetEncoderDecoderMultiTaskModel, TrainEncoderDecoderMultiTask
from data_utils import VertebraeDataset, VertebraeDataset2, Utils as DataUtils
from torch.utils.data import DataLoader
import torch.nn as nn

"""======================================================================================================"""
"""====================================== Starting new configuration ===================================="""
"""======================================================================================================"""
gen_new_data = False

train_encoder = True
train_full = True
k_train = 10

# 1. Common work
data_utils = DataUtils()
if gen_new_data:
    data_utils.rm_train_val_dirs()  # get rid of old data if exists
    data_utils.gen_train_val_test_dirs(k_train)  # gen new data

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. work for cls
if train_encoder:
    """Configuration:"""
    # 1. Set Data
    train_imgs_dir = 'train_images_cls'
    val_imgs_dir = 'val_images_cls'
    train_masks_dir, val_masks_dir = None, None

    # 2. Set architecture:
    train_encoder_single = False
    train_encoder_multi = False
    train_encoder_multi_decoder = True

    # 3. set label(s)
    gender = True
    age = True
    decoder = True

    if decoder:
        train_imgs_dir, val_imgs_dir = 'train_images', 'val_images'
        train_masks_dir, val_masks_dir = 'train_masks', 'val_masks'

    train_data = VertebraeDataset2(images_dir=train_imgs_dir, masks_dir=train_masks_dir,
                                   age=age, gender=gender, decoder=decoder, classes=["verte"])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=12)

    val_data = VertebraeDataset2(images_dir=val_imgs_dir, masks_dir=val_masks_dir,
                                 age=age, gender=gender, decoder=decoder, classes=["verte"])
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=12)

    if train_encoder_single:
        model = UnetEncoderClassiferModel(cls_params=None, encoder_params=None)
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # 2. Train and eval
        tr_encoder_cls = TrainEncoderClassifier(model_name='best_cls_params.pth')
        tr_encoder_cls.set_model(model)
        params_dict = {'num_epochs': 50, 'lr': 0.00003, 'optimizer': 'adam'}
        tr_encoder_cls.train_and_eval(train_loader, val_loader, params_dict=params_dict)

    if train_encoder_multi:
        # 1. Define Model and pass it to GPU
        model = UnetEncoderMultiTaskModel()
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Train and Eval
        tr_encoder_multi = TrainEncoderMultiTask(model_name='best_multi_params.pth')
        tr_encoder_multi.set_model(model)
        params_dict = {'num_epochs': 50, 'lr': 0.00002, 'optimizer': 'adam', 'criterion': 'ce'}
        tr_encoder_multi.train_and_eval(train_loader, val_loader, params_dict=params_dict)

    if train_encoder_multi_decoder:
        # 1. Define model and pass it to GPU
        model = UnetEncoderDecoderMultiTaskModel()
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # 2. Train and Eval
        tr_enc_dec_multi = TrainEncoderDecoderMultiTask(model_name='best_dec_multi_params.pth')
        tr_enc_dec_multi.set_model(model)
        params_dict = {'cls_criterion': 'ce', 'seg_criterion': 'dice', 'num_epochs': 80, 'lr': 0.0003}
        tr_enc_dec_multi.train_and_eval(train_loader, val_loader, params_dict=params_dict)

# 3. work for seg
if train_full:
    # 0. Configuration:
    set_new_encoder = True
    single_encoder = False
    multi_encoder = False
    dec_multi_encoder = True

    weights_type = 'custom'  # 'custom', 'scratch', 'imagenet'

    # 1. Set data
    train_imgs_dir, val_imgs_dir = 'train_images', 'val_images'
    train_masks_dir, val_masks_dir = 'train_masks', 'val_masks'

    train_data = VertebraeDataset2(images_dir=train_imgs_dir, masks_dir=train_masks_dir,
                                   age=False, gender=False, decoder=False, classes=["verte"])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=12)

    val_data = VertebraeDataset2(images_dir=val_imgs_dir, masks_dir=val_masks_dir,
                                 age=False, gender=False, decoder=False, classes=["verte"])
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=12)

    # 2. if wanted select a custom pre-trained weights
    custom_encoder_weights = None
    if set_new_encoder:
        if single_encoder:
            encoder_cls_params = torch.load('best_cls_params.pth')
            custom_encoder_weights = DataUtils.update_state_dict(encoder_cls_params, params_type='cls')
        elif multi_encoder:
            encoder_multi_params = torch.load('best_multi_params.pth')
            custom_encoder_weights = DataUtils.update_state_dict(encoder_multi_params, params_type='multi')
        elif dec_multi_encoder:
            encoder_dec_multi_params = torch.load('best_dec_multi_params.pth')
            custom_encoder_weights = DataUtils.update_state_dict(encoder_dec_multi_params, params_type='dec_multi')

    # 3. Set the pre-trained weights:
    if weights_type == 'scratch':
        encoder_weights = None
        custom_encoder_weights = None
    elif weights_type == 'imagenet':
        encoder_weights = 'imagenet'
        custom_encoder_weights = None
    elif weights_type == 'custom':
        encoder_weights = None
        custom_encoder_weights = custom_encoder_weights  # just for completeness:)
    else:
        raise ValueError("weights_type should be 'scratch', 'imagenet' or 'custom'")

    unet_model = UNetModel(classes=1, mode='train', device=device, loss='dice',
                           encoder_weights=encoder_weights, encoder_name='resnet34',
                           custom_encoder_weights=custom_encoder_weights)

    # 4. Train the full model:
    unet_model.set_epoch(mode='train')
    train_epoch = unet_model.epoch
    unet_model.set_epoch(mode='eval')
    val_epoch = unet_model.epoch

    tr_dice_loss = []
    val_dice_loss = []
    tr_iou_score = []
    val_iou_score = []
    max_score = 0.0
    for idx in range(150):
        print('\nEpoch: {}'.format(idx))
        train_logs = train_epoch.run(train_loader)
        val_logs = val_epoch.run(val_loader)

        # Keep best model:
        if max_score < val_logs['iou_score']:
            max_score = val_logs['iou_score']
            torch.save(unet_model.model, './best_custom_dec_multi_10_multi_model.pth')
            print('Model saved!')

        # keep the data and plot it after words into a .jpg
        tr_dice_loss.append(train_logs['dice_loss'])
        tr_iou_score.append(train_logs['iou_score'])

        val_dice_loss.append(val_logs['dice_loss'])
        val_iou_score.append(val_logs['iou_score'])

    dice_header = 'Dice Loss - Decoder-Multi Pre-Trained (k=10)'
    iou_header = 'IOU Score - Decoder-Multi Pre-Trained (k=10)'
    DataUtils.plot_train_val(tr_dice_loss, val_dice_loss, dice_header,
                             'Train', 'Validation', 'Epochs', 'Loss', 'custom_dec_multi_10_dice_loss.jpg')
    DataUtils.plot_train_val(tr_iou_score, val_iou_score, iou_header,
                             'Train', 'Validation', 'Epochs', 'Score', 'custom_dec_multi_10_iou_score.jpg')

    print('================')
    print('Best Score: ' + str(max_score))
    print('================')








