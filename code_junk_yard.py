import torch
from torch import nn, optim
import shutil
import os
import numpy as np

if False:
    model = None
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # 3.Train
    min_val_loss = 100.0
    for epoch in range(50):
        print('EPOCH:', epoch)
        train_loss = 0
        val_loss = 0
        model.train()
        with torch.set_grad_enabled(True):
            for idx, (images, labels) in enumerate(train_loader):

                # 1. Pass data to GPU
                images = images.to(device)
                labels = labels.to(device)
                # 2. Zero gradinets and calculate model outputs
                optimizer.zero_grad()
                outputs = model(images)
                outputs = outputs[:, 0].unsqueeze(dim=1)  # reshape outputs to match labels
                # 3. calculate loss, compute gradients and update weights
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print('Train Loss: ', train_loss / len(train_data))

        with torch.no_grad():
            model.eval()
            for idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                val_outputs = model(images)
                val_outputs = val_outputs[:, 0].unsqueeze(dim=1)  # reshape outputs to match labels
                val_loss += criterion(val_outputs, labels).item()  # TODO: Make this a scalar

            print('Validation Loss: ', val_loss / len(val_data))
            # Save parameters of best model
            if val_loss < min_val_loss:
                torch.save(model.state_dict(), 'best_model_params.pt')
                min_val_loss = val_loss

        print('====================================')


""" 4. Train encoder only using Age and Gender: """
if False:
    """ 4. Train encoder only using Age and Gender: """
    import segmentation_models_pytorch as smp
    aux_params = {'classes': 2, 'activation': 'softmax'}
    unet = smp.Unet(in_channels=1, encoder_name='resnet34', encoder_weights=None, aux_params=aux_params)
    encoder = unet.encoder
    cls = unet.classification_head
    print('reached here')

    # TODO: Use,
    # https://discuss.pytorch.org/t/how-can-i-connect-a-new-neural-network-after-a-trained-neural-network-and-optimize-them-together/48293/3
    # to concat the encoder to the model
    # start with the gender model and then add age.
    # Later think on how to get only the weights of the encoder

if False:
    # 2. set data:
    # TODO:Remove this transform elsewhere:
    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.NEAREST),
                                    transforms.ToTensor()])


    x_train_dir = 'images'
    y_train_dir = 'binary_masks'

    x_val_dir = 'images'
    y_val_dir = 'binary_masks'

    CLASSES = ['verte']
    train_data = VertebraeDataset2(
        x_train_dir,
        y_train_dir,
        # augmentation=get_training_augmentation(),
        # preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,)

    val_data = VertebraeDataset2(x_val_dir,
                                 y_val_dir,
                                 # augmentation=get_training_augmentation(),
                                 # preprocessing=get_preprocessing(preprocessing_fn),
                                 classes=CLASSES,)

    train_loader = DataLoader(train_data, batch_size=12, shuffle=True,
                              num_workers=12)
    val_loader = DataLoader(train_data, batch_size=3, shuffle=True,
                            num_workers=3)

if False:
    """General Preps Configuration"""
    # 0. Set what you want to train:
    train_plain_unet = False
    plain_unet_weights = None  # Can be None or 'imagenet'

    train_unet_cls = True
    train_unet_multi = False

    set_new_encoder = False
    cls_encoder = False
    multi_encoder = False
    dec_multi_encoder = False

    train_unet_multi_w_decoder = False

    gender = True
    age = False
    decoder = False

    # 1. Device:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Data - Set data according to train_type:
    if train_unet_cls or train_unet_multi:
        DataUtils.rm_train_val_dirs(train_type='cls_only')  # get rid of old data if exists

        train_masks_dir, val_masks_dir = None, None
        DataUtils.gen_train_val_test_dirs('cls_only', k=None) # get rid of old data if exists
    else:
        DataUtils.rm_train_val_dirs(train_type='segmentation')
        train_masks_dir, val_masks_dir = 'train_masks', 'val_masks'
        DataUtils.gen_train_val_test_dirs('segmentation', k=4)

    train_data = VertebraeDataset2(images_dir='train_images', masks_dir=train_masks_dir,
                                   age=age, gender=gender, decoder=decoder, classes=["verte"])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=12)

    val_data = VertebraeDataset2(images_dir='val_images', masks_dir=val_masks_dir,
                                 age=age, gender=gender, decoder=decoder, classes=["verte"])
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=12)


    """ 1. Train Unet Model and eval after each epoch """
    if train_plain_unet:
        tr_dice_loss = []
        tr_iou = []
        val_dice_loss = []
        val_iou = []
        max_score = 0.0
        unet_model = UNetModel(classes=1, mode='train', device=device, loss='dice',
                               encoder_weights=plain_unet_weights, encoder_name='resnet34')

        unet_model.set_epoch(mode='train')
        train_epoch = unet_model.epoch
        unet_model.set_epoch(mode='eval')
        val_epoch = unet_model.epoch
        for idx in range(150):
            print('\nEpoch: {}'.format(idx))
            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)

            # Keep best model:
            if max_score < val_logs['iou_score']:
                max_score = val_logs['iou_score']
                torch.save(unet_model.model, './best_unet_imagenet_model.pth')
                print('Model saved!')

            # keep the data and plot it after words into a .jpg
            tr_dice_loss.append(train_logs['dice_loss'])
            tr_iou.append(train_logs['iou_score'])

            val_dice_loss.append(val_logs['dice_loss'])
            val_iou.append(val_logs['iou_score'])

        dice_header = 'Dice Loss - Not Pre-Trained'
        iou_header = 'IOU Score - Not Pre-Trained'
        DataUtils.plot_train_val(tr_dice_loss, val_dice_loss, dice_header,
                                  'Train', 'Validation', 'Epochs', 'Loss', 'plain_imagenet_dice_loss.jpg')
        DataUtils.plot_train_val(tr_dice_loss, val_dice_loss, dice_header,
                                  'Train', 'Validation', 'Epochs', 'Score', 'plain_imagenet_iou_score.jpg')
        print('================')
        print('Best Score:' + str(max_score))
        print('================')

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
        params_dict = {'num_epochs': 150, 'lr': 0.00005, 'optimizer': 'adam'}
        tr_encoder_cls.train_and_eval(train_loader, val_loader, params_dict=params_dict)

    """3. Train UNet encoder with multi-task head for gender&age detection"""
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

    """ 5. Use encoder results from previous pre-training"""
    if set_new_encoder:
        # Adjust encoder weights
        if cls_encoder:
            encoder_cls_params = torch.load('best_cls_params.pth')
            custom_encoder_weights = DataUtils.update_state_dict(encoder_cls_params, params_type='cls')
        elif multi_encoder:
            encoder_multi_params = torch.load('best_multi_params.pth')
            custom_encoder_weights = DataUtils.update_state_dict(encoder_multi_params, params_type='multi')
        elif dec_multi_encoder:
            encoder_dec_multi_params = torch.load('best_dec_multi_params.pth')
            custom_encoder_weights = DataUtils.update_state_dict(encoder_dec_multi_params, params_type='dec_multi')

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

        tr_dice_loss = []
        val_dice_loss = []
        tr_iou_score = []
        val_iou_score = []
        max_score = 0.0
        for idx in range(500):
            print('\nEpoch: {}'.format(idx))
            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)

            # Keep best model:
            if max_score < val_logs['iou_score']:
                max_score = val_logs['iou_score']
                torch.save(unet_model.model, './best_unet_imagenet_model.pth')
                print('Model saved!')

            # keep the data and plot it after words into a .jpg
            tr_dice_loss.append(train_logs['dice_loss'])
            tr_iou.append(train_logs['iou_score'])

            val_dice_loss.append(val_logs['dice_loss'])
            val_iou.append(val_logs['iou_score'])

        dice_header = 'Dice Loss - Not Pre-Trained'
        iou_header = 'IOU Score - Not Pre-Trained'
        DataUtils.plot_train_val(tr_dice_loss, val_dice_loss, dice_header,
                                  'Train', 'Validation', 'Epochs', 'Loss', 'plain_imagenet_dice_loss.jpg')
        DataUtils.plot_train_val(tr_dice_loss, val_dice_loss, dice_header,
                                  'Train', 'Validation', 'Epochs', 'Score', 'plain_imagenet_iou_score.jpg')

if False:
    def gen_train_val_test_dirs(train_type, k=None):
        """
        :param train_type: str. if 'cls_only' can use all train data. else dependent on k.
        :param k: int. number of images to use for segmentation
        :return: None. generate necassery folders
        """
        # Images 13,14,15 are constantly test data. 1-12, 16-25 plays
        TEST_IMAGES = ('image013', 'image014', 'image015')
        TEST_MASKS = ('mask013', 'mask014', 'mask015')
        # 1. Set Test data to always evaluate against same images - keep 13,14,15 (40 slices each) for test
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
        if not os.path.exists('train_images_cls'):
            os.makedirs('train_images_cls')

        if not os.path.exists('train_images'):
            os.makedirs('train_images')
        if not os.path.exists('train_masks'):
                os.makedirs('train_masks')
        else:
            raise Exception('train_images directory already exists. please delete it')

        if not os.path.exists('val_images'):
            os.makedirs('val_images')
        if not os.path.exists('val_masks'):
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
