import torch
from torch import nn, optim
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