import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.optim as optim


class UNetModel:
    """
    This class handle the model definition
    of basic Encoder-Decoder architecture
    for U-net. It return either a model for train or
    validation mode
    """

    def __init__(self, classes, mode, device, encoder_name='resnet18',
                 encoder_weights='imagenet', activation='sigmoid', aux_params=None, loss='dice',
                 optimizer='adam', lr=0.0001, custom_encoder_weights=None):
        """
        params to build a model
        :param classes: int number of possible classes
        :param mode: str. 'train' or 'eval'
        :param device: str. 'cuda' or 'cpu'
        :param encoder_name: str. name of the encoder, e.g 'resnet18'
        :param encoder_weights: str. where pre-trained, e.g 'imagenet'
        :param activation: str. activation function, e.g 'sigmoid'
        :param aux_params: dict. contains hyper_params for the model
        :param loss: str. 'dice' for segmenation, 'ce' for multiclass, 'bce' for binary



        """
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.aux_params = aux_params
        self.classes = classes
        self.activation = activation  # Maybe this should be removed
        self.loss = loss
        self.device = device
        # self.mode = mode
        self.custom_encoder_weights = custom_encoder_weights

        self.model = None
        self.optimizer = None
        self.lr = lr
        self.metrics = None
        self.epoch = None

        # Set all the steps to make it ready for train/eval
        self.set_model()
        self.set_learning_regime()
        # self.set_epoch()

    def set_model(self):
        """Determines the architecture and pre-trained weights of UNet"""
        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            classes=self.classes,
            activation=self.activation,
            in_channels=1
        )

        # Using custom pre-trained_weights
        if self.custom_encoder_weights:
            encoder_state_dict = self.model.encoder.state_dict()
            encoder_state_dict.update(self.custom_encoder_weights)
            self.model.encoder.load_state_dict(encoder_state_dict)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def set_learning_regime(self):
        """Set learning rate and optimizer"""
        if self.loss == 'dice':
            self.loss = smp.utils.losses.DiceLoss()
        elif self.loss == 'ce':
            self.loss = smp.utils.losses.CrossEntropyLoss()
        elif self.loss == 'bce':
            self.loss = smp.utils.losses.BCELoss()
        else:
            raise ValueError('{} loss not supported'.format(self.loss))

        # TODO: Add Focal Loss

        self.optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=self.lr, weight_decay=0.01),
        ])

        # TODO: Make this configurable
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

    def set_epoch(self, mode):
        """Setting all the steps to perform one epoch"""
        if mode == 'train':
            self.epoch = smp.utils.train.TrainEpoch(
                self.model,
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.metrics,
                device=self.device,
                verbose=True
            )
        elif mode == 'eval':
            self.epoch = smp.utils.train.ValidEpoch(
                model=self.model,
                loss=self.loss,
                metrics=self.metrics,
                device=self.device,
                verbose=True
            )
        else:
            raise ValueError('Mode must be "train" or "eval"')


# TODO: Inherit methods from the above UnetModel class
class UnetEncoderClassiferModel(nn.Module, UNetModel):
    """
    Class supporting the training of
    the encoder part + classification head
    """

    def __init__(self, encoder_params, cls_params):
        # Inherit __init__ of parent classes
        super(UnetEncoderClassiferModel, self).__init__()

        self.encoder_params = encoder_params
        self.cls_params = cls_params
        # TODO: Add general params such as l_rate, l_regime and so on

        self.encoder, self.cls = self.set_encoder_classifier()

    # Defining the parts
    def set_encoder_classifier(self):
        """Set the encoder part"""
        aux_params = {'classes': 2, 'activation': 'softmax'}
        unet = smp.Unet(in_channels=1, encoder_name='resnet34', encoder_weights=None, aux_params=aux_params)

        # TODO: Allow for building custom classifier head
        return unet.encoder, unet.classification_head

    # TODO: Setting a future head
    def set_cls_head(self):
        """Setting a custom classification head"""
        pass

    # defining the forward pass
    def forward(self, x):
        x = self.encoder(x)[-1]  # Keeping only the last block of the encoder
        x = self.cls(x)

        return x


class UnetEncoderMultiTaskModel(nn.Module):
    """
    Setting the architecture for Multitask-learning
    with the encoder (resnet34) as backbone
    """
    def __init__(self):
        # Inherit __init__ of parent class
        super(UnetEncoderMultiTaskModel, self).__init__()
        aux_params_gender = {'classes': 2, 'activation': 'softmax'}
        aux_params_age = {'classes': 5, 'activation': 'softmax'}
        unet_gender = smp.Unet(in_channels=1, encoder_name='resnet34',
                               encoder_weights=None, aux_params=aux_params_gender)
        unet_age = smp.Unet(in_channels=1, encoder_name='resnet34',
                            encoder_weights=None, aux_params=aux_params_age)

        self.encoder = unet_gender.encoder
        self.cls_gender = unet_gender.classification_head
        self.cls_age = unet_age.classification_head

    def forward(self, x):
        x = self.encoder(x)[-1]
        x_gender = self.cls_gender(x)
        x_age = self.cls_age(x)

        return x_gender, x_age


class UnetEncoderDecoderMultiTaskModel(nn.Module):
    """
    Train with a triple head: Decoder for segmentation,
    and two cls heads for age and gender
    """
    def __init__(self):
        # Inherit __init__ of parent class
        super(UnetEncoderDecoderMultiTaskModel, self).__init__()
        aux_params_gender = {'classes': 2, 'activation': 'softmax'}
        aux_params_age = {'classes': 5, 'activation': 'softmax'}
        unet_gender = smp.Unet(in_channels=1, encoder_name='resnet34',
                               encoder_weights=None, aux_params=aux_params_gender)
        unet_age = smp.Unet(in_channels=1, encoder_name='resnet34',
                            encoder_weights=None, aux_params=aux_params_age)

        # Segmentation parts
        self.encoder = unet_gender.encoder
        self.decoder = unet_gender.decoder
        self.seg_head = unet_gender.segmentation_head

        # Multi Task parts
        self.cls_gender = unet_gender.classification_head
        self.cls_age = unet_age.classification_head

    def forward(self, x):
        # inputs decoder and cls heads
        x_encoder = self.encoder(x)
        x_cls = x_encoder[-1]

        # cls heads
        x_gender = self.cls_gender(x_cls)
        x_age = self.cls_age(x_cls)

        # segmentation head
        x_decoder = self.decoder(*x_encoder)
        x_seg_mask = self.seg_head(x_decoder)

        return x_seg_mask, x_gender, x_age


class TrainEncoderClassifier:

    def __init__(self, model_name=None):
        self.model = None
        self.device = None
        self.model_name = model_name

    def set_model(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # TODO: Should the model be defined outside - probably yes
        self.model = model
        # self.model.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        #     pass

    def train(self):
        # Just training for the final part
        pass

    def train_and_eval(self, train_loader, val_loader, params_dict):
        # 1. Set training params
        # TODO: Remember args and kwargs
        # Train and evaluate after each epoch
        # Get training params or set defaults
        num_epochs = params_dict.get('num_epochs', 500)
        lr_rate = params_dict.get('lr', 0.0001)
        lr_decay = params_dict.get('lr_decay', False)
        if isinstance(lr_decay, bool) and lr_decay:
            # TODO: Set some learning regime
            pass

        # TODO: Add more criterions
        criterion = params_dict.get('criterion', 'bce_loss')
        if criterion == 'bce_loss':
            criterion = nn.BCELoss()
        elif criterion == 'ce':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('{} is not a supported criterion'.format(criterion))

        optimizer = params_dict.get('optimizer', 'adam')
        # TODO: Add optimization kwargs
        if optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr_rate)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr_rate)
        elif optimizer == 'adamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr_rate)
        else:
            raise ValueError('{} is not a supported optimizer'.format(optimizer))

        # 2. Train and eval model:
        best_loss = 1000
        for epoch in range(num_epochs):

            train_loss = 0
            val_loss = 0
            min_val_loss = 100.0
            train_size = 0
            val_size = 0
            self.model.train()
            # Train
            with torch.set_grad_enabled(True):
                for idx, (images, labels) in enumerate(train_loader):

                    # 1. Pass data to gpu
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # 2. Zero gradients and forward pass
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    outputs = outputs[:, 0].unsqueeze(dim=1)

                    # 3. compute loss ,backprop and update weights
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_size += images.shape[0]

            # 3. Eval
            with torch.no_grad():
                self.model.eval()
                for idx, (images, labels) in enumerate(val_loader):
                    
                    # 1. Pass data to GPU
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # 2. Forward pass
                    outputs = self.model(images)
                    outputs = outputs[:, 0].unsqueeze(dim=1)

                    # 3. compute loss:
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_size += images.shape[0]

                # Save best model weights
                if ((val_loss / val_size) < best_loss) and self.model_name is not None:
                    torch.save(self.model.state_dict(), self.model_name)
                    best_loss = val_loss / val_size

            # Print results
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print('EPOCH:', epoch + 1)
                print('Train Loss:', train_loss / train_size)
                print('Validation Loss:', val_loss / val_size)
                print('==================================')


# TODO: Merge this into one formal trainers
class TrainEncoderMultiTask:
    """Traininig multi task classifiers"""
    # TODO: support weights for gender and age values
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.model = None
        self.device = None

    def set_model(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model

    def train(self):
        """Just training for when model is set"""
        pass

    def train_and_eval(self, train_loader, val_loader, params_dict):
        """Train and eval for model params selecting"""
        # 1. Set training params
        # TODO: Remember args and kwargs
        # Train and evaluate after each epoch
        # Get training params or set defaults
        num_epochs = params_dict.get('num_epochs', 500)
        lr_rate = params_dict.get('lr', 0.0001)
        lr_decay = params_dict.get('lr_decay', False)
        if isinstance(lr_decay, bool) and lr_decay:
            # TODO: Set some learning regime
            pass

        # TODO: Add more criteria
        criterion = params_dict.get('criterion', 'bce_loss')
        if criterion == 'bce_loss':
            criterion = nn.BCELoss()
        elif criterion == 'ce':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('{} is not a supported criterion'.format(criterion))

        optimizer = params_dict.get('optimizer', 'adam')
        # TODO: Add optimization kwargs
        if optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr_rate, weight_decay=0.01)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr_rate)
        elif optimizer == 'adamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr_rate)
        else:
            raise ValueError('{} is not a supported optimizer'.format(optimizer))

        # 2. Train and eval the model
        best_loss = 2000
        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            train_size = 0
            val_size = 0
            self.model.train()
            with torch.set_grad_enabled(mode=True):
                for idx, (images, labels_1, labels_2) in enumerate(train_loader):
                    # 1. Pass data to GPU
                    images = images.to(self.device)
                    labels_1 = labels_1.to(self.device).squeeze(dim=1)
                    labels_2 = labels_2.to(self.device).squeeze(dim=1)

                    # 2. Zero gradients and forward pass
                    optimizer.zero_grad()
                    output_1, output_2 = self.model(images)

                    # 3. Compute loss, back-prop and update weights:
                    loss_1 = criterion(output_1, labels_1)
                    loss_2 = criterion(output_2, labels_2)

                    # TODO: Consider adding weights
                    # loss = 0.5 * loss_1 + 0.5 * loss_2
                    loss = 0.7 * loss_1 + 0.3 * loss_2  # Keeping in proportions
                    loss.backward()
                    optimizer.step()

                    # 4. Compute running loss:
                    train_loss += loss.item()
                    train_size += images.shape[0]

            with torch.no_grad():
                for idx, (images, labels_1, labels_2) in enumerate(val_loader):

                    # 1. Pass data to GPU
                    images = images.to(self.device)
                    labels_1 = labels_1.to(self.device).squeeze(dim=1)
                    labels_2 = labels_2.to(self.device).squeeze(dim=1)

                    # 2. Forward pass
                    output_1, output_2 = self.model(images)

                    # 3. Compute loss
                    loss_1 = criterion(output_1, labels_1)
                    loss_2 = criterion(output_2, labels_2)
                    # loss = 0.5 * loss_1 + 0.5 * loss_2
                    loss = 0.65 * loss_1 + 0.35 * loss_2  # Keeping in proportions

                    # keep running loss:
                    val_loss += loss.item()
                    val_size += images.shape[0]

                # keep best model
                if (val_loss / val_size) < best_loss and self.model_name:
                    torch.save(self.model.state_dict(), self.model_name)
                    best_loss = val_loss / val_size

            # Print results
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print('EPOCH:', epoch + 1)
                print('Train Loss:', train_loss / train_size)
                print('Validation Loss:', val_loss / val_size)
                print('==================================')


class TrainEncoderDecoderMultiTask:
    """
    Trainer methods for training a full Unet along side
    two classification heads "stemming" from the decoder
    """
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.model = None
        self.device = None

    def set_model(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model

    def train(self, train_loader):
        """Just training for when model is set"""
        pass

    def train_and_eval(self, train_loader, val_loader, params_dict):
        """Train and eval for model params selecting"""
        # 1. Set training params
        # TODO: Remember args and kwargs
        # Train and evaluate after each epoch
        # Get training params or set defaults
        num_epochs = params_dict.get('num_epochs', 500)
        lr_rate = params_dict.get('lr', 0.0001)
        lr_decay = params_dict.get('lr_decay', False)
        if isinstance(lr_decay, bool) and lr_decay:
            # TODO: Set some learning regime
            pass

        # set classification loss
        cls_criterion = params_dict.get('cls_criterion', 'bce_loss')
        if cls_criterion == 'bce_loss':
            cls_criterion = nn.BCELoss()
        elif cls_criterion == 'ce':
            cls_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('{} is not a supported classification criterion'.format(cls_criterion))

        # set segmentation loss
        seg_criterion = params_dict.get('seg_criterion', 'dice')
        if seg_criterion == 'dice':
            seg_criterion = smp.utils.losses.DiceLoss()
        else:
            raise ValueError('{} is not a supported segmentation criterion'.format(seg_criterion))

        optimizer = params_dict.get('optimizer', 'adam')
        # TODO: Add optimization kwargs
        # TODO: Allow different learning for every head
        if optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr_rate)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr_rate)
        else:
            raise ValueError('{} is not a supported optimizer'.format(optimizer))

        # 2. Train and eval the model
        best_loss = 2000
        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            train_size = 0
            val_size = 0
            with torch.set_grad_enabled(True):
                self.model.train()
                for idx, (images, masks, labels_1, labels_2) in enumerate(train_loader):
                    # 1. Pass data to GPU
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    labels_1 = labels_1.to(self.device).squeeze(dim=1)
                    labels_2 = labels_2.to(self.device).squeeze(dim=1)

                    # 2. Forward pass
                    optimizer.zero_grad()
                    msk_output, output_1, output_2 = self.model(images)

                    # 3. Compute loss and update weights
                    seg_loss = seg_criterion(msk_output, masks)  # TODO: Understand required shapes
                    cls_loss_1 = cls_criterion(output_1, labels_1)
                    cls_loss_2 = cls_criterion(output_2, labels_2)

                    # TODO: Add scaling to the different loss
                    loss = (seg_loss + cls_loss_1 + cls_loss_2) / 3
                    loss.backward()
                    optimizer.step()

                    # Compute running loss
                    train_loss += loss.item()
                    train_size += images.shape[0]

            with torch.no_grad():
                self.model.eval()
                for idx, (images, masks, labels_1, labels_2) in enumerate(val_loader):
                    # 1. Pass data to GPU
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    labels_1 = labels_1.to(self.device).squeeze(dim=1)
                    labels_2 = labels_2.to(self.device).squeeze(dim=1)

                    # 2. Forward pass
                    msk_output, output_1, output_2 = self.model(images)

                    # 3. Compute loss
                    seg_loss = seg_criterion(msk_output, masks)
                    cls_loss_1 = cls_criterion(output_1, labels_1)
                    cls_loss_2 = cls_criterion(output_2, labels_2)

                    loss = (seg_loss + cls_loss_1 + cls_loss_2) / 3

                    # Compute running loss
                    val_loss += loss.item()
                    val_size += images.shape[0]

                    # Save best model:
                    if (val_loss / val_size) < best_loss:
                        torch.save(self.model.state_dict(), self.model_name)
                        best_loss = val_loss / val_size

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print('EPOCH:', epoch + 1)
                print('Train Loss:', train_loss / train_size)
                print('Validation Loss:', val_loss / val_size)
                print('==================================')











