import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.GRU,
                              nn.LSTM)):
                print(m)
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv(x)))
        return x


class SimCLR(nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()
        self.backbone = Encoder(args)
        self.projection_head = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )

    def forward(self, inputs):
        backbone = self.backbone(inputs)
        projection = self.projection_head(backbone)

        return projection


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.conv1 = ConvBlock(in_channels=args['input_size'],
                               out_channels=32,
                               kernel_size=24)
        self.conv2 = ConvBlock(in_channels=32,
                               out_channels=64,
                               kernel_size=16)
        self.conv3 = ConvBlock(in_channels=64,
                               out_channels=96,
                               kernel_size=8)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global Max Pooling (as per
        # https://github.com/keras-team/keras/blob
        # /7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/layers/pooling.py
        # #L559) for 'channels_first'
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.squeeze(2)

        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        # Encoder
        self.backbone = Encoder(args)

        # Softmax
        if args['classification_model'] == 'linear':
            self.softmax = nn.Linear(96, args['num_classes'])
        elif args['classification_model'] == 'mlp':
            # self.softmax = nn.Sequential(nn.Linear(96, 256),
            #                              nn.ReLU(inplace=True),
            #                              nn.Linear(256, args['num_classes']))
            self.softmax = nn.Sequential(nn.Linear(96, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, args['num_classes']))

            
    def forward(self, inputs):
        # Passing it through the encoder
        backbone = self.backbone(inputs)

        softmax = self.softmax(backbone)

        return softmax

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args['saved_model'])

        print('Loading the pre-trained weights')
        checkpoint = torch.load(state_dict_path, map_location=args['device'])
        pretrained_checkpoint = checkpoint['model_state_dict']

        self.load_state_dict(pretrained_checkpoint, False)

        return

    def freeze_encoder_layers(self):
        """
        To set only the softmax to be trainable
        :return: None, just setting the encoder part (or the CPC model) as
        frozen
        """
        # First setting the model to eval
        self.backbone.eval()

        # Then setting the requires_grad to False
        for param in self.backbone.parameters():
            param.requires_grad = False

        return

    def freeze_two_conv_layers(self):
        """
        Setting the first two conv layers to be frozen.
        Classifier and the last conv layer remain trainable.
        """
        # First setting the two conv layers to eval
        self.backbone.conv1.eval()
        self.backbone.conv2.eval()
        
        # Then setting the requires_grad to False
        for param in self.backbone.named_parameters():
          if 'conv3' in param[0]:
            param[1].requires_grad = True
          else:
            param[1].requires_grad = False

        return
