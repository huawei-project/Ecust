import os
import torch
import torch.nn as nn
import math
import torchvision.models as models


class VGG(nn.Module):
    cfg = {
        'VGG11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], 
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], 
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'], 
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    model_pres = {
        'VGG11':    models.vgg11,
        'VGG13':    models.vgg13,
        'VGG16':    models.vgg16,
        'VGG19':    models.vgg19,
        'VGG11_bn': models.vgg11_bn,
        'VGG13_bn': models.vgg13_bn,
        'VGG16_bn': models.vgg16_bn,
        'VGG19_bn': models.vgg19_bn,
    }
    model_urls = {
        'VGG11':    'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'VGG13':    'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'VGG16':    'https://download.pytorch.org/models/vgg16-397923af.pth',
        'VGG19':    'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'VGG11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'VGG13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'VGG16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'VGG19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }
    
    def __init__(self, in_channels, n_classes, input_size, netname, batch_norm=False, init_weights=True, finetune=True, pretrain_path=None):
        super(VGG, self).__init__()
        self.features = self._make_layer(netname, in_channels, input_size, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, n_classes),
        )

        if init_weights:
            self._initialize_weights()
        if finetune:
            if (pretrain_path is not None) and os.path.exists(pretrain_path):
                pretrained = torch.load(pretrain_path)
                self._fine_tune(pretrained)
            else:
                netname = netname + '_bn' if batch_norm else netname
                if netname in self.model_pres.keys():
                    pretrained = self.model_pres[netname](pretrained=True)
                    self._fine_tune(pretrained)

    def _fine_tune(self, model_pre):
        dict_totrain = self.features.state_dict()
        dict_pretrain = model_pre.features.state_dict()

        dict_update = {key: value\
                            for key, value in dict_pretrain.items()\
                            if (key in dict_totrain) and (key.split('.')[0]!='0')}
        dict_totrain.update(dict_update)
        self.features.load_state_dict(dict_totrain)


    def _make_layer(self, netname, in_channels, input_size, batch_norm=False):
        cfg = self.cfg[netname]

        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=(input_size//32, input_size//32))]

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
