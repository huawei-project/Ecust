"""
# Author: Yuru Chen
# Time: 2019 03 25
"""
import torch
import torch.nn as nn
import torchvision.models as models

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class myVGG(nn.Module):
    def __init__(self, config='A', num_classes=1000):
        super(myVGG, self).__init__()
        self.features = self.make_layers(cfg[config], batch_norm=False)
        self.global_avg = nn.AvgPool2d(kernel_size=(4, 4), padding=0)
        self.classifier_ = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128, num_classes),
        )
        self.parameter_initialization()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_(x)
        return x

    def inference(self, x):
        y = torch.argmax(x, -1)
        return y

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    class LossFunc(nn.Module):
        def __init__(self, num_classes):
            super(myVGG.LossFunc, self).__init__()
            self.classifier = nn.CrossEntropyLoss()

        def forward(self, pred, gt):
            loss = self.classifier(pred, gt)
            return loss
