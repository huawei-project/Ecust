import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class ChannelShuffle(nn.Module):
    """ shuffle channels 
    Attributes:
        n_groups: {int}
    """
    def __init__(self, n_groups):
        super(ChannelShuffle, self).__init__()
        self.n_groups = n_groups
    def forward(self, x):
        """
        Args:
            x: {tensor(N, C, H, W)}
        Returns:
            out: {tensor(N, C, H, W)}
        """
        (n_samples, n_channels, height, width) = x.shape
        n_channels_sub = n_channels // self.n_groups
        n_channels_subsub = n_channels_sub // self.n_groups
        if n_channels != n_channels_sub * self.n_groups or\
                 n_channels_sub != n_channels_subsub * self.n_groups:
            print('channels error!'); return
        
        x_groups = torch.zeros(size=(self.n_groups, n_samples, n_channels_sub, height, width))
        for i_groups in range(self.n_groups):
            x_groups[i_groups] = x[:, i_groups*n_channels_sub:(i_groups+1)*n_channels_sub, :, :]

        out = torch.zeros_like(x)
        for i_groups in range(self.n_groups):
            x_groups_sub = x_groups[:, :, i_groups*n_channels_subsub:(i_groups+1)*n_channels_subsub, :, :]
            out_groups = [x_groups_sub[j_groups] for j_groups in range(self.n_groups)]
            out[:, i_groups*n_channels_sub:(i_groups+1)*n_channels_sub, :, :] = torch.cat(out_groups, 1)
                    
        return out

class ShuffleUnit(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, n_groups, stride=1, downsample=None):
        """
        Args:
            inplanes: {int} 
            planes: {int}
            n_groups: {int}
        """
        super(ShuffleUnit, self).__init__()
        self.gconv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=n_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shuffle = ChannelShuffle(n_groups)
        self.conv = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        outplanes = planes*self.expansion
        self.gconv2 = nn.Conv2d(planes, outplanes, kernel_size=1, groups=n_groups)
        self.bn3 = nn.BatchNorm2d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.shuffle(out)

        out = self.conv(out)
        out = self.bn2(out)

        out = self.gconv2(out)
        self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)

        return out

class ShuffleNet(nn.Module):
    """
    Notes:
        - based on ResNet
        - inplace Bottleneck with ShuffleUnit
    """

    def __init__(self, in_channels, n_groups, layers, block=ShuffleUnit, num_classes=10):
        """
        Args:
            in_channels: {int} 
            n_groups: {int}
            layers: {list[int]}
        """
        self.inplanes = 64
        self.n_groups = n_groups
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, self.inplanes,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.n_groups, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.n_groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
