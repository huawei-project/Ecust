import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

safe_log = lambda x: torch.log(torch.clamp(x, 1e-8, 1e8))

class _BaseEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
        """
        Parameters
        ----------
        ignore_index : Specifies a target value that is ignored
                       and does not contribute to the input gradient
        reduction : Specifies the reduction to apply to the output: 
                    'mean' | 'sum'. 'mean': elemenwise mean, 
                    'sum': class dim will be summed and batch dim will be averaged.
        use_weight : whether to use weights of classes.
        weight : Tensor, optional
                a manual rescaling weight given to each class.
                If given, has to be a Tensor of size "nclasses"
        """
        super(_BaseEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_weights = use_weights
        if use_weights:
            print("w/ class balance")
            print(weight)
            self.weight = torch.FloatTensor(weight).cuda()
        else:
            print("w/o class balance")
            self.weight = None

    def get_entropy(self, pred, label):
        """
        Return
        ------
        entropy : shape [batch_size, d1, d2, ..., dl, num_classes]
        Description
        -----------
        Information Entropy based loss need to get the entropy according to your implementation, 
        echo element denotes the loss of a certain position and class.
        """
        raise NotImplementedError

    def forward(self, pred, label):
        """
        Parameters
        ----------
        pred: [batch_size, num_classes, d1, d2, ..., dl]
        label: [batch_size, d1, d2, ..., dl]
        """
        assert not label.requires_grad
        assert pred.dim() ==  label.dim() + 1
        assert pred.size(0) == label.size(0), "{0} vs {1} ".format(pred.size(0), label.size(0))

        n, c = pred.shape[:2]
        if self.use_weights:
            if self.weight is None:
                print('label size {}'.format(label.shape))
                freq = np.zeros(c)
                for k in range(c):
                    mask = (label == k)
                    freq[k] = torch.sum(mask)
                    print('{}th frequency {}'.format(k, freq[k]))
                weight = freq / np.sum(freq) * c
                weight = np.median(weight) / weight
                self.weight = torch.FloatTensor(weight).cuda()
                print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = 1
        if self.ignore_index is None:
            self.ignore_index = c + 1
        mask = label != self.ignore_index
        
        entropy = self.get_entropy(pred, label)
        weighted_entropy = entropy * self.weight

        if self.reduction == 'sum':
            loss = weighted_entropy.sum(-1)[mask].mean()
        elif self.reduction == 'mean':
            loss = weighted_entropy.mean(-1)[mask].mean()
        return loss

class CrossEntropy(_BaseEntropyLoss):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None,
                 eps=0.0, gamma=0, priorType='uniform'):
        """
        Parameters
        ----------
        eps : label smoothing factor
        gamma: reduces the relative loss for well-classified examples (p>0.5) putting more
               focus on hard misclassified example 
               ([0, 5], 2 is best according to https://arxiv.org/abs/1708.02002).
        prior : prior distribution, if uniform equivalent to the 
                label smoothing trick (https://arxiv.org/abs/1512.00567).
        """
        super(CrossEntropy, self).__init__(ignore_index, reduction, use_weights, weight)
        self.eps = eps
        self.gamma = gamma
        self.priorType = priorType

    def get_entropy(self, pred, label):
        """
        Parameters
        ----------
        pred: [batch_size, num_classes, d1, d2, ...]
        label: [batch_size, d1, d2, ...]

        Return
        ------
        entropy: [batch_size, d1, d2, ..., dl, num_classes]
        """
        n, c = pred.shape[:2]
        pred = F.softmax(pred, 1)
        spatial = [i for i in range(pred.dim())][1:-1]
        pred = pred.permute(0, *spatial, 1).contiguous()
        label = label.unsqueeze(-1).long()
        one_hot_label = ((torch.arange(c)).cuda() == label).float()

        if self.eps == 0:
            prior = 0
        else:
            if self.priorType == 'gaussian':
                tensor = (torch.arange(c).cuda() - label).float()
                prior = NormalDist(tensor, c / 10)
            elif self.priorType == 'uniform':
                prior = 1 / (c-1)

        smoothed_label = (1 - self.eps) * one_hot_label + self.eps * prior * (1 - one_hot_label)
        entropy = smoothed_label * (1 - pred)**self.gamma * safe_log(pred) + \
                  (1 - smoothed_label) * pred**self.gamma * safe_log(1 - pred)
        return -entropy 


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride,
                      1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class MobileFacenet(nn.Module):
    def __init__(self, num_classes, facesize=(112, 96), bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()
        self.num_classes = num_classes
        self.h, self.w = facesize
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (self.h // 16, self.w // 16), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 3, 1, 1, 0, linear=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.weight = Parameter(torch.Tensor(num_classes, 3))
        nn.init.xavier_uniform_(self.weight)

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine

    def get_feature(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        return feature

    class LossFunc(nn.Module):
        def __init__(self):
            super(MobileFacenet.LossFunc, self).__init__()
            self.ArcMargin = ArcMarginProduct()
            self.classifier = nn.CrossEntropyLoss()

        def forward(self, pred, gt):
            output = self.ArcMargin(pred, gt)
            loss = self.classifier(output, gt)
            return loss


class ArcMarginProduct(nn.Module):
    def __init__(self, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, label):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = self.s * output
        return output


class MobileFacenetLDA(nn.Module):
    def __init__(self, num_classes, facesize=(112, 96), bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()
        self.num_classes = num_classes
        self.h, self.w = facesize
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(
            512, 512, (self.h // 16, self.w // 16), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x

    def get_feature(self, x):
        return self.forward(x)

    class LossFunc(nn.Module):
        def __init__(self):
            super(MobileFacenetLDA.LossFunc, self).__init__()

        def forward(self, pred, gt):
            classes = torch.unique(gt)
            N = pred.shape[0]
            m_c_list = []
            m = pred.mean(0)
            loss_w = 0
            loss_b = 0
            for c in classes:
                mask = gt == c
                n_c = mask.sum()
                pred_c = pred[mask]
                m_c = pred_c.mean(0)
                loss_w += ((pred_c - m_c)**2).mean(-1).sum()
                loss_b += ((m_c - m)**2).sum() * n_c
            loss_w /= N
            loss_b /= N
            loss = loss_w / loss_b
            return loss


if __name__ == "__main__":
    input = Variable(torch.FloatTensor(2, 3, 112, 96))
    net = MobileFacenet()
    print(net)
    x = net(input)
    print(x.shape)
