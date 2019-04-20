from torch import nn
from torch.nn import functional as F
class ResdualBlock(nn.Module):
    """
    实现子Module:Residual Block
    """
    def __init__(self,in_channels,out_channels,stride=1,shortcut=None):
        super(ResdualBlock,self).__init__()
        self.left = nn.Sequential(
          nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
          nn.BatchNorm2d(out_channels))
        self.right = shortcut
    
    def forward(self,x):
        out = self.left(x)
        # 没看懂
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    """
    实现module：resnet34
    resnet34包含多个layer,每个layer又包含多个residual block
    用子module来实现residual block,用__make_layer函数来实现layer
    """
    def __init__(self,in_channels=1,num_classes=33):
        super(ResNet,self).__init__()
            #前几层图像转换
            
#         nn.Conv2d(in_channels, out_channels, kernel_size, 
#                   stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1) )

        #重复的layer,分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64,64,3)  # 
        self.layer2 = self._make_layer(64,128,4,stride=2)
        self.layer3 = self._make_layer(128,256,6,stride=2)
        self.layer4 = self._make_layer(256,512,3,stride=2)

        ## 卷积操作以后降为原来的1/32
        # 分类用的全连接
        self.fc = nn.Linear(2048,num_classes)
            
    def _make_layer(slef,in_channels,out_channels,block_num,stride=1):
        """
        构建layer,包含多个residule block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),#nn.Conv2d()里面的参数好迷啊
            nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResdualBlock(in_channels,out_channels,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResdualBlock(out_channels,out_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.pre(x)     # (batch_size,64,16,16)
       
        x = self.layer1(x)  # (batch_size,64,16,16)
        x = self.layer2(x)  # (batch_size,128,8,8)
        x = self.layer3(x)  # (batch_size,256,4,4)
        x = self.layer4(x)  # (batch_size,512,2,2)
        
        x = F.avg_pool2d(x,1)
        # print(x.shape)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        # print("after fc:",x.shape)
        
        return  x      