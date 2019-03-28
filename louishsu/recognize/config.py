from torch.cuda import is_available
from easydict import EasyDict

configer = EasyDict()
configer.datapath = "/datasets/ECUST2019"
# configer.datapath = "/home/louishsu/Work/Workspace/ECUST2019"
configer.logspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs"
configer.mdlspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles"

configer.facesize       = (64, 64)
configer.n_channels     = 46
configer.n_classes      = 40

configer.trainmode = 'RGB'
configer.splitmode = 'split_1'
configer.modelbase = "recognize_vgg11"



if configer.trainmode == 'Multi':
    configer.usedChannels    = [550]
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = "{}_{}_{}chs_{}sta_20nm".\
                format(configer.modelbase, configer.splitmode, configer.n_usedChannels, configer.usedChannels[0])
elif configer.trainmode == 'RGB':
    configer.usedRGBChannels = 'R'
    configer.n_usedChannels = 1
    configer.modelname = '{}_{}_{}'.\
                format(configer.modelbase, configer.splitmode, configer.usedRGBChannels)



configer.lossname  = 'crossent'
configer.learningrate  = 0.005
configer.batchsize     = 64
configer.n_epoch       = 350

configer.stepsize   = 250
configer.gamma      = 0.1

configer.cuda = is_available()

