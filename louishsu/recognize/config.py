from torch.cuda import is_available
from easydict import EasyDict

configer = EasyDict()

configer.datapath = "/datasets/ECUST2019"
# configer.datapath = "/home/louishsu/Work/Workspace/ECUST2019"
configer.logspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs"
configer.mdlspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles"

configer.facesize       = (64, 64)
configer.n_channels     = 46


configer.usedChannels   = [550]

configer.n_usedChannels = len(configer.usedChannels)
configer.n_classes      = 33

configer.splitmode = 'split_1'
configer.modelbase = "recognize_vgg11"
configer.modelname = "{}_{}_{}chs_{}sta_20nm".\
            format(configer.modelbase, configer.splitmode, configer.n_usedChannels, configer.usedChannels[0])

configer.lossname  = 'crossent'
configer.learningrate  = 1e-4
configer.batchsize     = 64
configer.n_epoch       = 300

configer.cuda = is_available()

