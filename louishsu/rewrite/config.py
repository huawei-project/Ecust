from easydict import EasyDict

configer = EasyDict()


configer.dsize = (64, 64)
configer.n_classes = 63






configer.datapath = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.\
                                format(configer.dsize[0], configer.dsize[1])
configer.logspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs"
configer.mdlspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles"


## training step
configer.batchsize = 64
configer.n_epoch   = 350

## learing rate
configer.lrbase = 0.005
configer.stepsize = 250
configer.gamma = 0.1
