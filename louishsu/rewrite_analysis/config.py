from easydict import EasyDict

configer = EasyDict()


configer.dsize = (64, 64)
configer.n_channel = 1
configer.n_class = 63


configer.splitmode = 'split_{}x{}_1'.format(configer.dsize[0], configer.dsize[1])
configer.modelbase = 'analysis_vgg11_bn'

configer.modelname = '{}_{}'.\
                    format(configer.modelbase, configer.splitmode)



configer.datapath = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.\
                                format(configer.dsize[0], configer.dsize[1])
configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs'
configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles'


## training step
configer.batchsize = 64
configer.n_epoch   = 350

## learing rate
configer.lrbase = 0.005
configer.stepsize = 250
configer.gamma = 0.1
