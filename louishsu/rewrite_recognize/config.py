from easydict import EasyDict

configer = EasyDict()


configer.dsize = (64, 64)
configer.n_channel = 23
configer.n_class = 63


configer.splitmode = 'split_{}x{}_7'.format(configer.dsize[0], configer.dsize[1])
configer.modelbase = 'recognize_vgg11_bn'


configer.datatype = 'Multi'
if configer.datatype == 'Multi':
    configer.usedChannels = [550+i*20 for i in range(23)]
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '{}_{}_{}chs_{}sta_20nm'.\
                    format(configer.modelbase, configer.splitmode, configer.n_usedChannels, configer.usedChannels[0])
elif configer.datatype == 'RGB':
    configer.usedChannels = 'R'
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '{}_{}_{}'.\
                    format(configer.modelbase, configer.splitmode, configer.usedChannels)


# configer.datapath = '/datasets/ECUST2019_{}x{}'.\
configer.datapath = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.\
                                format(configer.dsize[0], configer.dsize[1])
configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                format(configer.modelbase, configer.splitmode, configer.n_class)
configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                format(configer.modelbase, configer.splitmode, configer.n_class)


## training step
configer.batchsize = 16
configer.n_epoch   = 300    # 300 for multi, 350 for rgb

## learing rate
configer.lrbase = 0.001     # 0.001 for multi, 0.0005 for rgb
configer.stepsize = 250
configer.gamma = 0.2

configer.cuda = True
