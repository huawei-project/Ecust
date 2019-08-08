from easydict import EasyDict

configer = EasyDict()

configer.dsize = (112, 96)
configer.datatype = 'Multi'
configer.n_epoch  = 300 \
    if configer.datatype == 'Multi' else 350
configer.lrbase = 0.001 \
    if configer.datatype == 'Multi' else 0.0005

configer.n_channel = 23
configer.n_class = 63
configer.batchsize = 32
configer.stepsize = 250
configer.gamma = 0.2
configer.cuda = True

configer.splitmode = 'split_{}x{}_1'.format(configer.dsize[0], configer.dsize[1])
configer.modelbase = 'recognize_vgg11_bn'

if configer.datatype == 'Multi':
    configer.usedChannels = [550+i*20 for i in range(23)]
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '{}_{}_{}'.\
                    format(configer.modelbase, configer.splitmode, 
                            '_'.join(list(map(str, configer.usedChannels))))
elif configer.datatype == 'RGB':
    configer.usedChannels = 'RGB'
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '{}_{}_{}'.\
                    format(configer.modelbase, configer.splitmode, configer.usedChannels)


configer.datapath = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.\
                                format(configer.dsize[0], configer.dsize[1])
configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                format(configer.modelbase, configer.splitmode, configer.n_class)
configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                format(configer.modelbase, configer.splitmode, configer.n_class)

