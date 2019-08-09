from easydict import EasyDict

configer = EasyDict()

## -------------------------- 训练相关 --------------------------
configer.n_epoch  = 100
configer.stepsize = 70
configer.batchsize = 2**7
configer.lrbase = 5e-4
configer.gamma = 0.2
configer.cuda = True
configer.savepath = 'checkpoints'

## ------------------------- 数据集相关 -------------------------
configer.datapath = "/datasets/ECUSTDETECT"
configer.dsize = (112, 96)
configer.n_channel = 25                 # 一份多光谱数据，包含25通道
configer.n_class = 92                   # 人员数目共92人

configer.datatype = 'Multi'             # "Multi", "RGB"
configer.usedChannels = [i + 1 for i in range(25)]  # 多光谱: 列表，包含所用通道索引(1~25)； 可见光: 字符串，"RGB"或"R", "G", "B"
configer.splitratio = [0.6, 0.2, 0.2]   # 划分比例
configer.splitcount = 1

## -------------------------- 模型相关 --------------------------
configer.modelbase = 'recognize_vgg11_bn'


## ========================== 无需修改 ==========================
configer.splitmode = 'split_{}x{}_[{:.2f}:{:.2f}:{:.2f}]_[{:d}]'.\
            format(configer.dsize[0], configer.dsize[1], 
            configer.splitratio[0], configer.splitratio[1], configer.splitratio[2], 
            configer.splitcount)
configer.n_usedChannels = len(configer.usedChannels)
configer.modelname = '[{}]_{}_[{}]'.\
                format(configer.modelbase, configer.splitmode, 
                        '_'.join(list(map(str, configer.usedChannels))) \
                            if isinstance(configer.usedChannels, list) \
                            else configer.usedChannels)

configer.logspath = '{}/{}/logs'.format(configer.savepath, configer.modelname)
configer.mdlspath = '{}/{}/models'.format(configer.savepath, configer.modelname)

