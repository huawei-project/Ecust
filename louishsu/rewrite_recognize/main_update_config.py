from train import train
from test  import test
from gen_excel import gen_out_excel
from easydict import EasyDict
from utiles import getTime
import time

def main_split():

    ## 选出适当的划分比例

    for splitidx in range(6, 36):
        for datatype in ['Multi', 'RGB']:
             
            print(getTime(), splitidx, datatype, '...')

            configer = EasyDict()

            configer.dsize = (64, 64)
            configer.datatype = datatype
            configer.n_epoch =   300 if datatype == 'Multi' else 350
            configer.lrbase  = 0.001 if datatype == 'Multi' else 0.0005

            configer.n_channel = 23
            configer.n_class = 63
            configer.batchsize = 32
            configer.stepsize = 250
            configer.gamma = 0.2
            configer.cuda = True


            configer.splitmode = 'split_{}x{}_{}'.format(configer.dsize[0], configer.dsize[1], splitidx)
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


            configer.datapath = '/datasets/ECUST2019_{}x{}'.\
                                            format(configer.dsize[0], configer.dsize[1])
            configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)
            configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)

            train(configer)
            test(configer)
            gen_out_excel(configer)

def main_best_channels():

    # 波段选择依据
    # 以最佳的划分方式: 
    # 依次选择每个波段进行实验

    for splitidx in range(6, 36):   # TODO
        for datatype in ['Multi', 'RGB']:

            if datatype == 'Multi':
                usedChannelsList = [[i] for i in range(23)]
            else:
                usedChannelsList = ['R', 'G', 'B']

            for usedChannels in usedChannelsList:
                
                print(getTime(), splitidx, datatype, usedChannels, '...')

                configer = EasyDict()

                configer.dsize = (64, 64)
                configer.datatype = datatype
                configer.n_epoch =   300 if datatype == 'Multi' else 350
                configer.lrbase  = 0.001 if datatype == 'Multi' else 0.0005

                configer.n_channel = 23
                configer.n_class = 63
                configer.batchsize = 32
                configer.stepsize = 250
                configer.gamma = 0.2
                configer.cuda = True


                configer.splitmode = 'split_{}x{}_{}'.format(configer.dsize[0], configer.dsize[1], splitidx)
                configer.modelbase = 'recognize_vgg11_bn'


                if configer.datatype == 'Multi':
                    configer.usedChannels = usedChannels
                    configer.n_usedChannels = len(configer.usedChannels)
                    configer.modelname = '{}_{}_{}'.\
                                    format(configer.modelbase, configer.splitmode, 
                                            '_'.join(list(map(str, configer.usedChannels))))
                elif configer.datatype == 'RGB':
                    configer.usedChannels = usedChannels
                    configer.n_usedChannels = len(configer.usedChannels)
                    configer.modelname = '{}_{}_{}'.\
                                    format(configer.modelbase, configer.splitmode, configer.usedChannels)


                configer.datapath = '/datasets/ECUST2019_{}x{}'.\
                                                format(configer.dsize[0], configer.dsize[1])
                configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                                format(configer.modelbase, configer.splitmode, configer.n_class)
                configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                                format(configer.modelbase, configer.splitmode, configer.n_class)

                train(configer)
                test(configer)
                gen_out_excel(configer)
        

def main_several_channels():

    # 波段选择依据
    # 以最佳的划分方式: 0.6: 0.2: 0.2(5个)
    # 最优的波段排序: 
    #       [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    # 依次选择多个波段进行实验
    CHANNEL_SORT = [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    
    for splitidx in range(1, 6):
        usedChannelsList = [CHANNEL_SORT[:i+2] for i in range(23)]

        for usedChannels in usedChannelsList:
            
            print(getTime(), splitidx, len(usedChannels), '...')

            configer = EasyDict()

            configer.dsize = (64, 64)
            configer.datatype = 'Multi'
            configer.n_epoch   = 300 if configer.datatype == 'Multi' else 350
            configer.lrbase = 0.001  if configer.datatype == 'Multi' else 0.0005

            configer.n_channel = 23
            configer.n_class = 63
            configer.batchsize = 32
            configer.stepsize = 250
            configer.gamma = 0.2
            configer.cuda = True


            configer.splitmode = 'split_{}x{}_{}'.format(configer.dsize[0], configer.dsize[1], splitidx)
            configer.modelbase = 'recognize_vgg11_bn'


            if configer.datatype == 'Multi':
                configer.usedChannels = usedChannels
                configer.n_usedChannels = len(configer.usedChannels)
                configer.modelname = '{}_{}_{}'.\
                                format(configer.modelbase, configer.splitmode, 
                                        '_'.join(list(map(str, configer.usedChannels))))
            elif configer.datatype == 'RGB':
                configer.usedChannels = 'RGB'
                configer.n_usedChannels = len(configer.usedChannels)
                configer.modelname = '{}_{}_{}'.\
                                format(configer.modelbase, configer.splitmode, configer.usedChannels)


            configer.datapath = '/datasets/ECUST2019_{}x{}'.\
                                            format(configer.dsize[0], configer.dsize[1])
            configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)
            configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)

            train(configer)
            test(configer)
            gen_out_excel(configer)
        


if __name__ == "__main__":
    main_split()
    main_best_channels()
    main_several_channels()
