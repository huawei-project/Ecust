import os
import time
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import is_available, empty_cache
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from datasets import RecognizeDataset
from models import modeldict
from utiles import accuracy, getTime, getLabel

from train import train
from test  import test

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
    # 最优的波段排序: 
    #       [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    # 依次选择多个波段组合进行实, 组合的意思是[[850], [850, 870], [850, 870, 930], ..., [850, ..., 550]]
    CHANNEL_SORT = [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    
    for splitidx in range(31, 36):
        usedChannelsList = [CHANNEL_SORT[:i+1] for i in range(23)]

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


def main_several_channels_k_fold(k=5):

    # 波段选择依据
    # 最优的波段排序: 
    #       [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    # 依次选择多个波段组合进行实, 组合的意思是[[850], [850, 870], [850, 870, 930], ..., [850, ..., 550]]
    # 每组波段下进行5折交叉验证
    # 读取`split_64x64_1`中的`train/valid/test.txt`,按顺序划分为k折

    class KFoldDataset(Dataset):
        def __init__(self, datapath, filelist, usedChannels):
            filelist = list(map(lambda x: os.path.join('/'.join(datapath.split('/')[:-1]), x.strip()), filelist))
            self.samplelist = list(map(lambda x: [RecognizeDataset._load_image(x, 'Multi', usedChannels), getLabel(x)-1], filelist))
        def __getitem__(self, index):
            image, label = self.samplelist[index]
            return image, label
        def __len__(self):
            return len(self.samplelist)        


    CHANNEL_SORT = [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    usedChannelsList = [CHANNEL_SORT[:i+1] for i in range(23)]

    ## 读取所有文件
    filelist = []
    for mode in ['train', 'valid', 'test']:
        with open('./split/split_64x64_1/{}.txt'.format(mode), 'r') as f:
            filelist += f.readlines()

    ## 划分为k折
    n_files_fold = len(filelist) // k
    foldlist = []
    for i in range(k-1):
        foldlist += [filelist[i*n_files_fold: (i+1)*n_files_fold]]
    foldlist += [filelist[(k-1)*n_files_fold: ]]

    for i in range(k):

        ## k折交叉验证
        validlist = foldlist[i]
        trainlist = list(filter(lambda x: x not in validlist, filelist))

        for i_usedChannels in range(len(usedChannelsList)):
            usedChannels = usedChannelsList[i_usedChannels]
            
            print(getTime(), '[', i, '/', k, ']', len(usedChannels), '...')

            configer = EasyDict()
            configer.dsize = (64, 64)
            configer.datatype = 'Multi'
            configer.n_epoch   = 300
            configer.lrbase = 0.001
            configer.n_channel = 23
            configer.n_class = 63
            configer.batchsize = 32
            configer.stepsize = 250
            configer.gamma = 0.2
            configer.cuda = True
            configer.splitmode = 'split_{}x{}_1'.format(configer.dsize[0], configer.dsize[1])
            configer.modelbase = 'recognize_vgg11_bn'
            configer.usedChannels = usedChannels
            configer.n_usedChannels = len(configer.usedChannels)
            configer.modelname = '{}_{}_{}_[{}_{}]fold'.\
                            format(configer.modelbase, configer.splitmode, 
                                    '_'.join(list(map(str, configer.usedChannels))), i+1, k)
            configer.datapath = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.\
                                            format(configer.dsize[0], configer.dsize[1])
            configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)
            configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)

            ## datasets
            trainset = KFoldDataset(configer.datapath, trainlist, usedChannels)
            validset = KFoldDataset(configer.datapath, validlist,  usedChannels)
            trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
            validloader = DataLoader(validset, configer.batchsize, shuffle=False)

            ## model
            modelpath = os.path.join(configer.mdlspath, configer.modelname) + '.pkl'
            modeldir  = '/'.join(modelpath.split('/')[:-1])
            if not os.path.exists(modeldir): os.makedirs(modeldir)
            model = modeldict[configer.modelbase](configer.n_usedChannels, configer.n_class, configer.dsize[0])
            if configer.cuda and is_available(): model.cuda()

            ## loss
            loss = nn.CrossEntropyLoss()
            params = model.parameters()
            optimizer = optim.Adam(params, configer.lrbase, weight_decay=1e-3)
            scheduler = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
            logpath = os.path.join(configer.logspath, configer.modelname)
            if not os.path.exists(logpath): os.makedirs(logpath)
            logger = SummaryWriter(logpath)

            ## initialize
            acc_train = 0.
            acc_valid = 0.
            loss_train = float('inf')
            loss_valid = float('inf')
            loss_valid_last = float('inf')

            ## start training
            for i_epoch in range(configer.n_epoch):

                if configer.cuda and is_available(): empty_cache()
                scheduler.step(i_epoch)
                acc_train = []; acc_valid = []
                loss_train = []; loss_valid = []

                model.train()
                for i_batch, (X, y) in enumerate(trainloader):
                    
                    # get batch
                    X = Variable(X.float()); y = Variable(y)
                    if configer.cuda and is_available():
                        X = X.cuda(); y = y.cuda()

                    # forward
                    y_pred_prob = model(X)
                    loss_i = loss(y_pred_prob, y)
                    acc_i  = accuracy(y_pred_prob, y)

                    # backward
                    optimizer.zero_grad()
                    loss_i.backward() 
                    optimizer.step()
                    
                    loss_train += [loss_i.detach().cpu().numpy()]
                    acc_train  += [acc_i.cpu().numpy()]

                loss_train = np.mean(np.array(loss_train))
                acc_train  = np.mean(np.array(acc_train))
                
                logger.add_scalar('accuracy', acc_train,  i_epoch)
                logger.add_scalar('logloss',  loss_train, i_epoch)
                logger.add_scalar('lr', scheduler.get_lr()[-1], i_epoch)

            ## start testing
            model.eval()
            loss_test = []
            acc_test  = []
            output = None
            for i_batch, (X, y) in enumerate(validloader):
                # get batch
                X = Variable(X.float()); y = Variable(y)
                if configer.cuda and is_available():
                    X = X.cuda(); y = y.cuda()
                # forward
                y_pred_prob = model(X)
                loss_i = loss(y_pred_prob, y)
                acc_i  = accuracy(y_pred_prob, y)
                # log
                loss_test += [loss_i.detach().cpu().numpy()]
                acc_test  += [acc_i.cpu().numpy()]

                # save output
                if output is None:
                    output = y_pred_prob.detach().cpu().numpy()
                else:
                    output = np.concatenate([output, y_pred_prob.detach().cpu().numpy()], axis=0)

            # print('------------------------------------------------------------------------------------------------------------------')

            loss_test = np.mean(np.array(loss_test))
            acc_test  = np.mean(np.array(acc_test))
            print_log = "{} || test | acc: {:2.2%}, loss: {:4.4f}".\
                    format(getTime(), acc_test, loss_test)
            print(print_log)
            with open(os.path.join(logpath, 'test_log.txt'), 'w') as  f:
                f.write(print_log + '\n')
            np.save(os.path.join(logpath, 'test_out.npy'), output)





def main_finetune_channels():

    # 波段选择依据
    # 最优的波段排序: 
    #       [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    # 依次增加一个波段, 前一个模型进行微调

    CHANNEL_SORT = [850, 870, 930, 730, 790, 910, 770, 750, 670, 950, 990, 830, 890, 810, 970, 690, 710, 650, 590, 570, 630, 610, 550]
    
    for splitidx in range(1, 6):
        usedChannelsList = [CHANNEL_SORT[:i+1] for i in range(23)]

        for i_usedChannels in range(len(usedChannelsList)):

            usedChannels = usedChannelsList[i_usedChannels]

            print(getTime(), splitidx, len(usedChannels), '...')

            configer = EasyDict()

            configer.dsize = (64, 64)
            configer.datatype = 'Multi'
            configer.n_epoch   = 300
            configer.lrbase = 0.001

            configer.n_channel = 23
            configer.n_class = 63
            configer.batchsize = 32
            configer.stepsize = 250
            configer.gamma = 0.2
            configer.cuda = True

            configer.splitmode = 'split_{}x{}_{}'.format(configer.dsize[0], configer.dsize[1], splitidx)
            configer.modelbase = 'recognize_vgg11_bn'

            configer.usedChannels = usedChannels
            configer.n_usedChannels = len(configer.usedChannels)
            configer.modelname = '{}_{}_{}_finetune'.\
                            format(configer.modelbase, configer.splitmode, 
                                    '_'.join(list(map(str, configer.usedChannels))))


            configer.datapath = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.\
                                            format(configer.dsize[0], configer.dsize[1])
            configer.logspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs/{}_{}_{}subjects_logs'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)
            configer.mdlspath = '/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/{}_{}_{}subjects_models'.\
                                            format(configer.modelbase, configer.splitmode, configer.n_class)


            ## datasets
            trainset = RecognizeDataset(configer.datapath, configer.datatype, configer.splitmode, 'train', configer.usedChannels)
            validset = RecognizeDataset(configer.datapath, configer.datatype, configer.splitmode, 'valid', configer.usedChannels)
            trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
            validloader = DataLoader(validset, configer.batchsize, shuffle=False)


            ## ============================================================================================
            ## model
            modelpath = os.path.join(configer.mdlspath, configer.modelname) + '.pkl'
            modeldir  = '/'.join(modelpath.split('/')[:-1])
            if not os.path.exists(modeldir): os.makedirs(modeldir)
            
            if i_usedChannels == 0:
                model = modeldict[configer.modelbase](configer.n_usedChannels, configer.n_class, configer.dsize[0])
                params = model.parameters()
                torch.save(model, modelpath)
            else:
                modelpath_pretrain = os.path.join(
                    modeldir, '{}_{}_{}_finetune.pkl'.format(configer.modelbase, configer.splitmode, 
                                                '_'.join(list(map(str, usedChannelsList[i_usedChannels-1])))))
                model = torch.load(modelpath_pretrain)
                model.features[0] = nn.Conv2d(len(usedChannels), 64, 3, stride=1, padding=1)
                params = [
                    {'params': model.features[1:].parameters(), 'lr': configer.lrbase*0.01, },
                    {'params': model.features[0].parameters(),}
                ]
                torch.save(model, modelpath)
            if configer.cuda and is_available(): model.cuda()
            ## ============================================================================================


            ## optimizer
            optimizer = optim.Adam(params, configer.lrbase, weight_decay=1e-3)
            
            ## loss
            loss = nn.CrossEntropyLoss()

            ## learning rate scheduler
            scheduler = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
            
            ## log
            logpath = os.path.join(configer.logspath, configer.modelname)
            if not os.path.exists(logpath): os.makedirs(logpath)
            logger = SummaryWriter(logpath)

            ## initialize
            acc_train = 0.
            acc_valid = 0.
            loss_train = float('inf')
            loss_valid = float('inf')
            loss_valid_last = float('inf')


            ## start training
            for i_epoch in range(configer.n_epoch):

                if configer.cuda and is_available(): empty_cache()
                scheduler.step(i_epoch)
                acc_train = []; acc_valid = []
                loss_train = []; loss_valid = []


                model.train()
                for i_batch, (X, y) in enumerate(trainloader):
                    
                    # get batch
                    X = Variable(X.float()); y = Variable(y)
                    if configer.cuda and is_available():
                        X = X.cuda(); y = y.cuda()

                    # forward
                    y_pred_prob = model(X)
                    loss_i = loss(y_pred_prob, y)
                    acc_i  = accuracy(y_pred_prob, y)

                    # backward
                    optimizer.zero_grad()
                    loss_i.backward() 
                    optimizer.step()

                    loss_train += [loss_i.detach().cpu().numpy()]
                    acc_train  += [acc_i.cpu().numpy()]
                
                model.eval()
                for i_batch, (X, y) in enumerate(validloader):
                    
                    # get batch
                    X = Variable(X.float()); y = Variable(y)
                    if configer.cuda and is_available():
                        X = X.cuda(); y = y.cuda()

                    # forward
                    y_pred_prob = model(X)
                    loss_i = loss(y_pred_prob, y)
                    acc_i  = accuracy(y_pred_prob, y)

                    loss_valid += [loss_i.detach().cpu().numpy()]
                    acc_valid  += [acc_i.cpu().numpy()]

                loss_train = np.mean(np.array(loss_train))
                acc_train  = np.mean(np.array(acc_train))
                loss_valid = np.mean(np.array(loss_valid))
                acc_valid  = np.mean(np.array(acc_valid))
                
                logger.add_scalars('accuracy', {'train': acc_train,  'valid': acc_valid},  i_epoch)
                logger.add_scalars('logloss',  {'train': loss_train, 'valid': loss_valid}, i_epoch)
                logger.add_scalar('lr', scheduler.get_lr()[-1], i_epoch)

                if loss_valid_last > loss_valid:

                    loss_valid_last = loss_valid
                    torch.save(model, modelpath)


            test(configer)


if __name__ == "__main__":
    # main_split()
    # main_best_channels()
    # main_several_channels()
    main_several_channels_k_fold()
    # main_finetune_channels()
