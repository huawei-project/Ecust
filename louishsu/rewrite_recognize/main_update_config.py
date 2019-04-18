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
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets import RecognizeDataset
from models import modeldict
from utiles import accuracy, getTime

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
    # 依次选择多个波段进行实验
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
            elapsed_time = 0; total_time = 0; start_time = 0
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
                start_time = time.time()
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

                    # time
                    duration_time = time.time() - start_time
                    start_time    = time.time()
                    elapsed_time += duration_time
                    total_time    = duration_time * configer.n_epoch * len(trainset) // configer.batchsize

                    # log
                    # print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || lr: {:.6f} | accuracy: {:2.2%}, loss: {:4.4f}".\
                    #         format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                    #                 i_epoch, configer.n_epoch, i_batch, len(trainset) // configer.batchsize, 
                    #                 scheduler.get_lr()[-1], acc_i, loss_i)
                    # print(print_log)

                    loss_train += [loss_i.detach().cpu().numpy()]
                    acc_train  += [acc_i.cpu().numpy()]
                
                # print('------------------------------------------------------------------------------------------------------------------')


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

                    # log
                    # print_log = "{} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || accuracy: {:2.2%}, loss: {:4.4f}".\
                    #         format(getTime(), i_epoch, configer.n_epoch, i_batch, len(validset) // configer.batchsize, acc_i, loss_i)
                    # print(print_log)

                    loss_valid += [loss_i.detach().cpu().numpy()]
                    acc_valid  += [acc_i.cpu().numpy()]

                # print('------------------------------------------------------------------------------------------------------------------')


                loss_train = np.mean(np.array(loss_train))
                acc_train  = np.mean(np.array(acc_train))
                loss_valid = np.mean(np.array(loss_valid))
                acc_valid  = np.mean(np.array(acc_valid))
                # print_log = "{} || Epoch: [{:3d}]/[{:3d}] || lr: {:.6f} || train | acc: {:2.2%}, loss: {:4.4f} || valid | acc: {:2.2%}, loss: {:4.4f}".\
                #         format(getTime(), i_epoch, configer.n_epoch, scheduler.get_lr()[-1], acc_train, loss_train, acc_valid, loss_valid)
                # print(print_log)
                
                logger.add_scalars('accuracy', {'train': acc_train,  'valid': acc_valid},  i_epoch)
                logger.add_scalars('logloss',  {'train': loss_train, 'valid': loss_valid}, i_epoch)
                logger.add_scalar('lr', scheduler.get_lr()[-1], i_epoch)

                # print('------------------------------------------------------------------------------------------------------------------')

                if loss_valid_last > loss_valid:

                    loss_valid_last = loss_valid
                    torch.save(model, modelpath)
                    # print_log = "{} || Epoch: [{:3d}]/[{:3d}] || Saved as {}".\
                    #         format(getTime(), i_epoch, configer.n_epoch, modelpath)
                    # print(print_log)

                # print('==================================================================================================================')


            test(configer)
        


if __name__ == "__main__":
    # main_split()
    # main_best_channels()
    # main_several_channels()
    main_finetune_channels()
