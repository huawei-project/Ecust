# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-16 16:03:14
@LastEditTime: 2019-09-17 13:57:44
@Update: 
'''
import os
import time
import numpy as np

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
from utils import accuracy, getTime, is_with_no_glasses, is_with_no_sunglasses

def train(configer):
    """
    Update:
        2019.04.24: 固定权值
    """
    condition = None
    if configer.training_no_glass:
        condition = is_with_no_glasses
    if configer.training_no_sunglass:
        condition = is_with_no_sunglasses

    ## datasets
    trainset = RecognizeDataset(configer.datapath, configer.datatype, 
            configer.splitmode, 'train', configer.usedChannels, 
            dsize=configer.dsize, hist=configer.hist, condition=condition)
    validset = RecognizeDataset(configer.datapath, configer.datatype, 
            configer.splitmode, 'valid', configer.usedChannels, 
            dsize=configer.dsize, hist=configer.hist, condition=condition)
    trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
    validloader = DataLoader(validset, configer.batchsize, shuffle=False)

    ## model: pre-initialized
    modelpath = os.path.join(configer.mdlspath, configer.modelname) + '.pkl'
    modeldir  = '/'.join(modelpath.split('/')[:-1])
    if not os.path.exists(modeldir): os.makedirs(modeldir)
    preInitdir = os.path.join('/'.join(configer.mdlspath.split('/')[:-2]), "preinit")
    if not os.path.exists(preInitdir): os.makedirs(preInitdir)
    preInitmodelpath = os.path.join(preInitdir, configer.modelbase + '.pkl')

    model = modeldict[configer.modelbase](configer.n_usedChannels, configer.n_class, configer.dsize[0])
    if not os.path.exists(preInitmodelpath):
        model_state = model.state_dict()
        torch.save(model_state, preInitmodelpath)
    else:
        preinit_state = torch.load(preInitmodelpath)
        model_state = model.state_dict()
        toload_state = {k: v for k, v in preinit_state.items() \
                    if preinit_state[k].shape==model_state[k].shape}
        model_state.update(toload_state)
        model.load_state_dict(model_state)

    if configer.cuda and is_available(): model.cuda()
    torch.save(model, modelpath)

    ## loss
    loss = nn.CrossEntropyLoss()

    ## optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, configer.lrbase, weight_decay=1e-3)

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
    logger.close()

    
if __name__ == "__main__":
    
    from config import configer

    train(configer)