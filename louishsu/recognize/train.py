import os
import numpy as np
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import models
from metric import metrics
from dataset import HyperECUST, RGBECUST
from utiles import accuracy
from config import configer

def init_logger():
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = logging.DEBUG
    log_dir = os.path.join(configer.logspath, configer.modelname)
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    log_filename = os.path.join(log_dir,'{}.log'.format(configer.modelname))
    log_filemode = 'a'
    logging.basicConfig(format=log_format,
                        datefmt=log_datefmt, 
                        level=log_level,
                        filename=log_filename, 
                        filemode=log_filemode)
    logger = logging.getLogger(__name__)
    return logger

def init_model():
    modelpath = os.path.join(configer.mdlspath, '{}.pkl'.format(configer.modelname))
    if os.path.exists(modelpath):
        model = torch.load(modelpath)
    else:
        model = models[configer.modelbase]
        torch.save(model, modelpath)
    
    if configer.cuda:
        model.cuda()

    return model, modelpath

def init_loss():
    return metrics[configer.lossname]


def train():
    learning_rate  = configer.learningrate
    batch_size     = configer.batchsize
    n_epoch        = configer.n_epoch
    modelname      = configer.modelname
    logger         = init_logger()

    log_dir = os.path.join(configer.logspath, modelname)
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    if configer.trainmode == 'Multi':
        trainsets = HyperECUST(configer.facesize, 'train')
        validsets  = HyperECUST(configer.facesize, 'valid')
    elif configer.trainmode == 'RGB':
        trainsets = RGBECUST(configer.facesize, 'train')
        validsets  = RGBECUST(configer.facesize, 'valid')
    trainloader = DataLoader(trainsets, batch_size, shuffle=True)
    validloader  = DataLoader(validsets, batch_size)


    model, modelpath = init_model()
    # writer.add_graph(model, input_to_model=torch.Tensor(batch_size, configer.getint('global', 'N_CHANNLES'),
    #             eval(configer.get('global', 'N_CHANNLES'))[0], eval(configer.get('global', 'N_CHANNLES'))[1]))
    print_log = 'load model: {}'.format(modelpath)
    print(print_log); logger.debug(print_log)

    loss = init_loss()
    optimizor = optim.Adam(model.parameters(), learning_rate)
    scheduler = lr_scheduler.StepLR(optimizor, configer.stepsize, configer.gamma)

    elapsed_time = 0; total_time = 0

    acc_train_epoch = 0.; acc_valid_epoch = 0.
    loss_train_epoch = float('inf'); loss_valid_epoch = float('inf')
    loss_valid_epoch_last = loss_valid_epoch

    for i_epoch in range(n_epoch):
        scheduler.step(i_epoch)

        acc_train_epoch = []; acc_valid_epoch = []
        loss_train_epoch = []; loss_valid_epoch = []

        start_time = time.time()

        model.train()
        for i_batch, (X, y) in enumerate(trainloader):

            X = Variable(X.float())
            y = Variable(y)

            if configer.cuda:
                X = X.cuda()
                y = y.cuda()

            y_pred_prob = model(X)

            loss_train_batch = loss(y_pred_prob, y)
            optimizor.zero_grad()
            loss_train_batch.backward() 
            optimizor.step()

            duration_time = time.time() - start_time
            start_time    = time.time()
            elapsed_time += duration_time
            total_time    = duration_time * configer.n_epoch * len(trainsets) // configer.batchsize

            acc_train_batch  = accuracy(y_pred_prob, y, multi=False)
            print_log = 'training...   || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || epoch [{:3d}]/[{:3d}] | batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(elapsed_time / 3600, (total_time - elapsed_time) / 3600, configer.batchsize / duration_time,
                                    i_epoch+1, n_epoch, i_batch+1, len(trainsets)//batch_size, 
                                    acc_train_batch, loss_train_batch)
            print(print_log); logger.debug(print_log)

            acc_train_epoch.append(acc_train_batch.cpu().numpy())
            loss_train_epoch.append(loss_train_batch.detach().cpu().numpy())
        
        acc_train_epoch = np.mean(np.array(acc_train_epoch))
        loss_train_epoch = np.mean(np.array(loss_train_epoch))
        
        print_log = '-------------------------------------------------------------------------------------------------------------------------------------------------'

        model.eval()
        for i_batch, (X, y) in enumerate(validloader):
            X = Variable(X.float())
            y = Variable(y)

            if configer.cuda:
                X = X.cuda()
                y = y.cuda()
                
            y_pred_prob = model(X)

            loss_valid_batch = loss(y_pred_prob, y)
            acc_valid_batch  = accuracy(y_pred_prob, y, multi=False)
            print_log = 'validating...  epoch [{:3d}]/[{:3d}] | batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(i_epoch+1, n_epoch, i_batch+1, len(validsets)//batch_size, acc_valid_batch, loss_valid_batch)
            print(print_log); logger.debug(print_log)

            acc_valid_epoch.append(acc_valid_batch.cpu().numpy())
            loss_valid_epoch.append(loss_valid_batch.detach().cpu().numpy())
        

        acc_valid_epoch = np.mean(np.array(acc_valid_epoch))
        loss_valid_epoch = np.mean(np.array(loss_valid_epoch))

        writer.add_scalars('accuracy', {'train': acc_train_epoch,  'valid': acc_valid_epoch},  i_epoch)
        writer.add_scalars('logloss',  {'train': loss_train_epoch, 'valid': loss_valid_epoch}, i_epoch)
        writer.add_scalar('lr', scheduler.get_lr()[-1], i_epoch)

        print_log = '-------------------------------------------------------------------------------------------------------------------------------------------------'
        print(print_log); logger.debug(print_log)
        print_log = 'epoch [{:3d}]/[{:3d}] || training: accuracy: {:2.2%}, loss: {:4.4f} | validing: accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(i_epoch, n_epoch, acc_train_epoch, loss_train_epoch, acc_valid_epoch, loss_valid_epoch)
        print(print_log); logger.debug(print_log)


        if loss_valid_epoch_last > loss_valid_epoch:
            torch.save(model, modelpath)
            acc_train_epoch_last = acc_train_epoch; acc_valid_epoch_last = acc_valid_epoch
            loss_train_epoch_last = loss_train_epoch; loss_valid_epoch_last = loss_valid_epoch
            print_log = 'model saved!'
            print(print_log); logger.debug(print_log)


        print_log = '================================================================================================================================================='
        print(print_log); logger.debug(print_log)
