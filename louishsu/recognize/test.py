import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from metric import metrics
from model import models
from dataset import HyperECUST
from utiles import accuracy
from config import configer


def init_model():
    modelpath = os.path.join(configer.mdlspath, '{}.pkl'.format(configer.modelname))
    assert os.path.exists(modelpath), 'model {} does not exists! '.format(configer.modelname)
    model = torch.load(modelpath)

    if configer.cuda:
        model.cuda()

    return model, modelpath

def init_loss():
    return metrics[configer.lossname]

def test():
    batch_size = configer.batchsize
    modelname  = configer.modelname
    splitmode  = configer.splitmode
    
    if configer.trainmode == 'Multi':
        testsets = HyperECUST(configer.facesize, 'test')
    elif configer.trainmode == 'RGB':
        trainsets = RGBECUST(configer.facesize, 'test')
    testloader = DataLoader(testsets, batch_size)

    model, modelpath = init_model()
    print_log = 'load model: {}'.format(modelpath)
    print(print_log)

    loss = init_loss()

    acc_test = []; loss_test = []
    model.eval()

    output_tosave = None

    txtfile = os.path.join(configer.logspath, modelname, 'test_result_{}.txt'.format(splitmode))
    f = open(txtfile, 'w')

    for i_batch, (X, y) in enumerate(testloader):
        X = Variable(X.float())
        y = Variable(y)

        if configer.cuda:
            X = X.cuda()
            y = y.cuda()

        y_pred_prob = model(X)

        loss_test_batch = loss(y_pred_prob, y)
        acc_test_batch  = accuracy(y_pred_prob, y, multi=False)
        print_log = 'testing...     batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(i_batch+1, len(testsets)//batch_size, acc_test_batch, loss_test_batch)
        print(print_log); f.write(print_log + '\n')

        loss_test.append(loss_test_batch.detach().cpu().numpy())
        acc_test.append(acc_test_batch.cpu().numpy())

        # save outputs
        output = y_pred_prob.detach().cpu().numpy()
        if i_batch == 0:
            output_tosave = output
        else:
            output_tosave = np.concatenate([output_tosave, output], axis=0)
        

    acc_test = np.mean(np.array(acc_test))
    loss_test = np.mean(np.array(loss_test))

    print('--------------------------------------------------------------------')
    print_log = "{} || accuracy: {:2.2%}, loss: {:4.4f}".\
                    format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), acc_test, loss_test)
    print(print_log); f.write(print_log + '\n')

    npyfile = os.path.join(configer.logspath, modelname, 'test_output_{}.npy'.format(splitmode))
    np.save(npyfile, output_tosave)

    f.close()

    print('====================================================================')

