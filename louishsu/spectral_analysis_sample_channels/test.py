import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import metrics
from datasets import HyperECUST
from utiles import accuracy
from config import configer


def init_model():
    modelpath = os.path.join(configer.mdlspath, '{}.pkl'.format(configer.modelname))
    assert os.path.exists(modelpath), 'model {} does not exists! '.format(configer.modelname)
    model = torch.load(modelpath)
    return model, modelpath

def init_loss():
    return metrics._losses[configer.lossname]

def test():
    batch_size = configer.batchsize
    modelname  = configer.modelname
    splitmode  = configer.splitmode
    
    testsets = HyperECUST(splitmode, configer.facesize, 'test')
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
        y_pred_prob = model(X)

        loss_test_batch = loss(y_pred_prob, y)
        acc_test_batch  = accuracy(y_pred_prob, y, multi=False)
        print_log = 'testing...     batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(i_batch+1, len(testsets)//batch_size, acc_test_batch, loss_test_batch)
        print(print_log); f.write(print_log + '\n')

        loss_test.append(loss_test_batch.detach().numpy())
        acc_test.append(acc_test_batch.numpy())

        # save outputs
        output = y_pred_prob.detach().numpy()
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

def test_pos4():
    batch_size = configer.batchsize
    modelname  = configer.modelname
    splitmode  = configer.splitmode
    
    testsets = HyperECUST(splitmode, configer.facesize, 'test_pos4')
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
        y_pred_prob = model(X)

        loss_test_batch = loss(y_pred_prob, y)
        acc_test_batch  = accuracy(y_pred_prob, y, multi=False)
        print_log = 'testing...     batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(i_batch+1, len(testsets)//batch_size, acc_test_batch, loss_test_batch)
        print(print_log); f.write(print_log + '\n')

        loss_test.append(loss_test_batch.detach().numpy())
        acc_test.append(acc_test_batch.numpy())

        # save outputs
        output = y_pred_prob.detach().numpy()
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

    npyfile = os.path.join(configer.logspath, modelname, 'test_output_{}_pos4.npy'.format(splitmode))
    np.save(npyfile, output_tosave)

    f.close()

    print('====================================================================')


if __name__ == "__main__":
    test_pos4()