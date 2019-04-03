import os
import numpy as np

import torch
import torch.nn as nn
from torch.cuda import is_available
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import RecognizeDataset
from utiles import accuracy, getTime

def test(configer):

    ## datasets
    testset = RecognizeDataset(configer.datapath, configer.datatype, configer.splitmode, 'test', configer.usedChannels)
    testloader = DataLoader(testset, configer.batchsize, shuffle=False)

    ## model
    modelpath = os.path.join(configer.mdlspath, configer.modelname) + '.pkl'
    assert os.path.exists(modelpath), 'please train first! '
    model = torch.load(modelpath)
    if configer.cuda and is_available(): model.cuda()

    ## loss
    loss = nn.CrossEntropyLoss()

    ## log
    logpath = os.path.join(configer.logspath, configer.modelname)
    ftest = open(os.path.join(logpath, 'test_log.txt'), 'w')

    ## initialize
    acc_test = []; loss_test = []
    output = None

    ## start testing
    model.eval()
    for i_batch, (X, y) in enumerate(testloader):
            
        # get batch
        X = Variable(X.float()); y = Variable(y)
        if configer.cuda and is_available():
            X = X.cuda(); y = y.cuda()

        # forward
        y_pred_prob = model(X)
        loss_i = loss(y_pred_prob, y)
        acc_i  = accuracy(y_pred_prob, y)

        # log
        print_log = "{} || Batch: [{:3d}]/[{:3d}] || accuracy: {:2.2%}, loss: {:4.4f}".\
                format(getTime(), i_batch, len(testset) // configer.batchsize, acc_i, loss_i)
        print(print_log); ftest.write(print_log + '\n')

        loss_test += [loss_i.detach().cpu().numpy()]
        acc_test  += [acc_i.cpu().numpy()]

        # save output
        if output is None:
            output = y_pred_prob.detach().cpu().numpy()
        else:
            output = np.concatenate([output, y_pred_prob.detach().cpu().numpy()], axis=0)

    print('------------------------------------------------------------------------------------------------------------------')

    loss_test = np.mean(np.array(loss_test))
    acc_test  = np.mean(np.array(acc_test))
    print_log = "{} || test | acc: {:2.2%}, loss: {:4.4f}".\
            format(getTime(), acc_test, loss_test)
    print(print_log); ftest.write(print_log + '\n')
    np.save(os.path.join(logpath, 'test_out.npy'), output)

    print('==================================================================================================================')
    ftest.close()