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
        # print_log = "{} || Batch: [{:3d}]/[{:3d}] || accuracy: {:2.2%}, loss: {:4.4f}".\
        #         format(getTime(), i_batch, len(testset) // configer.batchsize, acc_i, loss_i)
        # print(print_log); ftest.write(print_log + '\n')

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
    # print(print_log); ftest.write(print_log + '\n')
    np.save(os.path.join(logpath, 'test_out.npy'), output)

    # print('==================================================================================================================')
    ftest.close()


def test_samples():

    samplefiles = ['{}normal/Multi_4_W1_1', '{}illum1/Multi_4_W1_1', '{}illum2/Multi_4_W1_1', '{}normal/Multi_1_W1_1', '{}normal/Multi_4_W1_5']
    samplefiles = list(map(lambda x: x.format('/home/louishsu/Work/Workspace/ECUST2019_64x64/DATA1/1/Multi/'), samplefiles))

    ## 23 channels
    modelpath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/recognize_vgg11_bn_split_64x64_1_63subjects_models/recognize_vgg11_bn_split_64x64_1_23chs_550sta_20nm.pkl"
    model = torch.load(modelpath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    images = []
    for samplefile in samplefiles:
        images += [RecognizeDataset._load_image(samplefile, 'Multi', [550+i*20 for i in range(23)]).unsqueeze(0)]
    images = torch.cat(images, 0)
    y_pred_23chs = model(images).detach().numpy()
    
    ## single channel
    MODELPATH = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles/recognize_vgg11_bn_split_64x64_1_63subjects_models/recognize_vgg11_bn_split_64x64_1_1chs_{}sta_20nm.pkl"
    for samplefile in samplefiles:
        y_pred = []
        for ch in [550+i*20 for i in range(23)]:
            image = RecognizeDataset._load_image(samplefile, 'Multi', [ch]).unsqueeze(0)
            model = torch.load(MODELPATH.format(ch), 
                map_location='cuda' if torch.cuda.is_available() else 'cpu')
            y_pred += [model(image).detach().numpy()]
        y_pred += [y_pred_23chs[samplefiles.index(samplefile)].reshape((1, -1))]
        y_pred = np.concatenate(y_pred, axis=0)

        filename = '/home/louishsu/Desktop/' + '_'.join(samplefile.split('/')[-2: ])
        np.save(filename + '.npy', y_pred)

        # import scipy.io as io
        # io.savemat(filename + ".mat", y_pred)


if __name__ == "__main__":
	test_samples()
