# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-17 09:32:21
@LastEditTime: 2019-09-17 10:45:49
@Update: 
'''
import os
import cv2
import scipy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mobilefacenet import MobileFacenet

# TODO:
featurepath = '/home/louishsu/Work/Workspace/features'
pklfile  = '{}/Casia+HyperECUST_HyperECUST/MobileFacenet_best.pkl'.format(featurepath)
splittxt = '{}/Casia+HyperECUST_HyperECUST/face_verification_split_3f/split_1/train_multi_normal.txt'.format(featurepath)
datapath = '/datasets/Indoordetect'
prefix = ':/:/:/:/:/:'
savepath = '{}/Casia+HyperECUST_HyperECUST/training.mat'.format(featurepath)

states = torch.load(pklfile, map_location='cpu')

net = MobileFacenet(10)
net.load_state_dict(states['net'])
net.cuda()

class Data(Dataset):

    def __init__(self):
        
        with open(splittxt, 'r') as f:
            lines = f.readlines()
        self.lines = list(map(lambda x: x.strip().split('\t'), lines))
        # lines = list(filter(lambda x: x[0].split('/')[-1] == '1.jpg', lines))

    def __getitem__(self, index):

        filename, label = self.lines[index]
        image = cv2.imread("{}/{}".format(datapath, filename))
        image = (image - 127.5) / 128.0   
        image = np.transpose(image, [2, 0, 1])

        return image, label, filename

    def __len__(self):

        return len(self.lines)

dataset = Data()
dataloader = Dataloader(dataset, 64, shuffle=False)

features = []; filenames = []
with torch.no_grad():
    for i, (X, y, filename) in enumerate(dataloader):

        X = Variable(X.float()).cuda(); y = Variable(y.float()).cuda()
        feature = net.get_feature(X)

        features += [feature.cpu().numpy()]
        filenames += ['{}/{}'.format(prefix, filename)]

matdict = {
        'featuresLs': np.array(features),
        'featuresRs': np.array(features),
        'filenameLs': np.array(filenames),
        'filenameRs': np.array(filenames),
        }
scipy.io.savemat(savepath, matdict)
