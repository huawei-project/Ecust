import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from config import configer
from utiles import get_label_from_path, get_vol, get_wavelen

def load_multi(imgdir, dsize=None):
    assert os.path.exists(imgdir), "multi directory does not exist!"
    imgfiles = os.listdir(imgdir)
    
    # 根据波长排序
    wavelength = []
    for imgfile in imgfiles:
        wavelength += [int(imgfile.split('.')[0].split('_')[-1])]
    imgfiles = np.array(imgfiles); wavelength = np.array(wavelength)
    imgfiles = imgfiles[np.argsort(wavelength)]
    imgfiles = list(imgfiles)
    
    # 载入图像
    c = len(imgfiles)
    if dsize is None:
        img = cv2.imread(os.path.join(imgdir, imgfiles[0]), cv2.IMREAD_GRAYSCALE)
        (h, w) = img.shape
    else:
        (w, h) = dsize
    imgs = np.zeros(shape=(h, w, c), dtype='uint8')
    for i in range(c):
        img = cv2.imread(os.path.join(imgdir, imgfiles[i]), cv2.IMREAD_GRAYSCALE)
        imgs[:, :, i] = img if (dsize is None) else cv2.resize(img, dsize)
    return imgs

def resizeMulti(image, dsize):
    """
    Params:
        image: {ndarray(H, W, C)}
        dsize: {tuple(H, W)}
    """
    c = image.shape[-1]
    ret = np.zeros(shape=(dsize[0], dsize[1], c))
    for i in range(c):
        ret[:, :, i] = cv2.resize(image[:, :, i], dsize[::-1])
    return ret

def getDicts():
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 5)]:
        txtfile = os.path.join(configer.datapath, vol, "detect.txt")
        with open(txtfile, 'r') as f:
            dicts[vol] = eval(f.read())
    return dicts

def splitData(train=0.6, valid=0.2, test=0.2):
    get_vol = lambda i: (i-1)//10+1

    FILENAME = "DATA{}/{}/Multi/{}/Multi_{}_W1_{}"
    obtypes = ["non-obtructive", "obtructive/ob1", "obtructive/ob2"]
    subidx  = [i for i in range(1, 41)]
    posidx  = [i for i in range(1,  8)]
    sessidx = [i for i in range(1,  10)]

    dicts = getDicts()
    datapath = "/home/louishsu/Work/Workspace/ECUST2019"
    splitdir = "./split/{}".format(configer.splitmode)
    if not os.path.exists(splitdir): os.mkdir(splitdir)
    train_txt = "{}/train.txt".format(splitdir)
    valid_txt = "{}/valid.txt".format(splitdir)
    test_txt  = "{}/test.txt".format(splitdir)
    ftrain = open(train_txt, 'w'); fvalid = open(valid_txt, 'w'); ftest  = open(test_txt, 'w')

    n_train, n_valid, n_test = 0, 0, 0

    for i in subidx:
        subfiles = []
        for pos in posidx:
            for obtype in obtypes:
                for sess in sessidx:
                    vol = get_vol(i)
                    filename = FILENAME.format(vol, i, obtype, pos, sess)
                    dict = dicts['DATA%d' % vol]

                    if obtype == "non-obtructive":
                        key = '/' + '/'.join(filename.split('/')[-4:])
                    else:
                        key = '/' + '/'.join(filename.split('/')[-5:])
                    
                    filepath = os.path.join(datapath, filename)
                    if os.path.exists(filepath)\
                            and (key in dict.keys()) \
                                and (dict[key][0] is not None):
                        subfiles += [filename + '\n']
        
        i_items = len(subfiles)
        i_train = int(i_items*train); n_train += i_train
        i_valid = int(i_items*valid); n_valid += i_valid
        i_test  = i_items - i_train - i_valid; n_test += i_test
        trainfiles = random.sample(subfiles, i_train)
        subfiles_valid_test = [f for f in subfiles if f not in trainfiles]
        validfiles = random.sample(subfiles_valid_test, i_valid)
        testfiles  = [f for f in subfiles_valid_test if f not in validfiles]

        ftrain.writelines(trainfiles)
        fvalid.writelines(validfiles)
        ftest.writelines(testfiles)

    n_items = n_train + n_valid + n_test
    print("number of samples: {}".format(n_items))
    print("number of train: {:5d}, ".format(n_train))
    print("number of valid: {:5d}, ".format(n_valid))
    print("number of test:  {:5d}, ".format(n_test ))
    print("train: valid: test:  {:.3f}: {:.3f}: {:.3f}".\
                    format(n_train / n_items, n_valid / n_items, n_test  / n_items))
    
    ftrain.close(); fvalid.close(); ftest.close()


class HyperECUST(Dataset):
    labels = [i for i in range(1, 41)]
    
    def __init__(self, facesize=None, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid', 'test'
        """
        with open('./split/{}/{}.txt'.format(configer.splitmode, mode), 'r') as f:
            self.filenames = f.readlines()
        self.filenames = [os.path.join(configer.datapath, filename) for filename in self.filenames]
        self.facesize = tuple(facesize)
        self.dicts = getDicts()

    def __getitem__(self, index):
        filename = self.filenames[index].strip()
        label = get_label_from_path(filename)
        dict_wave_bmp = {get_wavelen(bmp): filename + '/' + bmp for bmp in os.listdir(filename)}

        # get bbox
        vol = "DATA%d" % get_vol(label)
        imgname = filename[filename.find("DATA")+5:]
        bbox = self.dicts[vol][imgname][1]
        [x1, y1, x2, y2] = bbox

        # load image
        h, w, c = self.facesize[0], self.facesize[1], len(configer.usedChannels)
        image = np.zeros(shape=(h, w, c))
        for i in range(len(configer.usedChannels)):
            ch = configer.usedChannels[i]
            im = cv2.imread(dict_wave_bmp[ch], cv2.IMREAD_GRAYSCALE)
            im = im[y1: y2, x1: x2]
            image[:, :, i] = cv2.resize(im, self.facesize[::-1])
        image = ToTensor()(image)

        # get label
        label = self.labels.index(label)
        
        return image, label
    
    def __len__(self):
        return len(self.filenames)

if __name__ == "__main__":
    # splitData(0.6, 0.1, 0.3)
    D = HyperECUST((64, 64), 'train')
    for i in range(len(D)):
        X, y = D[i]
