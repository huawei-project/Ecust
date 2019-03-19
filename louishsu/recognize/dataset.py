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
    dicts = getDicts()

    FILENAME = "/{}/Multi/{}/Multi_{}_W1_{}"
    obtypes = ["non-obtructive", "obtructive/ob1", "obtructive/ob2"]
    posidx  = [i for i in range(1, 8)]
    sessidx = [i for i in range(1, 7)]

    all_files = []
    for i in range(1, 34):
        i_files = []
        for obtype in obtypes:
            for pos in posidx:
                for sess in sessidx:
                    vol = get_vol(i)
                    filename = FILENAME.format(i, obtype, pos, sess)
                    filepath = "{}/DATA{}".format(configer.datapath, vol) + filename
                    if not os.path.exists(filepath): continue
                    if dicts["DATA%d"%vol][filename][0] is None: continue
                    i_files += [filepath]
        all_files += [i_files]
    
    trainfiles = []; validfiles = []; testfiles = []
    for i in range(33):
        i_files = len(all_files[i])
        i_train = int(i_files*train)
        i_valid = int(i_files*valid)
        i_test  = i_files - i_train - i_valid
        
        i_test_valid_files = random.sample(all_files[i], i_test + i_valid)
        i_train_files = [f for f in all_files[i] if f not in i_test_valid_files]
        i_test_files  = random.sample(i_test_valid_files, i_test)
        i_valid_files = [f for f in i_test_valid_files if f not in i_test_files]

        trainfiles += i_train_files
        validfiles += i_valid_files
        testfiles  += i_test_files
    
    n_train = len(trainfiles)
    n_valid = len(validfiles)
    n_test  = len(testfiles )
    n_files = n_train + n_valid + n_test
    print("train: valid: test = {}: {}: {} = {}: {}: {}".format(n_train, n_valid, n_test, n_train/n_files, n_valid/n_files, n_test/n_files))

    splitdir = "./split/{}".format(configer.splitmode)
    if not os.path.exists(splitdir): os.mkdir(splitdir)
    with open("{}/train.txt".format(splitdir), 'w') as f:
        trainfiles = [i + '\n' for i in trainfiles]
        f.writelines(trainfiles)
    with open("{}/valid.txt".format(splitdir), 'w') as f:
        validfiles = [i + '\n' for i in validfiles]
        f.writelines(validfiles)
    with open("{}/test.txt".format(splitdir), 'w') as f:
        testfiles  = [i + '\n' for i in testfiles ]
        f.writelines(testfiles)

class HyperECUST(Dataset):
    labels = [i for i in range(1, 34)]
    
    def __init__(self, facesize=None, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid', 'test'
        """
        with open('./split/{}/{}.txt'.format(configer.splitmode, mode), 'r') as f:
            self.filenames = f.readlines()
        self.filenames = ['/datasets/'+filename[filename.find('ECUST'):] for filename in self.filenames]
        self.facesize = tuple(facesize)
        self.dicts = getDicts()

    def __getitem__(self, index):
        filename = self.filenames[index].strip()
        label = get_label_from_path(filename)

        # get bbox
        vol = "DATA%d" % get_vol(label)
        imgname = filename[filename.find("DATA")+5:]
        bbox = self.dicts[vol][imgname][1]
        [x1, y1, x2, y2] = bbox

        # load image array
        image = load_multi(filename)[y1: y2, x1: x2]
        if self.facesize is not None:
            image = resizeMulti(image, self.facesize)

        # select channels
        image = image[:, :, configer.usedChannels]

        image = ToTensor()(image)
        label = self.labels.index(label)
        return image, label
    
    def __len__(self):
        return len(self.filenames)

if __name__ == "__main__":
    splitData(0.5, 0.3, 0.2)
