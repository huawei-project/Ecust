"""
## NOTE!!
* All images of dataset are preprocessed following the [SphereFace](https://github.com/wy1iu/sphereface) 
* and you can download the aligned images at [Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg).
"""
import numpy as np
import scipy.misc
import cv2
import torch
import os
from torchvision.transforms import ToTensor


class LFW(object):
    def __init__(self, dataset_path, pairs_path, facesize=None):

        self.dataset_path = os.path.expanduser(dataset_path)
        self.pairs_path = os.path.expanduser(pairs_path)
        self.facesize = facesize
        self.parseList(self.pairs_path)

    def parseList(self, root):
        with open(root) as f:
            pairs = f.read().splitlines()[1:]
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []
        for i, p in enumerate(pairs):
            p = p.split('\t')
            if len(p) == 3:
                nameL = os.path.join(
                    self.dataset_path, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
                nameR = os.path.join(
                    self.dataset_path, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
                fold = i // 600
                flag = 1
            elif len(p) == 4:
                nameL = os.path.join(
                    self.dataset_path, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
                nameR = os.path.join(
                    self.dataset_path, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
                fold = i // 600
                flag = -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)
        # print(nameLs)
        return

    def __getitem__(self, index):
        #imgl = scipy.misc.imread(self.nameLs[index])
        imgl = cv2.imread(self.nameLs[index], cv2.IMREAD_COLOR)
        if imgl is None:
            print(self.nameLs[index])
            raise ValueError
        if self.facesize is not None:
            imgl = cv2.resize(imgl, self.facesize[::-1])
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        #imgr = scipy.misc.imread(self.nameRs[index])
        imgr = cv2.imread(self.nameRs[index], cv2.IMREAD_COLOR)
        if imgr is None:
            print(self.nameRs[index])
            raise ValueError
        if self.facesize is not None:
            imgr = cv2.resize(imgr, self.facesize[::-1])
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs

    def __len__(self):
        return len(self.nameLs)


if __name__ == '__main__':
    dataset_path = '~/yrc/myFile/sphereface/test/data/lfw-112X96'
    pairs_path = '~/yrc/myFile/sphereface/test/data/pairs.txt'
    lfw = LFW(dataset_path, pairs_path)
    imgs = lfw[0]
    print(len(lfw))
    print(len(imgs))
    print(imgs[0].shape)
