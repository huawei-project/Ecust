import os
import sys
import PIL
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from config import configer
from utiles import convert_to_npy_path, get_label_from_path

"""
\-- root
    \-- DATAn
        \-- subject index
            \-- Multi
                \-- non-obtructive
                    \-- Multi_{1~7}_W1_1        # 无眼镜
                    \-- Multi_{1~7}_W1_5        # 戴眼镜
                    \-- Multi_4_W1_6            # 墨镜1
                \-- obtructive
                    \-- ob1f
                        \-- Multi_{1~7}_W1_1    # 无眼睛
                    \-- ob2
                        \-- Multi_{1~7}_W1_1    # 无眼镜
            \-- RGB
                \-- non-obtructive
                    |-- RGB_{1~7}_W1_1
                    |-- RGB_{1~7}_W1_5
                    |-- RGB_4_W1_6
                \-- obtructive
                    \-- ob1
                        |-- RGB_{1~7}_W1_1
                    \-- ob2
                        |-- RGB_{1~7}_W1_1
"""

notUsed = [16, 17, 24, 28, 30, 32]

def load_rgb(imgpath, dsize=None):
    """
    Params:
        imgpath: {str}
        dsize:   {tuple(W, H)}
    Returns:
        imgs:   {ndarray(H, W, 3)}
    """
    assert os.path.exists(imgpath), "rgb file does not exist!"
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    if dsize is not None:
        img = cv2.resize(img, dsize)
    return img

def load_multi(imgdir, dsize=None):
    """
    Params:
        imgdir: {str}
        dsize:   {tuple(W, H)}
    Returns:
        imgs:   {ndarray(H, W, C)}
    """
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

def getBbox(imgpath):
    """
    Params:
        imgpath:    {str}
    Returns:
        score:      {float}
        bbox:       {ndarray(4)}
        landmark:   {ndarray(10)}
    Notes:
        - 可返回字典
    """
    imgpath = imgpath.split('.')[0]
    idx = imgpath.find("DATA") + 5
    txtfile = os.path.join(imgpath[: idx], "detect.txt")
    imgname = imgpath[idx: ]
    with open(txtfile, 'r') as f:
        dict_save = eval(f.read())
    
    score, bbox, landmark = dict_save[imgname]
    if (score is None):
        return score, bbox, landmark

    score = np.array(score)
    bbox = np.array(bbox, dtype='int')
    landmark = np.array(landmark, dtype='int')
    return score, bbox, landmark

def getDicts():
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 5)]:
        txtfile = os.path.join(configer.datapath, vol, "detect.txt")
        with open(txtfile, 'r') as f:
            dicts[vol] = eval(f.read())
    return dicts
    
def show_result(image, score, bbox, landmarks, winname="", waitkey=0):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        score:      {ndarray(n_faces)}
        bbox:       {ndarray(n_faces, 4)}
        landmarks:  {ndarray(n_faces, 10)}
        winname:    {str}
    """
    n_faces = bbox.shape[0]
    for i in range(n_faces):
        corpbbox = [int(bbox[i, 0]), int(bbox[i, 1]), int(bbox[i, 2]), int(bbox[i, 3])]
        cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                        (corpbbox[2], corpbbox[3]), 
                        (255, 0, 0), 1)
        cv2.putText(image, '{:.3f}'.format(score[i]), 
                        (corpbbox[0], corpbbox[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        for j in range(int(len(landmarks[i])/2)):
            cv2.circle(image, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(winname)

def splitDatasets():
    datapath  = configer.datapath
    train_txt = "./dataset/{}/train.txt".format(configer.splitmode)
    valid_txt = "./dataset/{}/valid.txt".format(configer.splitmode)
    test_txt  = "./dataset/{}/test.txt".format(configer.splitmode)
    ftrain = open(train_txt, 'w'); fvalid = open(valid_txt, 'w'); ftest  = open(test_txt, 'w')
    tmp = ["Multi/non-obtructive/Multi_{}_W1_1", "Multi/non-obtructive/Multi_{}_W1_5"]
    for i in range(1, 34):
        if i in notUsed: continue
        files = []
        for t in tmp:
            for p in range(1, 8):
                filename = str(i) + '/' + t.format(p)
                filepath = "{}/DATA{}/{}".format(datapath, (i-1)//10+1, filename)
                print(filepath)
                score, _, _ = getBbox(filepath)
                if (score is not None):
                    files += [filepath]
        idx = random.sample(range(1, len(files)), 1)
        
        # # test
        # ftest.write(files[idx[0]] + '\n')
        # # valid
        # for k in idx[0]:
        #     fvalid.write(files[k] + '\n')
        # # train
        # for k in range(1, 8):
        #     if (k not in idx):
        #         ftrain.write(files[k] + '\n')

        # valid
        fvalid.write(files[idx[0]] + '\n')
        # train
        for k in range(1, 8):
            ftrain.write(files[k] + '\n')

    ftrain.close(); fvalid.close(); ftest.close()

def read_all_to_test_txt():
    datapath = configer.datapath
    test_txt = "./dataset/{}/test.txt".format(configer.splitmode)
    ftest  = open(test_txt, 'w')

    multidirs = [
            "DATA{}/{}/Multi/non-obtructive/Multi_{}_W1_1", 
            "DATA{}/{}/Multi/non-obtructive/Multi_{}_W1_5", 
            "DATA{}/{}/Multi/non-obtructive/Multi_{}_W1_6", 
            "DATA{}/{}/Multi/non-obtructive/Multi_{}_W1_7", 
            "DATA{}/{}/Multi/obtructive/ob1/Multi_{}_W1_1", 
            "DATA{}/{}/Multi/obtructive/ob2/Multi_{}_W1_1", 
        ]

    get_vol = lambda i: (i - 1) // 10 + 1    
    index = [i for i in range(1, 34) if i not in notUsed]
    pos   = [i+1 for i in range(7)]

    for multidir in multidirs:
        for i in index:
            for p in pos:
                filedir = multidir.format(get_vol(i), i, p)
                filedir_abs = os.path.join(datapath, filedir)
                filedir_abs_npy = convert_to_npy_path(filedir_abs)
                if not (os.path.exists(filedir_abs) and (os.path.exists(filedir_abs_npy))): 
                    print("{} doesnot exist! ".format(filedir))
                    continue
                ftest.write(filedir_abs + '\n')
                
    ftest.close()


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

def square(bbox):
    (x1, y1, x2, y2) = bbox
    h = y2 - y1; w = x2 - x1
    max_side = np.maximum(h, w)
    x1 = x1 + w * 0.5 - max_side * 0.5
    y1 = y1 + h * 0.5 - max_side * 0.5
    x2 = x1 + max_side - 1
    y2 = y1 + max_side - 1
    return (int(x1), int(y1), int(x2), int(y2))

class HyperECUST(Dataset):
    labels = [i for i in range(1, 34) if (i not in notUsed)]

    def __init__(self, splitmode, facesize=None, mode='train', loadnpy=True):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        with open('./dataset/{}/{}.txt'.format(splitmode, mode), 'r') as f:
            self.filenames = f.readlines()
        self.facesize = tuple(facesize)
        self.loadnpy = loadnpy
        self.dicts = getDicts()
    def __getitem__(self, index):
        filename = self.filenames[index].strip()
        label = get_label_from_path(filename)

        if not self.loadnpy:
            # get bbox
            vol = "DATA%d" % ((label-1)//10+1)
            imgname = filename[filename.find("DATA")+5:]
            bbox = self.dicts[vol][imgname][1]
            [x1, y1, x2, y2] = bbox
            # load image array
            image = load_multi(filename)[y1: y2, x1: x2]

        else:
            # load image array
            filename = convert_to_npy_path(filename)
            image = np.load(filename)
        
        if self.facesize is not None:
            image = resizeMulti(image, self.facesize)

        image = ToTensor()(image)
        label = self.labels.index(label)
        return image, label
    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":

    read_all_to_test_txt()

    # from torch.utils.data import DataLoader
    # trainloader = DataLoader(HyperECUST(configer.splitmode, (64, 64), mode='test',  loadnpy=True))
    # for i_batch, (X, y) in enumerate(trainloader):
    #     X = X[0, 0, :, :].numpy().astype('uint8')
    #     cv2.imshow("", X)
    #     cv2.waitKey(10)

    # splitDatasets()   