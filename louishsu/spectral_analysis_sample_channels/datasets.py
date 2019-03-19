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
n_channels = 46
notUsedSubjects = []
waveLen = [550+10*i for i in range(46)]
get_vol = lambda i: (i-1)//10+1
get_wavelen = lambda bmp: int(bmp.split('.')[0].split('_')[-1])

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










# def splitDatasets_Multi_channels(n_channels=46, train=0.7, valid=0.25, test=0.05):
#     n_train = int(n_channels * train)
#     n_valid = int(n_channels * valid)
#     n_test  = n_channels - n_train - n_valid
#     i_train, i_valid, i_test = 0, 0, 0
#     print("n_train: {:d} n_valid: {:d} n_test: {:d} ".format(n_train, n_valid, n_test))

#     datapath  = configer.datapath
#     train_txt = "./dataset/{}/train.txt".format(configer.splitmode)
#     valid_txt = "./dataset/{}/valid.txt".format(configer.splitmode)
#     test_txt  = "./dataset/{}/test.txt".format(configer.splitmode)
#     ftrain = open(train_txt, 'w'); fvalid = open(valid_txt, 'w'); ftest  = open(test_txt, 'w')

#     dicts = getDicts()

#     filename = "{}/DATA{}/{}/Multi/{}/Multi_{}_W1_{}"
#     obtypes = ["non-obtructive", "obtructive/ob1", "obtructive/ob2"]
#     posidx  = [i for i in range(1, 8)]
#     sessidx = [i for i in range(1, 7)]

#     n_bmp_channels = [[0 for i in range(n_channels)] for i in range(3)]

#     for i in range(1, 34):
#         if i in notUsedSubjects: continue
#         for obtype in obtypes:
#             for pos in posidx:
#                 for sess in sessidx:
#                     vol = get_vol(i)
#                     file = filename.format(datapath, vol, i, obtype, pos, sess)
#                     dict = dicts['DATA%d' % vol]; 
                    
#                     if obtype == "non-obtructive":
#                         key = '/' + '/'.join(file.split('/')[-4:])
#                     else:
#                         key = '/' + '/'.join(file.split('/')[-5:])

#                     if os.path.exists(file) and (key in dict.keys()) and (dict[key][0] is not None):
#                         bmpfiles = os.listdir(file)
#                         bmpfiles = [os.path.join(file, bmp) + '\n' for bmp in bmpfiles]
                        
#                         bmpfiles_train_valid = random.sample(bmpfiles, n_train + n_valid)
#                         bmpfiles_test  = [bmp for bmp in bmpfiles if bmp not in bmpfiles_train_valid]
#                         bmpfiles_train = random.sample(bmpfiles_train_valid, n_train)
#                         bmpfiles_valid = [bmp for bmp in bmpfiles_train_valid if bmp not in bmpfiles_train]
                        
#                         bmplists = [bmpfiles_train, bmpfiles_valid, bmpfiles_test]
#                         for i_bmp in range(len(bmplists)):
#                             bmplist = bmplists[i_bmp]
#                             wls = [waveLen.index(get_wavelen(bmp)) for bmp in bmplist]
#                             for wl in wls: 
#                                 n_bmp_channels[i_bmp][wl] += 1

#                         ftrain.writelines(bmpfiles_train)
#                         fvalid.writelines(bmpfiles_valid)
#                         ftest.writelines (bmpfiles_test )

#                         i_train += n_train
#                         i_valid += n_valid
#                         i_test  += n_test

#     print("number of samples: {}".format(i_train//n_train))
#     print("number of train: {:5d}, ".format(i_train), n_bmp_channels[0])
#     print("number of valid: {:5d}, ".format(i_valid), n_bmp_channels[1])
#     print("number of test:  {:5d}, ".format(i_test) , n_bmp_channels[2])
#     ftrain.close(); fvalid.close(); ftest.close()


def splitDatasets_Multi_channels(n_channels=46, train=0.7, valid=0.25, test=0.05):
    n_train = int(n_channels * train)
    n_valid = int(n_channels * valid)
    n_test  = n_channels - n_train - n_valid
    i_train, i_valid, i_test = 0, 0, 0
    print("splitmode: {}".format(configer.splitmode))
    print("n_train: {:d} n_valid: {:d} n_test: {:d} ".format(n_train, n_valid, n_test))

    datapath  = configer.datapath
    splitdir = "./dataset/{}".format(configer.splitmode)
    if not os.path.exists(splitdir): os.mkdir(splitdir)
    train_txt = "./dataset/{}/train.txt".format(configer.splitmode)
    valid_txt = "./dataset/{}/valid.txt".format(configer.splitmode)
    test_txt  = "./dataset/{}/test.txt".format(configer.splitmode)
    ftrain = open(train_txt, 'w'); fvalid = open(valid_txt, 'w'); ftest  = open(test_txt, 'w')

    dicts = getDicts()

    filename = "{}/DATA{}/{}/Multi/{}/Multi_{}_W1_{}"
    obtypes = ["non-obtructive", "obtructive/ob1", "obtructive/ob2"]
    posidx  = [i for i in range(1, 8)]
    sessidx = [i for i in range(1, 7)]

    trainfiles = []; validfiles = []; testfiles = []

    # 按一定比例将所有样本划分
    for i in range(1, 34):
        if i in notUsedSubjects: continue
        for obtype in obtypes:
            for pos in posidx:
                for sess in sessidx:
                    vol = get_vol(i)
                    file = filename.format(datapath, vol, i, obtype, pos, sess)
                    dict = dicts['DATA%d' % vol]; 
                    
                    if obtype == "non-obtructive":
                        key = '/' + '/'.join(file.split('/')[-4:])
                    else:
                        key = '/' + '/'.join(file.split('/')[-5:])
                        
                    if os.path.exists(file) and (key in dict.keys()) and (dict[key][0] is not None):
                        bmpfiles = os.listdir(file)
                        bmpfiles = [os.path.join(file, bmp) + '\n' for bmp in bmpfiles]
                        
                        bmpfiles_train_valid = random.sample(bmpfiles, n_train + n_valid)
                        bmpfiles_test  = [bmp for bmp in bmpfiles if bmp not in bmpfiles_train_valid]
                        bmpfiles_train = random.sample(bmpfiles_train_valid, n_train)
                        bmpfiles_valid = [bmp for bmp in bmpfiles_train_valid if bmp not in bmpfiles_train]
                        
                        trainfiles += bmpfiles_train
                        validfiles += bmpfiles_valid
                        testfiles  += bmpfiles_test
    # 划分1: 
    if configer.splitmode == 'split_1' or\
        configer.splitmode == 'split_2':
        pass

    # 划分2： 将训练集中包含的干扰光和墨镜去除
    if configer.splitmode == 'split_3':
        trainfiles_ob_sunglass = []
        for item in trainfiles:
            a = item.split('/')[-2].split('_')[-1]
            b = item.split('/')[-3]
            if a!='1' or b!='non-obtructive':
                trainfiles_ob_sunglass += [item]
        trainfiles = [item for item in trainfiles if item not in trainfiles_ob_sunglass]
        
        trainfiles_ob_sunglass_valid = random.sample(trainfiles_ob_sunglass, int(len(trainfiles_ob_sunglass)*(valid / (valid + test))))
        trainfiles_ob_sunglass_test  = [item for item in trainfiles_ob_sunglass if item not in trainfiles_ob_sunglass_valid]

    # 划分3: 将训练集中包含的干扰光和墨镜数据，按比例放入验证集与测试集
    if configer.splitmode == 'split_4' or\
        configer.splitmode == 'split_5' or\
            configer.splitmode == 'split_6' or\
                configer.splitmode == 'split_7' or\
                    configer.splitmode == 'split_8' or\
                        configer.splitmode == 'split_9' or\
                            configer.splitmode == 'split_10' or\
                                configer.splitmode == 'split_11' or\
                                    configer.splitmode == 'split_12' or\
                                        configer.splitmode == 'split_13':
        trainfiles_ob_sunglass = []
        for item in trainfiles:
            a = item.split('/')[-2].split('_')[-1]
            b = item.split('/')[-3]
            if a!='1' or b!='non-obtructive':
                trainfiles_ob_sunglass += [item]
        trainfiles = [item for item in trainfiles if item not in trainfiles_ob_sunglass]
        
        trainfiles_ob_sunglass_valid = random.sample(trainfiles_ob_sunglass, int(len(trainfiles_ob_sunglass)*(valid / (valid + test))))
        trainfiles_ob_sunglass_test  = [item for item in trainfiles_ob_sunglass if item not in trainfiles_ob_sunglass_valid]

        validfiles += trainfiles_ob_sunglass_valid
        testfiles  += trainfiles_ob_sunglass_test
    
    # 划分4: 
    #   - 将训练集中包含的干扰光和墨镜数据，按比例放入验证集与测试集
    #   - 随机划分后不确定性太大，故正面照全部放入测试集，保证干扰光(2种)、有无近视眼镜(2中)均有一份完整的正面数据
    #       - `non-obtructive/Multi_4_W1_1`
    #       - `non-obtructive/Multi_4_W1_5`
    #       - `obtructive/ob1/Multi_4_W1_1`
    #       - `obtructive/ob2/Multi_4_W1_1`
    #   - 最终在该正面数据上进行分析
    if configer.splitmode == 'split_14':

        obtype_4_W1_pos = []
        for item in trainfiles:
            if item.split('/')[-2].split('_')[-3] == '4' and\
                item.split('/')[-2].split('_')[-1] in ['1', '5']:
                obtype_4_W1_pos += [item]
        for item in validfiles:
            if item.split('/')[-2].split('_')[-3] == '4' and\
                item.split('/')[-2].split('_')[-1] in ['1', '5']:
                obtype_4_W1_pos += [item]


        trainfiles_ob_sunglass = []
        for item in trainfiles:
            a = item.split('/')[-2].split('_')[-1]
            b = item.split('/')[-3]
            if a!='1' or b!='non-obtructive':
                trainfiles_ob_sunglass += [item]
        trainfiles = [item for item in trainfiles if item not in trainfiles_ob_sunglass]
        
        trainfiles_ob_sunglass_valid = random.sample(trainfiles_ob_sunglass, int(len(trainfiles_ob_sunglass)*(valid / (valid + test))))
        trainfiles_ob_sunglass_test  = [item for item in trainfiles_ob_sunglass if item not in trainfiles_ob_sunglass_valid]

        validfiles += trainfiles_ob_sunglass_valid
        testfiles  += trainfiles_ob_sunglass_test

    # statistic
    n_bmp_channels = [[0 for i in range(n_channels)] for i in range(3)]
    for item in trainfiles:
        idxch = waveLen.index(get_wavelen(item))
        n_bmp_channels[0][idxch] += 1
        ftrain.write(item)
    for item in validfiles:
        idxch = waveLen.index(get_wavelen(item))
        n_bmp_channels[1][idxch] += 1
        fvalid.write(item)
    for item in testfiles:
        idxch = waveLen.index(get_wavelen(item))
        n_bmp_channels[2][idxch] += 1
        ftest.write(item)

    n_train = len(trainfiles)
    n_valid = len(validfiles)
    n_test  = len(testfiles)
    n_items = n_train + n_valid + n_test
    print("number of samples: {}".format(n_items))
    print("number of train: {:5d}, ".format(n_train), n_bmp_channels[0])
    print("number of valid: {:5d}, ".format(n_valid), n_bmp_channels[1])
    print("number of test:  {:5d}, ".format(n_test ), n_bmp_channels[2])
    print("train: valid: test:  {:.3f}: {:.3f}: {:.3f}".\
                    format(n_train / n_items, n_valid / n_items, n_test  / n_items))

    ftrain.close(); fvalid.close(); ftest.close()

def gen_test_txt_pos4():
    dicts = getDicts()
    f = open("./dataset/{}/test_pos4.txt".format(configer.splitmode), 'w')
    obtypes_glasses = [
        'non-obtructive/Multi_4_W1_1',
        'non-obtructive/Multi_4_W1_5',
        'obtructive/ob1/Multi_4_W1_1',
        'obtructive/ob2/Multi_4_W1_1',
    ]
    DIRNAME = 'DATA{}/{}/Multi/{}'

    for i in range(1, 34):
        for ob_glass in obtypes_glasses:
            dirname = DIRNAME.format(get_vol(i), i, ob_glass)
            a = configer.datapath + '/' + dirname
            if not os.path.exists(configer.datapath + '/' + dirname): continue
            vol = dirname.split('/')[0]
            dir = '/' + '/'.join(dirname.split('/')[1:])
            if dicts[vol][dir][0] is None: continue
            bmplists = os.listdir(configer.datapath + '/' + dirname)
            bmplists.sort(key=get_wavelen)
            bmplists = [configer.datapath + '/' + dirname + '/' + bmp + '\n' for bmp in bmplists]
            f.writelines(bmplists)
    f.close()

class HyperECUST(Dataset):
    labels = [i for i in range(1, 34) if (i not in notUsedSubjects)]

    def __init__(self, splitmode, facesize=None, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        with open('./dataset/{}/{}.txt'.format(splitmode, mode), 'r') as f:
            self.filenames = f.readlines()
        self.facesize = tuple(facesize)
        self.dicts = getDicts()

    def __getitem__(self, index):
        filename = self.filenames[index].strip()
        label = get_label_from_path(filename)

        # get bbox
        vol = "DATA%d" % get_vol(label)
        imgname = filename[filename.find("DATA")+5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        [x1, y1, x2, y2] = bbox

        # load image array
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])

        image = image[:, :, np.newaxis]
        image = ToTensor()(image)
        label = self.labels.index(label)
        return image, label
    
    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    # splitDatasets_Multi_channels(46, 0.6, 0.2, 0.2)
    gen_test_txt_pos4()

    # from torch.utils.data import DataLoader
    # trainloader = DataLoader(HyperECUST(configer.splitmode, (64, 64), mode='test'))
    # for i_batch, (X, y) in enumerate(trainloader):
    #     X = (X[0, 0, :, :].numpy()*255).astype('uint8')
    #     cv2.imshow("", X)
    #     cv2.waitKey(10)
