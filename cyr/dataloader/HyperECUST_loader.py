"""
# Author: Yuru Chen
# Time: 2019 03 20
"""
import os
import sys
import glob
import cv2
import scipy.misc
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from vis_utils import show_result
from noise import addsalt_pepper


def get_id(x): return int(x.split('/')[1])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_vol(i): return (i - 1) // 10 + 1


def get_path_from_band_range(data_list, band_list):
    rel = list(filter(lambda x: int(
        x.split('_')[-1].split('.')[0]) in band_list, data_list))
    assert len(rel) != 0, 'could not get path from band_list!'
    return rel


def getDicts(dataset_path):
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 10)]:
        txtfile = os.path.join(dataset_path, vol, "detect.txt")
        if os.path.isfile(txtfile):
            with open(txtfile, 'r') as f:
                dicts[vol] = eval(f.read())
    return dicts


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


def filename_parse(filename):
    parse_dict = dict()
    parse_dict['image_type'] = 'Multi' if 'Multi' in filename else 'RGB'
    if '/ob1' in filename:
        parse_dict['ob'] = 'ob1'
    elif '/ob2' in filename:
        parse_dict['ob'] = 'ob2'
    else:
        parse_dict['ob'] = 'nonob'
    parse_dict['pose'] = filename.split('_')[1]
    parse_dict['glasses'] = filename.split('_')[3].split('/')[0]
    parse_dict['band'] = get_band(filename)
    parse_dict['id'] = get_id(filename)
    return parse_dict


def hisEqulColor(img):
    # according to [https://www.jianshu.com/p/9a9000d226b6]
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


class HyperECUST_FV(Dataset):
    def __init__(self, dataset_path, pairs_path,
                 facesize=None, cropped_by_bbox=False,
                 equalization=False, snr=1.0, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.dataset_path = os.path.expanduser(dataset_path)
        self.pairs_path = os.path.expanduser(pairs_path)
        self.facesize = facesize
        self.cropped_by_bbox = cropped_by_bbox
        self.equalization = equalization
        self.snr = snr
        self.mode = mode
        self.dicts = getDicts(self.dataset_path)
        self.parseList(self.pairs_path)

    def parseList(self, root):
        with open(root) as f:
            pairs = f.read().splitlines()
        n_sample = len(pairs)
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []
        for i, p in enumerate(pairs):
            p = p.split('\t')
            nameL, nameR = p[0], p[2]
            fold = i // (np.ceil(n_sample / 10))
            flag = 1 if p[1] == p[3] else -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)
        # print(nameLs)
        return

    def get_bbox(self, filename):
        # get bbox
        vol = "DATA%d" % get_vol(get_id(filename))
        imgname = filename[filename.find("DATA") + 5:]
        if 'RGB' in filename:
            dirname = imgname.split('.')[0]
        else:
            dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filename):
        filename = filename.replace('bmp', 'JPG')
        # load image array
        if 'RGB' in filename:
            image = cv2.imread(os.path.join(
                self.dataset_path, filename), cv2.IMREAD_COLOR)
            image = addsalt_pepper(image, self.snr)
        else:
            image = cv2.imread(os.path.join(
                self.dataset_path, filename), cv2.IMREAD_GRAYSCALE)
            image = addsalt_pepper(image, self.snr)
        if image is None:
            print(filename)
            raise ValueError
        if self.cropped_by_bbox:
            x1, y1, x2, y2 = self.get_bbox(filename)[0]
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(x2, w)
            y1, y2 = max(0, y1), min(y2, h)
            image = image[y1: y2, x1: x2]
        if self.equalization:
            if 'RGB' in filename:
                image = hisEqulColor(image)
            else:
                image = cv2.equalizeHist(image)
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])
        if 'RGB' not in filename:
            image = np.stack([image] * 3, 2)
        return image

    def __getitem__(self, index):
        filenameL = self.nameLs[index]
        filenameR = self.nameRs[index]
        fold = self.folds[index]
        label = self.flags[index]
        imgl = self.get_image(filenameL)
        imgr = self.get_image(filenameR)
        imglist = [imgl, imgl[:, ::-1, :].copy(),
                   imgr, imgr[:, ::-1, :].copy()]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [(torch.from_numpy(x).float() - 127.5) / 128.0 for x in imglist]
        #imgs = [(ToTensor()(x) - 127.5) / 128.0 for x in imglist]
        #imgs = [ToTensor()(x) for x in imglist]
        return imgs

    def __len__(self):
        return len(self.nameLs)


class HyperECUST_FI(Dataset):
    def __init__(self, dataset_path, dataset_txt,
                 facesize=None, cropped_by_bbox=False, equalization=False, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset_txt = os.path.expanduser(dataset_txt)
        self.facesize = facesize
        self.cropped_by_bbox = cropped_by_bbox
        self.equalization = equalization
        self.mode = mode
        self.dicts = getDicts(self.dataset_path)
        self.image_list = []
        self.label_list = []
        with open(self.dataset_txt, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            path, label = line.split()
            self.image_list.append(path)
            self.label_list.append(int(label))
        self.class_nums = len(np.unique(self.label_list))

    def get_bbox(self, filename):
        # get bbox
        vol = "DATA%d" % get_vol(get_id(filename))
        imgname = filename[filename.find("DATA") + 5:]
        if 'RGB' in filename:
            dirname = imgname.split('.')[0]
        else:
            dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filename):
        filename = filename.replace('bmp', 'JPG')
        # load image array
        if 'RGB' in filename:
            image = cv2.imread(os.path.join(
                self.dataset_path, filename), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(os.path.join(
                self.dataset_path, filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(os.path.join(self.dataset_path, filename))
            raise ValueError
        if self.cropped_by_bbox:
            x1, y1, x2, y2 = self.get_bbox(filename)[0]
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(x2, w)
            y1, y2 = max(0, y1), min(y2, h)
            image = image[y1: y2, x1: x2]
        if self.equalization:
            if 'RGB' in filename:
                image = hisEqulColor(image)
            else:
                image = cv2.equalizeHist(image)
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])
        if 'RGB' not in filename:
            image = np.stack([image] * 3, 2)
        return image

    def __getitem__(self, index):
        filename = self.image_list[index]
        label = self.label_list[index]
        image = self.get_image(filename)
        flip = np.random.choice(2) * 2 - 1
        image = image[:, ::flip, :]
        image = (image - 127.5) / 128.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        return image, label

    def __len__(self):
        return len(self.image_list)


class HyperECUST_FI_MI(Dataset):
    def __init__(self, dataset_path, dataset_txt, bands,
                 facesize=None, cropped_by_bbox=False, equalization=True, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset_txt = os.path.expanduser(dataset_txt)
        self.bands = bands
        self.facesize = facesize
        self.cropped_by_bbox = cropped_by_bbox
        self.equalization = equalization
        self.mode = mode
        self.dicts = getDicts(self.dataset_path)
        self.image_list = []
        self.label_list = []
        with open(self.dataset_txt, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            path, label = line.split()
            self.image_list.append(path)
            self.label_list.append(int(label))
        self.class_nums = len(np.unique(self.label_list))
        self.in_channels = len(bands)

    def get_bbox(self, filename):
        # get bbox
        vol = "DATA%d" % get_vol(get_id(filename))
        imgname = filename[filename.find("DATA") + 5:]
        if 'RGB' in filename:
            dirname = imgname.split('.')[0]
        else:
            dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filedir):
        filenames = glob.glob(os.path.join(self.dataset_path, filedir, '*'))
        filenames = [x[x.find('DATA'):] for x in filenames]
        filenames_ = get_path_from_band_range(filenames, self.bands)
        filenames_.sort()
        # load image array
        images = None
        for i in range(len(filenames_)):
            image = cv2.imread(os.path.join(self.dataset_path, filenames_[i]),
                               cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(filenames_[i])
                raise ValueError
            if self.cropped_by_bbox:
                x1, y1, x2, y2 = self.get_bbox(filenames_[i])[0]
                h, w = image.shape[:2]
                x1, x2 = max(0, x1), min(x2, w)
                y1, y2 = max(0, y1), min(y2, h)
                image = image[y1: y2, x1: x2]
            if self.equalization:
                image = cv2.equalizeHist(image)
            if self.facesize is not None:
                image = cv2.resize(image, self.facesize[::-1])
            image = np.expand_dims(image, 2)
            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), 2)
        return images

    def __getitem__(self, index):
        filedir = self.image_list[index]
        label = self.label_list[index]
        image = self.get_image(filedir)
        flip = np.random.choice(2) * 2 - 1
        image = image[:, ::flip, :]
        image = (image - 127.5) / 128.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        return image, label

    def __len__(self):
        return len(self.image_list)


class HyperECUST_FV_MI(Dataset):
    def __init__(self, dataset_path, pairs_txt, bands,
                 facesize=None, cropped_by_bbox=False, equalization=False, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.dataset_path = os.path.expanduser(dataset_path)
        self.pairs_txt = os.path.expanduser(pairs_txt)
        self.bands = bands
        self.facesize = facesize
        self.cropped_by_bbox = cropped_by_bbox
        self.equalization = equalization
        self.mode = mode
        self.dicts = getDicts(self.dataset_path)
        self.in_channels = len(bands)
        self.parseList(self.pairs_txt)

    def parseList(self, root):
        with open(root) as f:
            pairs = f.read().splitlines()
        n_sample = len(pairs)
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []
        for i, p in enumerate(pairs):
            p = p.split('\t')
            nameL, nameR = p[0], p[2]
            fold = i // (np.ceil(n_sample / 10))
            flag = 1 if p[1] == p[3] else -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)
        # print(nameLs)
        return

    def get_bbox(self, filename):
        # get bbox
        vol = "DATA%d" % get_vol(get_id(filename))
        imgname = filename[filename.find("DATA") + 5:]
        if 'RGB' in filename:
            dirname = imgname.split('.')[0]
        else:
            dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filedir):
        filenames = glob.glob(os.path.join(self.dataset_path, filedir, '*'))
        filenames = [x[x.find('DATA'):] for x in filenames]
        filenames_ = get_path_from_band_range(filenames, self.bands)
        filenames_.sort()
        # load image array
        images = None
        for i in range(len(filenames_)):
            image = cv2.imread(os.path.join(self.dataset_path, filenames_[i]),
                               cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(filenames_[i])
                raise ValueError
            if self.cropped_by_bbox:
                x1, y1, x2, y2 = self.get_bbox(filenames_[i])[0]
                h, w = image.shape[:2]
                x1, x2 = max(0, x1), min(x2, w)
                y1, y2 = max(0, y1), min(y2, h)
                image = image[y1: y2, x1: x2]
            if self.equalization:
                image = cv2.equalizeHist(image)
            if self.facesize is not None:
                image = cv2.resize(image, self.facesize[::-1])
            image = np.expand_dims(image, 2)
            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), 2)
        return images

    def __getitem__(self, index):
        filenameL = self.nameLs[index]
        filenameR = self.nameRs[index]
        fold = self.folds[index]
        label = self.flags[index]
        imgl = self.get_image(filenameL)
        imgr = self.get_image(filenameR)
        imglist = [imgl, imgl[:, ::-1, :].copy(),
                   imgr, imgr[:, ::-1, :].copy()]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [(torch.from_numpy(x).float() - 127.5) / 128.0 for x in imglist]
        #imgs = [(ToTensor()(x) - 127.5) / 128.0 for x in imglist]
        #imgs = [ToTensor()(x) for x in imglist]
        return imgs

    def __len__(self):
        return len(self.nameLs)


def example_face_verification():
    # HyperECUST Face Verification Dataset
    pairset_path = '~/myDataset/ECUST_112x96/'  # Your HyperECUST dataset path
    pairset_txt = '~/yrc/myFile/huaweiproj/code/datasets/face_verfication_split/split_9/pairs_fold_0.txt'
    pairset = HyperECUST_FV(
        pairset_path, pairset_txt, facesize=None, cropped_by_bbox=False, equalization=True)
    folds = pairset.folds
    flags = pairset.flags
    pairloader = DataLoader(pairset, batch_size=1, shuffle=False)

    index = 800
    imglist = pairset[index]
    print(np.array(pairset.folds).shape)
    print('pairset length {}'.format(len(pairset)))
    print('image shape {}, {}'.format(imglist[0].shape, imglist[1].shape))
    print('left image path {}'.format(pairset.nameLs[index]))
    print('right image path {}'.format(pairset.nameRs[index]))
    print('label {}, fold {}'.format(flags[index], folds[index]))
    image_1 = imglist[0].numpy().transpose(1, 2, 0) * 128 + 127.5
    image_1 = image_1.astype(np.uint8)
    image_2 = imglist[2].numpy().transpose(1, 2, 0) * 128 + 127.5
    image_2 = image_2.astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.show()
    #show_result(image_1, winname='people1', waitkey=0)
    #show_result(image_2, winname='people2', waitkey=0)


def example_face_identification():
    # HyperECUST Face Identification Dataste
    trainset_path = '~/myDataset/ECUST_112x96/'  # Your HyperECUST dataset path
    trainset_txt = '~/yrc/myFile/huaweiproj/code/datasets/face_verfication_split/split_2/train_exp_0.txt'
    trainset = HyperECUST_FI(trainset_path, trainset_txt,
                             facesize=None, cropped_by_bbox=False, equalization=True)
    trainloader = DataLoader(trainset, batch_size=10, shuffle=False)

    index = 800
    image, label = trainset[index]
    print('trainset length {}'.format(len(trainset)))
    print('image shape {}'.format(image.shape))
    print('label {}'.format(label))
    image = image.numpy().transpose(1, 2, 0) * 128 + 127.5
    image = image.astype(np.uint8)
    show_result(image)


def example_face_identification_multi_input():
    # HyperECUST Face Identification Dataste
    trainset_path = '~/myDataset/ECUST_112x96/'  # Your HyperECUST dataset path
    trainset_txt = '~/yrc/myFile/huaweiproj/code/datasets/face_identification_split/split_20/train_exp_0.txt'
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    trainset = HyperECUST_FI_MI(trainset_path, trainset_txt, bands,
                                facesize=(112, 96), cropped_by_bbox=False, equalization=True)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)

    # for images, labels in trainloader:
    #     print(images.shape, labels)
    index = 200
    image, label = trainset[index]
    path = trainset.image_list[index]
    print('trainset length {}'.format(len(trainset)))
    print('trainloader length {}'.format(len(trainloader)))
    print('image shape {}'.format(image.shape))
    print('image path {}'.format(path))
    print('label {}'.format(label))
    band = 15
    gray = image[band]
    gray = gray.numpy() * 128 + 127.5
    gray = gray.astype(np.uint8)
    show_result(gray)


def example_face_verification_multi_input():
    # HyperECUST Face Identification Dataste
    trainset_path = '~/myDataset/ECUST_112x96/'  # Your HyperECUST dataset path
    trainset_txt = '~/yrc/myFile/huaweiproj/code/datasets/face_verification_split/split_14/train_fold_0.txt'
    validset_path = '~/myDataset/ECUST_112x96/'  # Your HyperECUST dataset path
    validset_txt = '~/yrc/myFile/huaweiproj/code/datasets/face_verification_split/split_14/pairs_fold_0.txt'
    bands = np.arange(550, 991, 40)
    num_band = len(bands)
    print('number of bands {}, bands: {}'.format(num_band, bands))
    trainset = HyperECUST_FI_MI(trainset_path, trainset_txt, bands)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)

    validset = HyperECUST_FV_MI(validset_path, validset_txt, bands)
    validloader = DataLoader(validset, batch_size=32, shuffle=False)
    # for images, labels in trainloader:
    #     print(images.shape, labels)
    index = 200
    image, label = trainset[index]
    path = trainset.image_list[index]
    print('trainset length: {}'.format(len(trainset)))
    print('trainloader length: {}'.format(len(trainloader)))
    print('image shape: {}'.format(image.shape))
    print('image path: {}'.format(path))
    print('label: {}'.format(label))
    idx_band = 4
    gray = image[idx_band]
    gray = gray.numpy() * 128 + 127.5
    gray = gray.astype(np.uint8)
    show_result(gray)
    print('')
    index = 800
    images = validset[index]
    label = validset.flags[index]
    pathL = validset.nameLs[index]
    pathR = validset.nameRs[index]
    print('validset length: {}'.format(len(validset)))
    print('validloader length: {}'.format(len(validloader)))
    print('image shape: {}'.format(images[0].shape))
    print('left image path: {}'.format(pathL))
    print('rigth image path: {}'.format(pathR))
    print('label: {}'.format(label))
    idx_band = 4
    gray = images[0][idx_band]
    gray = gray.numpy() * 128 + 127.5
    gray = gray.astype(np.uint8)
    show_result(gray)
    gray = images[2][idx_band]
    gray = gray.numpy() * 128 + 127.5
    gray = gray.astype(np.uint8)
    show_result(gray)


if __name__ == '__main__':
    # example_face_verification()
    # example_face_identification()
    # example_face_identification_multi_input()

    example_face_verification_multi_input()
