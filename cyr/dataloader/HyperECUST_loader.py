"""
# Author: Yuru Chen
# Time: 2019 03 20
"""
import os
import sys
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def get_id(x): return int(x.split('/')[1])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_vol(i): return (i - 1) // 10 + 1


def getDicts(dataset_path):
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 10)]:
        txtfile = os.path.join(dataset_path, vol, "detect.txt")
        if os.path.isfile(txtfile):
            with open(txtfile, 'r') as f:
                dicts[vol] = eval(f.read())
    return dicts


def show_result(image, score=None, bbox=None, landmarks=None, winname="", waitkey=0):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        score:      {ndarray(n_faces)}
        bbox:       {ndarray(n_faces, 4)}
        landmarks:  {ndarray(n_faces, 10)}
        winname:    {str}
    """
    if bbox is not None:
        n_faces = bbox.shape[0]
        for i in range(n_faces):
            corpbbox = [int(bbox[i, 0]), int(bbox[i, 1]),
                        int(bbox[i, 2]), int(bbox[i, 3])]
            cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]),
                          (255, 0, 0), 1)
            cv2.putText(image, '{:.3f}'.format(score[i]),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            if landmarks is not None:
                for j in range(int(len(landmarks[i]) / 2)):
                    cv2.circle(image, (int(
                        landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(winname)


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
        parse_dict['ob_type'] = 'ob1'
    elif '/ob2' in filename:
        parse_dict['ob_type'] = 'ob2'
    else:
        parse_dict['ob_type'] = 'nonob'
    parse_dict['pos_type'] = filename.split('_')[1]
    parse_dict['makeup_type'] = filename.split('_')[3].split('/')[0]
    parse_dict['band'] = get_band(filename)
    parse_dict['id'] = get_id(filename)
    return parse_dict


class HyperECUST_FV(Dataset):
    def __init__(self, data_path, pairs_path, facesize=None, cropped_by_bbox=True, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.pairs_path = pairs_path
        self.data_path = data_path
        self.facesize = tuple(facesize)
        self.cropped_by_bbox = cropped_by_bbox
        self.mode = mode
        self.dicts = getDicts(data_path)
        self.parseList(pairs_path)

    def parseList(self, root):
        with open(root) as f:
            pairs = f.read().splitlines()
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []
        for i, p in enumerate(pairs):
            p = p.split('\t')
            nameL, nameR = p[0], p[2]
            fold = i // 580
            flag = 1 if p[1] == p[3] else -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)
        # print(nameLs)
        return

    def get_bbox(self, filename):
        image_attr = filename_parse(filename)
        # get bbox
        vol = "DATA%d" % get_vol(image_attr['id'])
        imgname = filename[filename.find("DATA") + 5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filename):
        # load image array
        if self.cropped_by_bbox:
            x1, y1, x2, y2 = self.get_bbox(filename)[0]
            image = cv2.imread(os.path.join(
                self.data_path, filename), cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        else:
            image = cv2.imread(os.path.join(self.data_path, filename),
                               cv2.IMREAD_GRAYSCALE)
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])
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


if __name__ == '__main__':
    DATASET_PATH = '/home/lilium/myDataset/ECUST/'  # Your HyperECUST dataset path
    pairs_path = '/home/lilium/yrc/myFile/huaweiproj/code/datasets/face_verfication_split/split_1/pairs.txt'
    pairset = HyperECUST_FV(
        DATASET_PATH, pairs_path, facesize=(112, 96), cropped_by_bbox=True)
    folds = pairset.folds
    flags = pairset.flags
    pairloader = DataLoader(pairset, batch_size=1, shuffle=False)

    index = 4396
    imglist = pairset[index]
    print('pairset length {}'.format(len(pairset)))
    print('image shape {}, {}'.format(imglist[0].shape, imglist[1].shape))
    print('label {}, fold {}'.format(flags[index], folds[index]))
    show_result(imglist[2].numpy().transpose(1, 2, 0))
