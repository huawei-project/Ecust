"""
## NOTE!!
* All images of dataset are preprocessed following the [SphereFace](https://github.com/wy1iu/sphereface) 
* and you can download the aligned images at [Align-CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1k3Cel2wSHQxHO9NkNi3rkg).
"""
import numpy as np
import scipy.misc
import os
import torch
from torch.utils.data import DataLoader


class CASIA_Face(object):
    def __init__(self, dataset_path, txt_path):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.txt_path = os.path.expanduser(txt_path)
        image_list = []
        label_list = []
        with open(self.txt_path) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(self.dataset_path, image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = scipy.misc.imread(img_path)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        flip = np.random.choice(2) * 2 - 1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    CASIA_DATA_DIR = '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96'
    CASIA_TXT_DIR = '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96.txt'
    dataset = CASIA_Face(CASIA_DATA_DIR, CASIA_TXT_DIR)
    trainloader = DataLoader(dataset, batch_size=32,
                             shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
        break
