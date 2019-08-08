import os
import cv2
import numpy as np

from torch.utils.data       import Dataset
from torchvision.transforms import ToTensor

import utils 

class RecognizeDataset(Dataset):
    """ 识别数据集

    Attributes:
        labels:     {list[int]} label of subjects
        samplelist: {list[list[tensor(C, H, W), int]]}
        
    Example:
        ```
        from torch.utils.data import DataLoader

        trainset = RecognizeDataset('/datasets/ECUSTDETECT', 'RGB', 
                'split_112x96_1', 'train', "RGB", load_in_memory=False)
        trainloader = DataLoader(trainsets, batch_size, shuffle=True)

        for i, (X, y) in enumerate(trainloader):
            X = X.cuda(); y = y.cuda()
        ```
    
    Update:
    """
    labels = [i+1 for i in range(63)]
    
    def __init__(self, datapath, type, splitmode, mode, 
                    usedChannels=None, condition=None, hist=True, load_in_memory=True):
        """
        Params:
            datapath    {str} '/datasets/ECUSTDETECT/'
            type:       {str} 'Multi', 'RGB'
            splitmode:  {str} 'split_112x96_1'
            mode:       {str} 'train', 'valid', 'test'
            usedChannels: 
                    - 多光谱数据: {list[int]} 形如[1, 2, 3, ...]
                    - 可见光数据: {str} 形如 "RGB", "R", "G", "B" 
            condition:  {callablc functions} 自定义筛选函数
                    e.g. utils.is_with_no_sun_glasses
            hist:       {bool} 是否直方图均衡化
        """
        txtfile = './split/{}/{}_{}.txt'.format(splitmode, mode, type)
        with open(txtfile, 'r') as f:
            self.filelist = f.readlines()
        self.filelist = list(map(lambda x: x.strip(), self.filelist))

        if condition is not None:
            self.filelist = list(filter(condition, self.filelist))

        self.datapath = datapath
        self.type = type
        self.usedChannels = usedChannels
        self.hist = hist
        self.load_in_memory = load_in_memory
        
        if load_in_memory:
            self.samplelist = list(map(lambda  x: \
                                        [
                                            self.load_image(self.datapath, x, self.type, self.usedChannels, self.hist), 
                                            self._get_label(x)
                                        ], self.filelist))
        
        print("Total: {}".format(len(self)))

    @staticmethod
    def load_image(datapath, path, type, usedChannels=None, hist=True):
        """
        Params:
            path:   {str}
                - 若为多光谱，则路径形如
                    - '1/multi/illum3/Multi_3_W1_1/2'；
                    - '1/multi/illum3/Multi_3_W1_5'；
                - 若为可见光，则路径形如
                    - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                    - '1/rgb/illum3/RGB_3_W1_5.jpg'；
            type:           {str} 'Multi', 'RGB'
            usedChannels:   {list[int] / str} 1 ~ 25
            hist:           {bool} 是否直方图均衡化

        Returns:
            image:  {tensor(C, H, W)}
        """
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

        if type == 'Multi':
            image = []
            for ch in usedChannels:
                imgpath = "{}/{}/{}.jpg".format(datapath, path, ch)
                img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
                if hist:
                    img = clahe.apply(img)
                image += [img[:, :, np.newaxis]]
            image = np.concatenate(image, axis=2)

        elif type == 'RGB':
            imgpath = os.path.join(datapath, path)
            image = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR)   # BGR
            
            b, g, r = cv2.split(image)
            if hist:
                b = clahe.apply(b)
                g = clahe.apply(g)
                r = clahe.apply(r)
            
            if usedChannels == 'RGB':
                image = cv2.merge([r, g, b])
            else:
                if usedChannels == 'R':
                    image = r[:, :, np.newaxis]
                elif usedChannels == 'G':
                    image = g[:, :, np.newaxis]
                elif usedChannels == 'B':
                    image = b[:, :, np.newaxis]

        
        image = ToTensor()(image)
        
        # X = image[0].numpy(); cv2.imshow("", X); cv2.waitKey(0)   # for DEBUG

        return image

    def _get_label(self, path):
        """
        Params:
            path:   {str}
                - 若为多光谱，则路径形如
                    - '1/multi/illum3/Multi_3_W1_1/2'；
                    - '1/multi/illum3/Multi_3_W1_5'；
                - 若为可见光，则路径形如
                    - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                    - '1/rgb/illum3/RGB_3_W1_5.jpg'；
        """
        return self.labels.index(int(path.split('/')[0]))

    def __getitem__(self, index):
        """
        Params:
            index:  {int} index of sample
        Returns:
            image:  {tensor(C, H, W)}
            label:  {int}
        """
        image = label = None
        if self.load_in_memory:
            image, label = self.samplelist[index]

        else:
            path = self.filelist[index]
            image = self.load_image(self.datapath, path, self.type, self.usedChannels, self.hist)
            label = self._get_label(path)

        return image, label

    def __len__(self):
        """
        Returns:
            length: {int} number of samples
        """
        
        return len(self.filelist)


if __name__ == "__main__":
    # trainset = RecognizeDataset('/datasets/ECUSTDETECT', 'Multi', 
    #             'split_112x96_[0.10:0.70:0.20]_[1]', 'train', [1, 4, 6], load_in_memory=True)
    # trainset = RecognizeDataset('/datasets/ECUSTDETECT', 'RGB', 
    #             'split_112x96_[0.10:0.70:0.20]_[1]', 'train', "RGB", load_in_memory=False)
    trainset = RecognizeDataset('/datasets/ECUSTDETECT', 'RGB', 
                'split_112x96_[0.10:0.70:0.20]_[1]', 'train', "RGB", condition=utils.is_with_no_sun_glasses, load_in_memory=False)
    for i in range(10):
        X, y = trainset[i]
