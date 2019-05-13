import os
import cv2
import numpy as np

from torch.utils.data       import Dataset
from torchvision.transforms import ToTensor

from utiles import getWavelen, getLabel

class RecognizeDataset(Dataset):
    """ 识别数据集

    Attributes:
        labels:     {list[int]} label of subjects
        samplelist: {list[list[tensor(C, H, W), int]]}
        
    Example:
        ```
        from torch.utils.data import DataLoader

        trainset = RecognizeDataset('/datasets/ECUST2019_64x64', 'Multi', 'split_64x64_1', 'train', [550+i*20 for i in range(23)])
        trainloader = DataLoader(trainsets, batch_size, shuffle=True)

        for i, (X, y) in enumerate(trainloader):
            X = X.cuda(); y = y.cuda()
    
    Update:
        2019.04.24: equalize hist

        ```
    """
    labels = [i+1 for i in range(63)]
    
    def __init__(self, datapath, type, splitmode, mode, usedChannels, hist=True):
        """
        Params:
            datapath    {str} 'xxxx/ECUST2019_xxx'
            type:       {str} 'Multi', 'RGB'
            splitmode:  {str} 
            mode:       {str} 'train', 'valid', 'test'
        """
        if type == 'Multi':
            txtfile = './split/{}/{}.txt'.format(splitmode, mode)
        elif type == 'RGB':
            txtfile = './split/{}/{}_rgb.txt'.format(splitmode, mode)
        
        with open(txtfile, 'r') as f:
            filelist = f.readlines()
        
        filelist = [os.path.join('/'.join(datapath.split('/')[:-1]), filename.strip()) for filename in filelist]
        self.samplelist = [[self._load_image(filename, type, usedChannels, hist), self.labels.index(getLabel(filename))] for filename in filelist]
    
    @classmethod
    def _load_image(self, path, type, usedChannels, hist=True):
        """
        Params:
            path:   {str} 
            type:   {str} 'Multi', 'RGB'
            usedChannels:   {list[int] / str}
        Returns:
            image:  {tensor(C, H, W)}
        """
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

        if type == 'Multi':
            dict_wave_path = {getWavelen(img): os.path.join(path, img) for img in os.listdir(path)}
            image = []
            for ch in usedChannels:
                img = cv2.imread(dict_wave_path[ch], cv2.IMREAD_GRAYSCALE)
                if hist:
                    # img = cv2.equalizeHist(img)
                    img = clahe.apply(img)
                image += [img[:, :, np.newaxis]]
            image = np.concatenate(image, axis=2)


        elif type == 'RGB':
            image = cv2.imread(path + '.JPG', cv2.IMREAD_ANYCOLOR)   # BGR
            
            b, g, r = cv2.split(image)
            if hist:
                # b = cv2.equalizeHist(b)
                # g = cv2.equalizeHist(g)
                # r = cv2.equalizeHist(r)
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
        
        # X = image[0].numpy()
        # cv2.imshow("", X)
        # cv2.waitKey(0)

        return image
    
    def __getitem__(self, index):
        """
        Params:
            index:  {int} index of sample
        Returns:
            image:  {tensor(C, H, W)}
            label:  {int}
        """

        image, label = self.samplelist[index]
        return image, label

    def __len__(self):
        """
        Returns:
            length: {int} number of samples
        """
        
        return len(self.samplelist)


if __name__ == "__main__":
    # trainset = RecognizeDataset('/media/louishsu/备份/ECUST2019_64x64', 'Multi', 'split_64x64_1', 'train', [550+i*20 for i in range(23)])
    trainset = RecognizeDataset('/media/louishsu/备份/ECUST2019_64x64', 'Multi', 'split_64x64_1', 'train', [550+i*20 for i in range(23)], False)
    for i in range(10):
        X, y = trainset[i]
