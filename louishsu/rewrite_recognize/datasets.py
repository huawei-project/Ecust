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
            
        ```
    """
    labels = [i+1 for i in range(63)]
    
    def __init__(self, datapath, type, splitmode, mode, usedChannels):
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
        self.samplelist = [[self.__load_image(filename, type, usedChannels), self.labels.index(getLabel(filename))] for filename in filelist]
    
    def __load_image(self, path, type, usedChannels):
        """
        Params:
            path:   {str} 
            type:   {str} 'Multi', 'RGB'
            usedChannels:   {list[int] / str}
        Returns:
            image:  {tensor(C, H, W)}
        """

        if type == 'Multi':
            dict_wave_path = {getWavelen(img): os.path.join(path, img) for img in os.listdir(path)}
            image = None
            for ch in usedChannels:
                img = cv2.imread(dict_wave_path[ch], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
                if image is None:
                    image = img
                else:
                    image = np.concatenate([image, img], axis=2)


        elif type == 'RGB':
            image = cv2.imread(path + '.JPG', cv2.IMREAD_ANYCOLOR)   # BGR
            
            if usedChannels == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                b, g, r = cv2.split(image)
                b = b[:, :, np.newaxis]; g = g[:, :, np.newaxis]; r = r[:, :, np.newaxis]
                if usedChannels == 'R':
                    image = r
                elif usedChannels == 'G':
                    image = g
                elif usedChannels == 'B':
                    image = b
        
        image = ToTensor()(image)
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


