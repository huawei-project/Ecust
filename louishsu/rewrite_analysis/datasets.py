import os
import cv2
import numpy as np

from torch.utils.data       import Dataset
from torchvision.transforms import ToTensor

from utiles import getLabel

class AnalysisDataset(Dataset):
    """ 分析数据集

    Attributes:
        labels:     {list[int]} label of subjects
        samplelist: {list[list[tensor(C, H, W), int]]}

    Example:
        ```
        from torch.utils.data import DataLoader

        trainset = AnalysisDataset('/datasets/ECUST2019_64x64', 'split_64x64_1', 'train')
        trainloader = DataLoader(trainsets, batch_size, shuffle=True)

        for i, (X, y) in enumerate(trainloader):
            X = X.cuda(); y = y.cuda()

        ```
    """
    labels = [i+1 for i in range(63)]

    def __init__(self, datapath, splitmode, mode):
        """
        Params:
            datapath:   {str} 'xxxx/ECUST2019_xxx'
            splitmode:  {str} 
            mode:       {str} 'train', 'valid', 'test'
        """
        txtfile = './split/{}/{}.txt'.format(splitmode, mode)
        
        with open(txtfile, 'r') as f:
            filelist = f.readlines()
        
        filelist = [os.path.join('/'.join(datapath.split('/')[:-1]), filename.strip()) for filename in filelist]
        self.samplelist = [[self.__load_image(filename), self.labels.index(getLabel(filename))] for filename in filelist]
    
    def __load_image(self, path):
        """
        Params:
            path:   {str} 
        Returns:
            image:  {tensor(C, H, W)}
        """

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = image[:, :, np.newaxis]
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
