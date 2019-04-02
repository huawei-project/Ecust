import os
import cv2
import numpy as np

from torch.utils.data       import Dataset
from torchvision.transforms import ToTensor

class RecognizeDataset(Dataset):
    """ 识别数据集
    """

    def __init__(self, type, mode):
        """
        Params:
            type:   {str} 'Multi', 'RGB'
            mode:   {str} 'train', 'valid', 'test'
        """
        if type == 'Multi':
            pass
        
        if type == 'RGB'
            pass