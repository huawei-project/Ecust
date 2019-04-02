import os
import time
import torch
import numpy as np

getTime     = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
getVol      = lambda subidx: (subidx - 1) // 10 + 1
getWavelen  = lambda path: int(path.split('.')[0].split('_')[-1])
getLabel    = lambda path: int(path[path.find('DATA') + len('DATAx/'):].split('/')[0])

def getDicts(datapath):
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 8)]:
        txtfile = os.path.join(datapath, vol, "detect.txt")
        with open(txtfile, 'r') as f:
            dicts[vol] = eval(f.read())
    return dicts

accuracy = lambda x1, x2: np.mean(x1==x2)

def accuracy(y_pred_prob, y_true):
    """
    Params:
        y_pred_prob:{tensor(N, n_classes) or tensor(N, C, n_classes)}
        y_true:     {tensor(N)}
    Returns:
        acc:        {tensor(1)}
    """
    y_pred = torch.argmax(y_pred_prob, 1)
    acc = torch.mean((y_pred==y_true).float())
    return acc