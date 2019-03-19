import os
import torch
import numpy as np

get_vol = lambda i: (i - 1) // 10 + 1
get_wavelen = lambda bmp: int(bmp.split('.')[0].split('_')[-1])

def get_label_from_path(path):
    path_split = path.split('/')
    idx = path_split.index('ECUST2019')
    label = int(path_split[idx+2])
    return label

def accuracy(y_pred_prob, y_true, multi=False):
    """
    Params:
        y_pred_prob:{tensor(N, n_classes) or tensor(N, C, n_classes)}
        y_true:     {tensor(N)}
        multi:      {bool}
    """
    acc = 0

    if not multi:
        y_pred = torch.argmax(y_pred_prob, 1)
        acc = torch.mean((y_pred==y_true).float())
    else:
        C = y_pred_prob.shape[1]
        for c in range(C):
            y_pred = torch.argmax(y_pred_prob[:, c], 1)
            acc += torch.mean((y_pred==y_true).float())
        acc /= C
    
    return acc