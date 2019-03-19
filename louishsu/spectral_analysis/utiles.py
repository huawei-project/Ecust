import os
import numpy as np
import torch

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

def fine_tune(modelPretrained, modelToTrain):
    my_model_dict = modelToTrain.state_dict()
    new_model_dict = {key: value for key, value in modelPretrained.state_dict().items() if key in my_model_dict}
    my_model_dict.update(new_model_dict)
    modelToTrain.load_state_dict(my_model_dict)
    return modelToTrain

def adjust_learning_rate(optimizer, step, total_step, epoch, num_epochs, lr, decay_rate=.9, mode='poly'):
    if mode == 'lr_finder':
        start_lr = 1e-7   # 6e-5
        end_lr = 1e-3     # 1e-4
        new_lr = (end_lr - start_lr) / num_epochs * epoch + start_lr
    if mode == 'leslie':
        a1 = 0.4 * num_epochs
        a2 = 2 * a1
        low = 1e-6
        high = 1e-4
        if epoch <= a1:
            new_lr = (high - low) / a1 * epoch + low
        elif epoch <= a2:
            new_lr = (low - high) / a1 * (epoch - a1) + high
        else:
            new_lr = low * (1 - (epoch - a2) / num_epochs) ** decay_rate
    if mode == 'poly':
        new_lr = lr * (1 - step / total_step) ** decay_rate
    if mode == 'step':
        begin = 50
        if epoch < begin:
            new_lr = lr
        else:
            new_lr = lr * (0.5**((epoch - begin) // 50))
    optimizer.param_groups[0]['lr'] = new_lr
    optimizer.param_groups[1]['lr'] = new_lr * 1
    return new_lr

def convert_to_npy_path(path):
    path_split = path.split('/')
    idx = path_split.index('ECUST2019')
    path_split[idx] = 'ECUST2019_NPY'
    path = '/'.join(path_split) + '.npy'
    return path

def get_label_from_path(path):
    path_split = path.split('/')
    idx = path_split.index('ECUST2019')
    label = int(path_split[idx+2])
    return label

def get_labels_from_pathlist(pathlist):
    labels = []
    for path in pathlist:
        label = get_label_from_path(path)
        labels += [label]
    return labels