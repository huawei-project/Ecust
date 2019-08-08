import os
import time
import numpy as np
from matplotlib import pyplot as plt 
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import is_available, empty_cache
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from datasets import RecognizeDataset
from models import modeldict
from utils import accuracy, gen_markdown_table_2d

from train import train
from test  import test

def get_configer(n_epoch=300, stepsize=250, batchsize=32, lrbase=0.001, gamma=0.2, cuda=True, 
                dsize=(112, 96), n_channel=25, n_class=92, datatype='Multi', 
                usedChannels=[i+1 for i in range(25)], splitratio=[0.6, 0.2, 0.2], 
                splitcount=1, modelbase='recognize_vgg11_bn',
                datapath = "/datasets/ECUSTDETECT",
                savepath = '/home/louishsu/Work/Workspace/HUAWEI/stage2'):
    """
    Params:
    n_epoch:        {int}
    stepsize:       {int}
    batchsize:      {int}
    lrbase:         {float}
    gamma:          {float}
    cuda:           {bool}
    dsize:          {tuple(int, int)}
    n_channel:      {int}
    n_class:        {int}
    datatype:       {str}
    usedChannels:   {list[int] or str}
    splitratio:     {list[float(3)]}
    splitcount:     {int}
    modelbase:      {str}
    datapath:       {str}
    savepath:       {str}
    """
    configer = EasyDict()

    ## -------------------------- 训练相关 --------------------------
    configer.n_epoch  = n_epoch
    configer.stepsize = stepsize
    configer.batchsize = batchsize
    configer.lrbase = lrbase
    configer.gamma = gamma
    configer.cuda = cuda
    configer.savepath = savepath

    ## ------------------------- 数据集相关 -------------------------
    configer.datapath = datapath
    configer.dsize = (112, 96)
    configer.n_channel = 25                 # 一份多光谱数据，包含25通道
    configer.n_class = 92                   # 人员数目共92人

    configer.datatype = datatype            # "Multi", "RGB"
    configer.usedChannels = usedChannels    # 多光谱: 列表，包含所用通道索引(1~25)； 可见光: 字符串，"RGB"或"R", "G", "B"
    configer.splitratio = splitratio        # 划分比例
    configer.splitcount = splitcount

    ## -------------------------- 模型相关 --------------------------
    configer.modelbase = modelbase

    ## ========================== 无需修改 ==========================
    configer.splitmode = 'split_{}x{}_[{:.2f}:{:.2f}:{:.2f}]_[{:d}]'.\
                format(configer.dsize[0], configer.dsize[1], 
                configer.splitratio[0], configer.splitratio[1], configer.splitratio[2], 
                configer.splitcount)
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '[{}]_{}_[{}]'.\
                    format(configer.modelbase, configer.splitmode, 
                            '_'.join(list(map(str, configer.usedChannels))) \
                                if isinstance(configer.usedChannels, list) \
                                else configer.usedChannels)

    configer.logspath = '{}/{}/logs'.format(configer.savepath, configer.modelname)
    configer.mdlspath = '{}/{}/models'.format(configer.savepath, configer.modelname)

    return configer

def main_3_1():

    splitcounts = [i for i in range(1, 6)]
    trains      = [0.1*(i + 1) for i in range(7)]
    H, W = len(splitcounts), len(trains)

    data_acc  = np.zeros(shape=(H, W))
    data_loss = np.zeros(shape=(H, W))

    for i in range(H):                  # 1, 2, ..., 5

        splitcount = splitcounts[i]
        test = 0.2

        for j in range(W):

            valid = 1 - test - trains[j]

            splitratio = [train, valid, test]

            configer = get_configer(splitratio=splitratio, splitcount=splitcount)

            train(configer)
            data_acc[i, j], data_loss[i, j] = test(configer)
    
    avg_acc  = np.mean(data_acc,  axis=0)
    avg_loss = np.mean(data_loss, axis=0)

    ## 保存数据
    table_data_acc  = np.r_[data_acc,  avg_acc.reshape(1, -1) ]
    table_data_loss = np.r_[data_loss, avg_loss.reshape(1, -1)]
    table_data = np.concatenate([table_data_acc[np.newaxis], 
                                table_data_loss[np.newaxis]], axis=0)
    np.savetxt("images/数据3_1.txt", table_data)

    ## 做表格
    head_name = "count/比例"
    rows_name = [str(i) for i in splitcounts] + ['average']
    cols_name = ["{:.2f}: {:.2f}: 0.2".format(i, 0.8 - i) for i in trains]
    
    table_acc  = gen_markdown_table_2d(head_name, rows_name, cols_name, table_data_acc)
    table_loss = gen_markdown_table_2d(head_name, rows_name, cols_name, table_data_loss)

    with open("images/表3_1.txt", 'w') as f:
        f.write("\n\nacc\n")
        f.write(table_acc)
        f.write("\n\nloss\n")
        f.write(table_loss)
    
    ## 作图
    plt.figure()
    plt.subplot(121); plt.title("acc");  plt.bar(np.arange(avg_acc.shape[0]),  avg_acc )
    plt.subplot(122); plt.title("loss"); plt.bar(np.arange(avg_loss.shape[0]), avg_loss)
    plt.savefig("images/图3_1.png")
    

def main_3_2():

    pass

def main_3_3():

    pass

def main_3_4():

    pass

def main_3_5():

    pass