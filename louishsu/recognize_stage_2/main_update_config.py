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
    n_epoch:        {int}                   总计迭代周期数
    stepsize:       {int}                   学习率衰减周期
    batchsize:      {int}                   批次大小
    lrbase:         {float}                 初始学习率
    gamma:          {float}                 学习率衰减倍率
    cuda:           {bool}                  是否使用cuda加速
    dsize:          {tuple(int, int)}       数据空间尺寸
    n_channel:      {int}                   数据通道数
    n_class:        {int}                   数据类别数
    datatype:       {str}                   多光谱或可见光
    usedChannels:   {list[int] or str}      使用的通道索引
    splitratio:     {list[float(3)]}        划分比例
    splitcount:     {int}                   当前划分比例下，当前次随机划分
    modelbase:      {str}                   使用模型
    datapath:       {str}                   数据根目录
    savepath:       {str}                   程序输出目录
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

def main_3_1(make_table_figure=False):

    datatypes   = ["Multi", "RGB"]
    splitcounts = [i for i in range(1, 6)]
    trains      = [0.1*(i + 1) for i in range(7)]
    H, W = len(splitcounts), len(trains)

    if get_table_figure:

        for datatype in datatypes:

            print("Generating tables and figures [{}]...".format(datatype))

            table_data = np.loadtxt("images/3_1_<data>_[{}].txt".format(datatype))
            table_data_acc, table_data_loss = np.vsplit(table_data, 2)      # 按竖直方向划分为两块，即上下

            ## 做表格
            head_name = "count/比例"
            rows_name = [str(i) for i in splitcounts] + ['average']
            cols_name = ["{:.2f}: {:.2f}: 0.2".format(i, 0.8 - i) for i in trains]
            
            table_acc  = gen_markdown_table_2d(head_name, rows_name, cols_name, table_data_acc)
            table_loss = gen_markdown_table_2d(head_name, rows_name, cols_name, table_data_loss)
        
            with open("images/3_1_<table>_[{}].txt".format(datatype), 'w') as f:
                f.write("\n\nacc\n")
                f.write(table_acc)
                f.write("\n\nloss\n")
                f.write(table_loss)
            
            ## 作图
            plt.figure()
            plt.subplot(121); plt.title("acc");  plt.bar(np.arange(avg_acc.shape[0]),  avg_acc )
            plt.subplot(122); plt.title("loss"); plt.bar(np.arange(avg_loss.shape[0]), avg_loss)
            plt.savefig("images/3_1_<figure>_[{}].png".format(datatype))

        return

    start_time = time.time(); elapsed_time = 0

    for datatype in datatypes:

        data_acc  = np.zeros(shape=(H, W))
        data_loss = np.zeros(shape=(H, W))

        for i in range(H):                  # 1, 2, ..., 5

            splitcount = splitcounts[i]
            test = 0.2

            for j in range(W):

                valid = 1 - test - trains[j]
                splitratio = [train, valid, test]

                configer = get_configer(splitratio=splitratio, splitcount=splitcount)

                elapsed_time += time.time() - start_time
                start_time    = time.time()
                print("Main 3.1 <{}> [{}]... Elaped >>> {} min".\
                            format(configer.datatype, configer.splitmode, elapsed_time/60))

                train(configer)

                data_acc[i, j], data_loss[i, j] = test(configer)
        
        ## 保存数据
        avg_acc  = np.mean(data_acc,  axis=0)
        avg_loss = np.mean(data_loss, axis=0)

        table_data_acc  = np.r_[data_acc,  avg_acc.reshape(1, -1) ]
        table_data_loss = np.r_[data_loss, avg_loss.reshape(1, -1)]
        table_data      = np.r_[table_data_acc, table_data_loss]

        np.savetxt("images/3_1_<data>_[{}].txt".format(datatype), table_data)

def main_3_2():

    pass

def main_3_3():

    pass

def main_3_4():

    pass

def main_3_5():

    pass