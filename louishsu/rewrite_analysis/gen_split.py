import os
import random
from utiles import getVol, getWavelen


def splitDatasets_Multi_23channels(datapath, splitmode):
    """
    从每份多光谱数据(23通道, 550nm~990nm, 间隔20nm)中依次抽取训练集样本，将相邻的波段放入验证集，其余放入测试集
    如果选中的训练样本相邻波段已被放入验证集，则不再放入
    训练集样本个数设置为6，则验证集12左右，剩余5张放入测试集
    """
    ## 初始化参数
    n_train_per_sample = 6
    FILENAME = "DATA{}/{}/Multi/{}/Multi_{}_W1_{}"
    obtypes = ['normal', 'illum1', 'illum2']
    subidx  = [i+1 for i in range(63)]
    posidx  = [i+1 for i in range(7)]
    sessidx = [i+1 for i in range(10)]
    n_train, n_valid, n_test = 0, 0, 0

    ## 创建文件目录与`.txt`文件
    splitdir = "./split/{}".format(splitmode)
    if not os.path.exists(splitdir): os.makedirs(splitdir)
    train_txt = "{}/train.txt".format(splitdir)
    valid_txt = "{}/valid.txt".format(splitdir)
    test_txt  = "{}/test.txt".format(splitdir)
    ftrain = open(train_txt, 'w'); fvalid = open(valid_txt, 'w'); ftest  = open(test_txt, 'w')


    for i in subidx:
        for obtype in obtypes:
            for pos in posidx:
                for sess in sessidx:
                    filename = FILENAME.format(getVol(i), i, obtype, pos, sess)
                    filepath = os.path.join(datapath, filename)
                    
                    if os.path.exists(filepath):
                        wavelen = [550 + 20*i for i in range(23)]
                        bmpfiles = os.listdir(filepath)

                        ## 选出6个训练集样本
                        for i_train_per_sample in range(n_train_per_sample):
                            wl = random.sample(wavelen, 1)[0]   # 随机选取一个
                            wla = wl - 20                       # 相邻1
                            wls = wl + 20                       # 相邻2

                            ## 保存样本
                            for bmp in bmpfiles:
                                if getWavelen(bmp) == wl:
                                    ftrain.write(os.path.join(datapath.split('/')[-1], filename, bmp) + '\n')
                                    n_train += 1
                                elif getWavelen(bmp) == wla or getWavelen(bmp) == wls:
                                    fvalid.write(os.path.join(datapath.split('/')[-1], filename, bmp) + '\n')
                                    n_valid += 1

                            ## 删除这些样本
                            if wl in wavelen: wavelen.remove(wl)
                            if wla in wavelen: wavelen.remove(wla)
                            if wls in wavelen: wavelen.remove(wls)

                        ## 其余作为测试集
                        for wl in wavelen:
                            for bmp in bmpfiles:
                                if getWavelen(bmp) == wl:
                                    ftest.write(os.path.join(datapath.split('/')[-1], filename, bmp) + '\n')
                                    n_test += 1

    n_items = n_train + n_valid + n_test
    print_log = 'n_items: {}, n_train: {}, n_valid: {}, n_test: {}, ratio: {:.3f}: {:.3f}: {:.3f}'.\
                    format(n_items, n_train, n_valid, n_test, n_train / n_items, n_valid / n_items, n_test  / n_items)
    print(print_log)
    with open('{}/note.txt'.format(splitdir), 'w') as f:
        f.write(print_log)

    ftrain.close(); fvalid.close(); ftest.close()


if __name__ == "__main__":
    from config import configer
    splitDatasets_Multi_23channels(configer.datapath, configer.splitmode)