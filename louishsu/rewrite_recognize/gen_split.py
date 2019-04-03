import os
import random
from utiles import getVol


def gen_Multi_split(datapath, splitmode, train=0.6, valid=0.2, test=0.2):
    """ 划分数据集
    Notes:
        每人的多光谱数据按比例划分
    """
    ## 初始化参数
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
        subfiles = []

        for pos in posidx:
            for obtype in obtypes:
                for sess in sessidx:

                    filename = FILENAME.format(getVol(i), i, obtype, pos, sess)
                    filepath = os.path.join(datapath, filename)
                    if os.path.exists(filepath):
                        subfiles += ['{}/{}'.format(datapath.split('/')[-1], filename) + '\n']
        
        i_items = len(subfiles)
        i_train = int(i_items*train); n_train += i_train
        i_valid = int(i_items*valid); n_valid += i_valid
        i_test  = i_items - i_train - i_valid; n_test += i_test

        trainfiles = random.sample(subfiles, i_train)
        subfiles_valid_test = [f for f in subfiles if f not in trainfiles]
        validfiles = random.sample(subfiles_valid_test, i_valid)
        testfiles  = [f for f in subfiles_valid_test if f not in validfiles]

        ftrain.writelines(trainfiles)
        fvalid.writelines(validfiles)
        ftest.writelines(testfiles)

    n_items = n_train + n_valid + n_test
    print_log = 'n_items: {}, n_train: {}, n_valid: {}, n_test: {}, ratio: {:.3f}: {:.3f}: {:.3f}'.\
                    format(n_items, n_train, n_valid, n_test, n_train / n_items, n_valid / n_items, n_test  / n_items)
    print(print_log)
    with open('{}/note.txt'.format(splitdir), 'w') as f:
        f.write(print_log)
    
    ftrain.close(); fvalid.close(); ftest.close()

def gen_RGB_split(datapath, splitmode):
    """ 根据已划分的多光谱数据集产生RGB数据集
    """

    def multi2rgb(path):
        path = path.split('/')
        path[-3] = 'RGB'
        file = path[-1].split('_')
        file[0] = 'RGB'
        path[-1] = '_'.join(file)
        path = '/'.join(path)
        return path

    for mode in ['train', 'valid', 'test']:
        txtfile = "./split/{}/{}.txt".format(splitmode, mode)
        with open(txtfile, 'r') as f:
            filenames = f.readlines()
        filenames = [multi2rgb(filename) for filename in filenames]
        filenames = [filename for filename in filenames if os.path.exists(os.path.join('/'.join(datapath.split('/')[:-1]), filename.strip())+'.JPG')]
        txtfile = "./split/{}/{}_rgb.txt".format(splitmode, mode)
        with open(txtfile, 'w') as f:
            f.writelines(filenames)

def gen_split(datapath, splitmode, train=0.6, valid=0.2, test=0.2):

    gen_Multi_split(datapath, splitmode, train, valid, test)
    gen_RGB_split(datapath, splitmode)

if __name__ == "__main__":
    from config import configer
    gen_split(configer.datapath, configer.splitmode, train=0.6, valid=0.2, test=0.2)