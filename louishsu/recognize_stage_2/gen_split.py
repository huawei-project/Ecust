# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-13 13:09:10
@LastEditTime: 2019-08-13 18:38:51
@Update: 
'''
import os
from os.path import exists, join
import random

from utils import get_path_by_attr

def gen_split(datapath, splitmode, train=0.6, valid=0.2, test=0.2, with_2_3_4=True):
    """ Multi-spectral
    
    Params:
        datapath: {str} e.g. "[prefix]/ECUSTDETECT"
        splitmode: {str} e.g. "split_64x64_[index]"
        train, valid, test: {float} train + valid + test = 1.0
        with_2_3_4: {bool}
    Notes:
    -   保存`datapath`以下的路径；
    -   多光谱划分至最后一级目录，下面包括25张图片；
            e.g. `1/multi/illum1/Multi_4_W1_1/4`
                 `1/multi/illum1/Multi_4_W1_5`
    -   可见光划分至图片文件；
            e.g. `1/rgb/illum1/RGB_4_W1_1/4.jpg`
                 `1/rgb/illum1/RGB_4_W1_5.jpg`
    """
    n_train, n_valid, n_test = 0, 0, 0

    ## 候选属性初始化
    labels = [i + 1 for i in range(92)]                                 # 共92人，编号 1 ~ 92
    image_types = ["Multi", "RGB"]
    illum_types = ["illum1", "illum2", "illum3", "normal"]
    positions = [i + 1 for i in range(7)]                               # 共7个拍摄角度，编号 1 ~ 7
    glass_types = [1, 5, 6]                                             # `1`表示无眼镜，`5`表示戴眼镜，`6`表示太阳镜
    image_indexes = [i + 1 for i in range(4)] if with_2_3_4 else [1]    # 某些角度拍摄4次，编号 1 ~ 4
    channel_indexes = [i + 1 for i in range(25)]                        # 多光谱图像索引，编号 1 ~ 4

    ## 创建文件目录与`.txt`文件
    splitdir = "./split/{}".format(splitmode)
    if not exists(splitdir): os.makedirs(splitdir)
    ftrain_Multi = open("{}/train_Multi.txt".format(splitdir), 'w')
    fvalid_Multi = open("{}/valid_Multi.txt".format(splitdir), 'w')
    ftest_Multi  = open("{}/test_Multi.txt".format(splitdir),  'w')
    ftrain_RGB = open("{}/train_RGB.txt".format(splitdir), 'w')
    fvalid_RGB = open("{}/valid_RGB.txt".format(splitdir), 'w')
    ftest_RGB  = open("{}/test_RGB.txt".format(splitdir),  'w')

    ## 遍历每个文件
    for label in labels:

        existing_attributes = []

        for illum_type in illum_types:
            for position in positions:
                for glass_type in glass_types:

                    if glass_type == 5: # 太阳镜下只拍摄一组
                        
                        ## 多光谱文件夹
                        Multi_path = get_path_by_attr(label, "Multi", 
                            illum_type, position, glass_type, None, 0)

                        ## 可见光文件
                        RGB_path = get_path_by_attr(label, "RGB",
                            illum_type, position, glass_type)

                        if not (exists(join(datapath, Multi_path)) and exists(join(datapath, RGB_path))):       # 均存在
                            continue

                        existing_attributes += [[label, illum_type, position, glass_type, None, 0]]

                    else:               # 无眼镜、近视镜下拍摄4组
                        
                        for image_index in image_indexes:
                        
                            ## 多光谱文件夹
                            Multi_path = get_path_by_attr(label, "Multi", 
                                illum_type, position, glass_type, image_index, 0)
                            
                            ## 可见光文件
                            RGB_path = get_path_by_attr(label, "RGB",
                                illum_type, position, glass_type, image_index)
                                    
                            if not (exists(join(datapath, Multi_path)) and exists(join(datapath, RGB_path))):   # 均存在
                                continue

                            existing_attributes += [[label, illum_type, position, glass_type, image_index, 0]]
        
        i_items = len(existing_attributes)
        i_train = int(i_items*train); n_train += i_train
        i_valid = int(i_items*valid); n_valid += i_valid
        i_test  = i_items - i_train - i_valid; n_test += i_test

        train_attributes = random.sample(existing_attributes, i_train)
        valid_test_attributes = list(filter(lambda x: x not in train_attributes, existing_attributes))
        valid_attributes = random.sample(valid_test_attributes, i_valid)
        test_attributes = list(filter(lambda x: x not in valid_attributes, valid_test_attributes))

        train_Multi_paths = list(map(lambda  x: get_path_by_attr(x[0], "Multi", x[1], x[2], x[3], x[4], x[5]) + "\n", train_attributes))
        train_RGB_paths   = list(map(lambda  x: get_path_by_attr(x[0], "RGB",   x[1], x[2], x[3], x[4], x[5]) + "\n", train_attributes))
        valid_Multi_paths = list(map(lambda  x: get_path_by_attr(x[0], "Multi", x[1], x[2], x[3], x[4], x[5]) + "\n", valid_attributes))
        valid_RGB_paths   = list(map(lambda  x: get_path_by_attr(x[0], "RGB",   x[1], x[2], x[3], x[4], x[5]) + "\n", valid_attributes))
        test_Multi_paths  = list(map(lambda  x: get_path_by_attr(x[0], "Multi", x[1], x[2], x[3], x[4], x[5]) + "\n", test_attributes))
        test_RGB_paths    = list(map(lambda  x: get_path_by_attr(x[0], "RGB",   x[1], x[2], x[3], x[4], x[5]) + "\n", test_attributes))

        ftrain_Multi.writelines(train_Multi_paths); fvalid_Multi.writelines(valid_Multi_paths); ftest_Multi.writelines(test_Multi_paths)
        ftrain_RGB.writelines(train_RGB_paths); fvalid_RGB.writelines(valid_RGB_paths); ftest_RGB.writelines(test_RGB_paths)

    ## 统计数目
    n_items = n_train + n_valid + n_test
    print_log = '[{}] n_items: {}, n_train: {}, n_valid: {}, n_test: {}, ratio: {:.3f}: {:.3f}: {:.3f}'.\
                    format(splitmode, n_items, n_train, n_valid, n_test, n_train / n_items, n_valid / n_items, n_test  / n_items)
    print(print_log)
    with open('{}/note.txt'.format(splitdir), 'w') as f:
        f.write(print_log)

    ## 关闭文件
    ftrain_Multi.close(); fvalid_Multi.close(); ftest_Multi.close()
    ftrain_RGB.close(); fvalid_RGB.close(); ftest_RGB.close()


if __name__ == "__main__":

    REPEAT = 5
    test = 0.2
    TRAIN = [0.1*(i+1) for i in range(7)]    # 0.1, ..., 0. 7
    WITH_2_3_4 = False

    for train in TRAIN:
        valid = 1 - test - train
        
        for i in range(REPEAT):

            gen_split("/datasets/ECUSTDETECT", "split_112x96_[{:.2f}:{:.2f}:{:.2f}]_[{}]".format(
                            train, valid, test, i + 1), train, valid, test, with_2_3_4=WITH_2_3_4)