"""
# Author: Yuru Chen
# Time: 2019 03 20
"""
import os
import sys
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def get_id(x): return int(x.split('/')[1])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_vol(i): return (i - 1) // 10 + 1


def get_glasses(x): return int(x[x.find('W') + 3])


def is_glasses_1(x): return get_glasses(x) == 5  # glasses


def is_glasses_2(x): return 6 <= get_glasses(x) <= 7  # sunglasses


def is_glasses_3(x): return not (5 <= get_glasses(x) <= 7)  # nonglasses


def get_glasses_mask(g, fLs, fRs):
    # eval faster than globals
    mask_glassesL = np.array(list(map(eval('is_glasses_{}'.format(g)), fLs)))
    mask_glassesR = np.array(list(map(eval('is_glasses_{}'.format(g)), fRs)))
    if g <= 2:
        mask_glasses = np.logical_or(mask_glassesL, mask_glassesR)
    else:
        mask_glasses = np.logical_and(mask_glassesL, mask_glassesR)
    return mask_glasses


def get_path_from_condition(data_list, condition):
    rel = list(filter(lambda x: condition in x, data_list))
    assert len(rel) != 0, 'could not get path from condition {}!'.format(
        condition)
    return rel


def get_path_from_id(data_list, ID):
    rel = list(filter(lambda x: int(x.split('/')[1]) == ID, data_list))
    assert len(rel) != 0, 'could not get path from ID {}!'.format(ID)
    return rel


def get_path_from_id_range(data_list, low, high):
    rel = list(filter(lambda x: low <= int(
        x.split('/')[1]) <= high, data_list))
    assert len(rel) != 0, 'could not get path from id range {} to {}!'.format(
        low, high)
    return rel


def get_path_from_band(data_list, band):
    rel = list(filter(lambda x: int(
        x.split('_')[-1].split('.')[0]) == band, data_list))
    assert len(rel) != 0, 'could not get path from band {}!'.format(band)
    return rel


def get_path_from_band_range(data_list, low, high):
    rel = list(filter(lambda x: low <= int(
        x.split('_')[-1].split('.')[0]) <= high, data_list))
    assert len(rel) != 0, 'could not get path from band range {} to {}!'.format(
        low, high)
    return rel


def get_path_from_sunglasses(data_list):
    rel = list(filter(lambda x: 6 <= int(
        x[x.find('W') + 3:x.find('W') + 4]) <= 7, data_list))
    #assert len(rel) != 0, 'could not get path from sunglasses!'
    return rel


def get_path_from_glasses(data_list):
    rel = list(filter(lambda x: int(
        x[x.find('W') + 3:x.find('W') + 4]) == 5, data_list))
    #assert len(rel) != 0, 'could not get path from glasses!'
    return rel


def get_path_from_sun_or_glasses(data_list):
    rel = list(filter(lambda x: 5 <= int(
        x[x.find('W') + 3:x.find('W') + 4]) <= 7, data_list))
    #assert len(rel) != 0, 'could not get path from sun or glasses!'
    return rel


def get_path_from_nonglasses(data_list):
    rel = list(filter(lambda x: not (5 <= int(
        x[x.find('W') + 3:x.find('W') + 4]) <= 7), data_list))
    #assert len(rel) != 0, 'could not get path from nonglasses!'
    return rel


def get_path_from_pose(data_list, pose):
    assert 1 <= pose <= 7, 'pose index error!'
    rel = list(filter(lambda x: int(
        x[x.find('W') - 2:x.find('W') - 1]) == pose, data_list))
    return rel


def get_multi_dir(data_list):
    return list(np.unique(['/'.join(x.split('/')[:-1]) for x in data_list]))
# -----------------------------------------------------------------


def split_multi_identification_dataset(split_set, index, num_exp=5):
    splitdir = "./face_identification_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    IDs = range(1, 64)
    labels = dict(zip(IDs, range(len(IDs))))
    # create trainset
    for n in range(num_exp):
        train_txt = "{}/train_exp_{}.txt".format(splitdir, n)
        test_txt = "{}/test_exp_{}.txt".format(splitdir, n)
        ftrain = open(train_txt, 'w')
        ftest = open(test_txt, 'w')
        n_train = 0
        n_test = 0
        for i in IDs:
            id_set = get_path_from_id(split_set, i)
            id_nonglasses_set = get_path_from_nonglasses(id_set)
            id_multi_nonglasses_set = get_multi_dir(id_nonglasses_set)
            for sample in id_multi_nonglasses_set:
                ftrain.write('\t'.join([sample, str(labels[i])]) + '\n')
                n_train += 1

            id_glasses_set = [x for x in id_set if x not in id_nonglasses_set]
            id_multi_glasses_set = get_multi_dir(id_glasses_set)
            for sample in id_multi_glasses_set:
                ftest.write('\t'.join([sample, str(labels[i])]) + '\n')
                n_test += 1
        print('{}th experiment, collect {} train samples, {} test samples.'.format(
            n, n_train, n_test))
        ftrain.close()
        ftest.close()
    return


def split_multi_identification_dataset_cv(split_set, index=21, fold=5):
    splitdir = "./face_identification_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    IDs = range(1, 64)
    labels = dict(zip(IDs, range(len(IDs))))
    # create trainset
    for n in range(fold):
        train_txt = "{}/train_fold_{}.txt".format(splitdir, n)
        test_txt = "{}/test_fold_{}.txt".format(splitdir, n)
        ftrain = open(train_txt, 'w')
        ftest = open(test_txt, 'w')
        n_train = 0
        n_test = 0
        for i in IDs:
            id_set = get_path_from_id(split_set, i)
            id_multi_set = get_multi_dir(id_set)
            number = len(id_multi_set)
            test = int(np.ceil(number / 5))
            id_multi_test_set = id_multi_set[test * n: test * (n + 1)]
            id_multi_train_set = [x for x in id_multi_set
                                  if x not in id_multi_test_set]
            for sample in id_multi_train_set:
                ftrain.write('\t'.join([sample, str(labels[i])]) + '\n')
                n_train += 1

            for sample in id_multi_test_set:
                ftest.write('\t'.join([sample, str(labels[i])]) + '\n')
                n_test += 1
        print('{}th fold, collect {} train samples, {} test samples.'.format(
            n, n_train, n_test))
        ftrain.close()
        ftest.close()
    return


if __name__ == '__main__':
    DATASET_PATH = '/home/lilium/myDataset/ECUST/'  # Your HyperECUST dataset path

    multi_paths = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/Multi/*/**/*.bmp'), recursive=True)
    index = multi_paths[0].find('DATA')
    multi_paths = sorted([x[index:] for x in multi_paths])

    rgb_paths = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/RGB/**/*.JPG'), recursive=True)
    index = rgb_paths[0].find('DATA')
    rgb_paths = sorted([x[index:] for x in rgb_paths])

    print('Multispectral images: {}, rgb images: {}'.format(
        len(multi_paths), len(rgb_paths)))

    nonob_multi_set = get_path_from_condition(multi_paths, 'non-ob')
    print('Multispectral non-ob images: {}'.format(len(nonob_multi_set)))
    ob_multi_set = get_path_from_condition(multi_paths, '/ob')
    print('Multispectral ob images: {}'.format(len(ob_multi_set)))

    nonob_rgb_set = get_path_from_condition(rgb_paths, 'non-ob')
    print('RGB non-ob images: {}'.format(len(nonob_rgb_set)))
    ob_rgb_set = get_path_from_condition(rgb_paths, '/ob')
    print('RGB ob images: {}'.format(len(ob_rgb_set)))

    num_id = 63
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]

    # Example of constructing the dataset path
    #split_multi_identification_dataset(nonob_multi_set, 20)

    split_multi_identification_dataset_cv(nonob_multi_set, 22)
