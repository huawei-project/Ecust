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


def split_multi_verification_dataset(split_set, index, num_exp=5, ratio=0.3, n_pos=4, n_neg=4):
    splitdir = "./face_verfication_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    IDs = range(1, 64)
    # create trainset
    if ratio != 0:
        IDs_tests = []
        for n in range(num_exp):
            train_txt = "{}/train_exp_{:02}.txt".format(splitdir, n)
            ftrain = open(train_txt, 'w')
            n_sample = 0
            IDs_train = np.random.choice(IDs,
                                         int(ratio * len(IDs)), replace=False)
            IDs_train = sorted(IDs_train)
            labels = dict(zip(IDs_train, range(len(IDs_train))))
            for i in IDs_train:
                id_set = get_path_from_id(split_set, i)
                for sample in id_set:
                    ftrain.write('\t'.join([sample, str(labels[i])]) + '\n')
                    n_sample += 1
            print('{}th experiment, collect {} train samples.'.format(n, n_sample))
            ftrain.close()
            IDs_tests.append([x for x in IDs if x not in IDs_train])
    else:
        IDs_tests = [IDs for i in range(num_exp)]

    # create testset (pairs set)
    for n in range(num_exp):
        pairs_txt = "{}/pairs_exp_{:02}.txt".format(splitdir, n)
        fpairs = open(pairs_txt, 'w')
        IDs_test = IDs_tests[n]
        labels = dict(zip(IDs_test, range(len(IDs_test))))
        n_sample = 0
        # create positive samples
        for i in IDs_test:
            id_set = get_path_from_id(split_set, i)
            for j in range(len(bands)):
                band_set = get_path_from_band(id_set, bands[j])
                # find all the availible pairs
                pairs = [(x, y) for i, x in enumerate(band_set)
                         for y in band_set[i + 1:]]
                # randomly choose n_pos pairs
                pairs = np.array(pairs)[
                    np.random.randint(0, len(pairs), n_pos)]
                for p in range(n_pos):
                    fpairs.write('\t'.join([pairs[p][0], str(labels[i]),
                                            pairs[p][1], str(labels[i])]) + '\n')
                    n_sample += 1
        # create negtive samples
        for i in IDs_test:
            # the path of id_i
            id_i_set = get_path_from_id(split_set, i)
            # randomly choose the [n_neg] ids which is not equal to id_i
            id_index = np.random.choice([x for x in IDs_test if x != i],
                                        n_neg, replace=False)
            for k in range(n_neg):
                # the path of id_j
                id_j_set = get_path_from_id(split_set, id_index[k])
                for j in range(len(bands)):
                    # iterate band
                    band_i_set = get_path_from_band(id_i_set, bands[j])
                    band_j_set = get_path_from_band(id_j_set, bands[j])
                    # randomly choose one path for id_i and id_j
                    chosed_i = np.random.choice(band_i_set, 1)[0]
                    chosed_j = np.random.choice(band_j_set, 1)[0]
                    fpairs.write('\t'.join([chosed_i, str(labels[i]),
                                            chosed_j, str(labels[id_index[k]])]) + '\n')
                    n_sample += 1
        print('{}th experiment, collect {} pairs samples.'.format(n, n_sample))
        fpairs.close()
    return


def split_rgb_verification_dataset(split_set, index, num_exp=5):
    splitdir = "./face_verfication_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    IDs = range(1, 64)
    # create testset (pairs set)
    for n in range(num_exp):
        pairs_txt = "{}/pairs_exp_{}.txt".format(splitdir, n)
        pos_txt = "{}/pos_exp_{}.txt".format(splitdir, n)
        neg_txt = "{}/neg_exp_{}.txt".format(splitdir, n)
        fpairs = open(pairs_txt, 'w')
        fpos = open(pos_txt, 'w')
        fneg = open(neg_txt, 'w')
        pos_sample = 0
        neg_sample = 0
        # create positive samples
        for i in IDs:
            id_set = get_path_from_id(split_set, i)
            # find all the availible pairs
            pairs = [(x, y) for i, x in enumerate(id_set)
                     for y in id_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs.write('\t'.join([ii, str(i - 1),
                                        jj, str(i - 1)]) + '\n')
                fpos.write('\t'.join([ii, str(i - 1),
                                      jj, str(i - 1)]) + '\n')
                pos_sample += 1
        # create negtive samples
        pairs_id = [(x, y) for i, x in enumerate(IDs)
                    for y in IDs[i + 1:]]
        for (i, j) in pairs_id:
            # the path of id_i and id_j
            id_i_set = get_path_from_id(split_set, i)
            id_j_set = get_path_from_id(split_set, j)
            for ii in id_i_set:
                for jj in id_i_set:
                    fpairs.write('\t'.join([ii, str(i - 1),
                                            jj, str(j - 1)]) + '\n')
                    fneg.write('\t'.join([ii, str(i - 1),
                                          jj, str(j - 1)]) + '\n')
                    neg_sample += 1
        print('{}th random experiment, collect pos {}, neg {} total {}.'.format(n,
                                                                                pos_sample, neg_sample, pos_sample + neg_sample))
        fpairs.close()
        fpos.close()
        fneg.close()
    return


def split_multi_verification_dataset_cv(split_set, index, fold=5, n_pos=4, n_neg=4):
    splitdir = "./face_verfication_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    IDs = range(1, 64)
    IDs = np.random.choice(IDs, len(IDs), replace=False)
    num_train_id = int(len(IDs) * 0.8)
    num_test_id = len(IDs) - num_train_id
    for f in range(fold):
        # create testset (pairs set)
        IDs_test = IDs[f * num_test_id:(f + 1) * num_test_id]
        IDs_train = [x for x in IDs if x not in IDs_test]

        pairs_txt = "{}/pairs_fold_{}.txt".format(splitdir, f)
        fpairs = open(pairs_txt, 'w')
        IDs_test = sorted(IDs_test)
        labels = dict(zip(IDs_test, range(len(IDs_test))))
        n_sample = 0
        # create positive samples
        for i in IDs_test:
            id_set = get_path_from_id(split_set, i)
            for j in range(len(bands)):
                band_set = get_path_from_band(id_set, bands[j])
                # find all the availible pairs
                pairs = [(x, y) for i, x in enumerate(band_set)
                         for y in band_set[i + 1:]]
                # randomly choose n_pos pairs
                pairs = np.array(pairs)[
                    np.random.randint(0, len(pairs), n_pos)]
                for p in range(n_pos):
                    fpairs.write('\t'.join([pairs[p][0], str(labels[i]),
                                            pairs[p][1], str(labels[i])]) + '\n')
                    n_sample += 1
        # create negtive samples
        for i in IDs_test:
            # the path of id_i
            id_i_set = get_path_from_id(split_set, i)
            # randomly choose the [n_neg] ids which is not equal to id_i
            id_index = np.random.choice([x for x in IDs_test if x != i],
                                        n_neg, replace=False)
            for k in range(n_neg):
                # the path of id_j
                id_j_set = get_path_from_id(split_set, id_index[k])
                for j in range(len(bands)):
                    # iterate band
                    band_i_set = get_path_from_band(id_i_set, bands[j])
                    band_j_set = get_path_from_band(id_j_set, bands[j])
                    # randomly choose one path for id_i and id_j
                    chosed_i = np.random.choice(band_i_set, 1)[0]
                    chosed_j = np.random.choice(band_j_set, 1)[0]
                    fpairs.write('\t'.join([chosed_i, str(labels[i]),
                                            chosed_j, str(labels[id_index[k]])]) + '\n')
                    n_sample += 1
        print('{}th fold, collect {} pairs samples.'.format(f, n_sample))
        fpairs.close()

        # create trainset
        train_txt = "{}/train_fold_{}.txt".format(splitdir, f)
        ftrain = open(train_txt, 'w')
        n_sample = 0
        IDs_train = sorted(IDs_train)
        labels = dict(zip(IDs_train, range(len(IDs_train))))
        for i in IDs_train:
            id_set = get_path_from_id(split_set, i)
            for sample in id_set:
                ftrain.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_sample += 1
        print('{}th fold, collect {} train samples.'.format(f, n_sample))
        ftrain.close()
    return


def split_multi_verification_dataset_5fold(split_set, index, fold=5):
    splitdir = "./face_verfication_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    IDs = range(1, 64)
    IDs = np.random.choice(IDs, len(IDs), replace=False)
    num_train_id = int(len(IDs) * (1 / fold * 4))
    num_test_id = len(IDs) - num_train_id
    for f in range(fold):
        # create testset (pairs set)
        IDs_test = IDs[f * num_test_id:(f + 1) * num_test_id]
        IDs_train = [x for x in IDs if x not in IDs_test]
        pairs_txt = "{}/pairs_fold_{}.txt".format(splitdir, f)
        pos_txt = "{}/pos_fold_{}.txt".format(splitdir, f)
        neg_txt = "{}/neg_fold_{}.txt".format(splitdir, f)
        fpairs = open(pairs_txt, 'w')
        fpos = open(pos_txt, 'w')
        fneg = open(neg_txt, 'w')
        IDs_test = sorted(IDs_test)
        labels = dict(zip(IDs_test, range(len(IDs_test))))
        pos_sample = 0
        neg_sample = 0
        n_sample = 0
        # create positive samples
        for i in IDs_test:
            id_set = get_path_from_id(split_set, i)
            for j in range(len(bands)):
                band_set = get_path_from_band(id_set, bands[j])
                # find all the availible pairs
                pairs = [(x, y) for i, x in enumerate(band_set)
                         for y in band_set[i + 1:]]
                for (ii, jj) in pairs:
                    fpairs.write('\t'.join([ii, str(labels[i]),
                                            jj, str(labels[i])]) + '\n')
                    fpos.write('\t'.join([ii, str(labels[i]),
                                          jj, str(labels[i])]) + '\n')
                    pos_sample += 1
        # create negtive samples
        pairs_id = [(x, y) for i, x in enumerate(IDs_test)
                    for y in IDs_test[i + 1:]]
        for (i, j) in pairs_id:
            # the path of id_i and id_j
            id_i_set = get_path_from_id(split_set, i)
            id_j_set = get_path_from_id(split_set, j)
            for band in bands:
                # iterate band
                id_i_band_set = get_path_from_band(id_i_set, band)
                id_j_band_set = get_path_from_band(id_j_set, band)
                # id_i_band_glasses_set = get_path_from_sun_or_glasses(
                #     id_i_band_set)
                # id_j_band_glasses_set = get_path_from_sun_or_glasses(
                #     id_j_band_set)
                # id_i_band_nonglasses_set = get_path_from_nonglasses(
                #     id_i_band_set)
                # id_j_band_nonglasses_set = get_path_from_nonglasses(
                #     id_j_band_set)
                for ii in id_i_band_set:
                    for jj in id_j_band_set:
                        fpairs.write('\t'.join([ii, str(labels[i]),
                                                jj, str(labels[j])]) + '\n')
                        fneg.write('\t'.join([ii, str(labels[i]),
                                              jj, str(labels[j])]) + '\n')
                        neg_sample += 1
        print('{}th fold, collect pos {}, neg {}, total {}.'.format(
            f, pos_sample, neg_sample, pos_sample + neg_sample))
        fpairs.close()
        fpos.close()
        fneg.close()
        # create trainset
        train_txt = "{}/train_fold_{}.txt".format(splitdir, f)
        ftrain = open(train_txt, 'w')
        n_sample = 0
        IDs_train = sorted(IDs_train)
        labels = dict(zip(IDs_train, range(len(IDs_train))))
        for i in IDs_train:
            id_set = get_path_from_id(split_set, i)
            for sample in id_set:
                ftrain.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_sample += 1
        print('{}th fold, collect {} train samples.'.format(f, n_sample))
        ftrain.close()
    return


def split_multi_verification_dataset_5fold_ob(nonob_set, ob_set,
                                              nonob_rgb_set, ob_rgb_set,
                                              index, fold=5):
    splitdir = "./face_verfication_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    readme_txt = "{}/readme.txt".format(splitdir)
    freadme = open(readme_txt, 'a+')
    IDs = np.arange(1, 64)
    np.random.shuffle(IDs)
    #IDs = np.random.choice(IDs, len(IDs), replace=False)
    num_train_id = int(len(IDs) * (1 / fold * 4))
    num_test_id = len(IDs) - num_train_id
    for f in range(fold):
        # create testset (pairs set)
        IDs_test = IDs[f * num_test_id:(f + 1) * num_test_id]
        IDs_train = [x for x in IDs if x not in IDs_test]
        pairs_txt = "{}/pairs_fold_{}.txt".format(splitdir, f)
        pos_txt = "{}/pos_fold_{}.txt".format(splitdir, f)
        neg_txt = "{}/neg_fold_{}.txt".format(splitdir, f)
        pairs_ob_txt = "{}/pairs_ob_fold_{}.txt".format(splitdir, f)
        pos_ob_txt = "{}/pos_ob_fold_{}.txt".format(splitdir, f)
        neg_ob_txt = "{}/neg_ob_fold_{}.txt".format(splitdir, f)

        pairs_rgb_txt = "{}/pairs_rgb_fold_{}.txt".format(splitdir, f)
        pos_rgb_txt = "{}/pos_rgb_fold_{}.txt".format(splitdir, f)
        neg_rgb_txt = "{}/neg_rgb_fold_{}.txt".format(splitdir, f)
        pairs_ob_rgb_txt = "{}/pairs_ob_rgb_fold_{}.txt".format(splitdir, f)
        pos_ob_rgb_txt = "{}/pos_ob_rgb_fold_{}.txt".format(splitdir, f)
        neg_ob_rgb_txt = "{}/neg_ob_rgb_fold_{}.txt".format(splitdir, f)

        fpairs = open(pairs_txt, 'w')
        fpos = open(pos_txt, 'w')
        fneg = open(neg_txt, 'w')
        fpairs_ob = open(pairs_ob_txt, 'w')
        fpos_ob = open(pos_ob_txt, 'w')
        fneg_ob = open(neg_ob_txt, 'w')

        fpairs_rgb = open(pairs_rgb_txt, 'w')
        fpos_rgb = open(pos_rgb_txt, 'w')
        fneg_rgb = open(neg_rgb_txt, 'w')
        fpairs_ob_rgb = open(pairs_ob_rgb_txt, 'w')
        fpos_ob_rgb = open(pos_ob_rgb_txt, 'w')
        fneg_ob_rgb = open(neg_ob_rgb_txt, 'w')

        IDs_test = sorted(IDs_test)
        labels = dict(zip(IDs_test, range(len(IDs_test))))
        pos = 0
        pos_ob = 0
        neg = 0
        neg_ob = 0
        pos_rgb = 0
        neg_rgb = 0
        pos_ob_rgb = 0
        neg_ob_rgb = 0
        # create positive samples
        for i in IDs_test:
            # rgb
            id_set = get_path_from_id(nonob_rgb_set, i)
            pairs = [(x, y) for i, x in enumerate(id_set)
                     for y in id_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs_rgb.write('\t'.join([ii, str(labels[i]),
                                            jj, str(labels[i])]) + '\n')
                fpos_rgb.write('\t'.join([ii, str(labels[i]),
                                          jj, str(labels[i])]) + '\n')
                pos_rgb += 1
            # rgb_ob
            id_set = get_path_from_id(ob_rgb_set, i)
            pairs = [(x, y) for i, x in enumerate(id_set)
                     for y in id_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                               jj, str(labels[i])]) + '\n')
                fpos_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                             jj, str(labels[i])]) + '\n')
                pos_ob_rgb += 1

            # multi
            id_set = get_path_from_id(nonob_set, i)
            for j in range(len(bands)):
                band_set = get_path_from_band(id_set, bands[j])
                # find all the availible pairs
                pairs = [(x, y) for i, x in enumerate(band_set)
                         for y in band_set[i + 1:]]
                for (ii, jj) in pairs:
                    fpairs.write('\t'.join([ii, str(labels[i]),
                                            jj, str(labels[i])]) + '\n')
                    fpos.write('\t'.join([ii, str(labels[i]),
                                          jj, str(labels[i])]) + '\n')
                    pos += 1
            # multi_ob
            id_set = get_path_from_id(ob_set, i)
            for j in range(len(bands)):
                band_set = get_path_from_band(id_set, bands[j])
                # find all the availible pairs
                pairs = [(x, y) for i, x in enumerate(band_set)
                         for y in band_set[i + 1:]]
                for (ii, jj) in pairs:
                    fpairs_ob.write('\t'.join([ii, str(labels[i]),
                                               jj, str(labels[i])]) + '\n')
                    fpos_ob.write('\t'.join([ii, str(labels[i]),
                                             jj, str(labels[i])]) + '\n')
                    pos_ob += 1

        # create negtive samples
        pairs_id = [(x, y) for i, x in enumerate(IDs_test)
                    for y in IDs_test[i + 1:]]
        for (i, j) in pairs_id:
            # rgb
            # the path of id_i and id_j
            id_i_set = get_path_from_id(nonob_rgb_set, i)
            id_j_set = get_path_from_id(nonob_rgb_set, j)
            for ii in id_i_set:
                for jj in id_j_set:
                    fpairs_rgb.write('\t'.join([ii, str(labels[i]),
                                                jj, str(labels[j])]) + '\n')
                    fneg_rgb.write('\t'.join([ii, str(labels[i]),
                                              jj, str(labels[j])]) + '\n')
                    neg_rgb += 1
            # rgb_ob
            id_i_set = get_path_from_id(ob_rgb_set, i)
            id_j_set = get_path_from_id(ob_rgb_set, j)
            for ii in id_i_set:
                for jj in id_j_set:
                    fpairs_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                                   jj, str(labels[j])]) + '\n')
                    fneg_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                                 jj, str(labels[j])]) + '\n')
                    neg_ob_rgb += 1

            # multi
            # the path of id_i and id_j
            id_i_set = get_path_from_id(nonob_set, i)
            id_j_set = get_path_from_id(nonob_set, j)
            for band in bands:
                # iterate band
                id_i_band_set = get_path_from_band(id_i_set, band)
                id_j_band_set = get_path_from_band(id_j_set, band)
                for ii in id_i_band_set:
                    for jj in id_j_band_set:
                        fpairs.write('\t'.join([ii, str(labels[i]),
                                                jj, str(labels[j])]) + '\n')
                        fneg.write('\t'.join([ii, str(labels[i]),
                                              jj, str(labels[j])]) + '\n')
                        neg += 1
            # multi_ob
            id_i_set = get_path_from_id(ob_set, i)
            id_j_set = get_path_from_id(ob_set, j)
            for band in bands:
                # iterate band
                id_i_band_set = get_path_from_band(id_i_set, band)
                id_j_band_set = get_path_from_band(id_j_set, band)
                for ii in id_i_band_set:
                    for jj in id_j_band_set:
                        fpairs_ob.write('\t'.join([ii, str(labels[i]),
                                                   jj, str(labels[j])]) + '\n')
                        fneg_ob.write('\t'.join([ii, str(labels[i]),
                                                 jj, str(labels[j])]) + '\n')
                        neg_ob += 1

        total = pos + neg
        total_ob = pos_ob + neg_ob
        total_rgb = pos_rgb + neg_rgb
        total_ob_rgb = pos_ob_rgb + neg_ob_rgb
        s = '{}th fold, nonob: pos {} neg {} total {}, ob: pos {} neg {} total {} | nonob_rgb: pos {} neg {} total {}, ob_rgb: pos {} neg {} total {}.'.format(
            f, pos, neg, total, pos_ob, neg_ob, total_ob,
            pos_rgb, neg_rgb, total_rgb, pos_ob_rgb, neg_ob_rgb, total_ob_rgb)
        print(s)
        freadme.write(s + '\n')
        fpairs.close()
        fpos.close()
        fneg.close()
        fpairs_ob.close()
        fpos_ob.close()
        fneg_ob.close()

        fpairs_rgb.close()
        fpos_rgb.close()
        fneg_rgb.close()
        fpairs_ob_rgb.close()
        fpos_ob_rgb.close()
        fneg_ob_rgb.close()

        # create trainset
        train_txt = "{}/train_fold_{}.txt".format(splitdir, f)
        train_ob_txt = "{}/train_ob_fold_{}.txt".format(splitdir, f)
        train_rgb_txt = "{}/train_rgb_fold_{}.txt".format(splitdir, f)
        train_ob_rgb_txt = "{}/train_ob_rgb_fold_{}.txt".format(splitdir, f)

        ftrain = open(train_txt, 'w')
        ftrain_ob = open(train_ob_txt, 'w')
        ftrain_rgb = open(train_rgb_txt, 'w')
        ftrain_ob_rgb = open(train_ob_rgb_txt, 'w')
        n = 0
        n_ob = 0
        n_rgb = 0
        n_ob_rgb = 0

        IDs_train = sorted(IDs_train)
        labels = dict(zip(IDs_train, range(len(IDs_train))))
        # multi
        for i in IDs_train:
            id_set = get_path_from_id(nonob_set, i)
            id_ob_set = get_path_from_id(ob_set, i)
            for sample in id_set:
                ftrain.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n += 1
            for sample in id_ob_set:
                ftrain_ob.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_ob += 1
        # rgb
        for i in IDs_train:
            id_set = get_path_from_id(nonob_rgb_set, i)
            id_ob_set = get_path_from_id(ob_rgb_set, i)
            for sample in id_set:
                ftrain_rgb.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_rgb += 1
            for sample in id_ob_set:
                ftrain_ob_rgb.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_ob_rgb += 1
        s = '{}th fold, nonob {}, ob {} | nonob_rgb {}, ob_rgb {}'.format(
            f, n, n_ob, n_rgb, n_ob_rgb)
        print(s)
        freadme.write(s + '\n')
        ftrain.close()
        ftrain_ob.close()
        ftrain_rgb.close()
        ftrain_ob_rgb.close()
        freadme.close()
    return


def split_multi_verification_dataset_5fold_MI(nonob_set, ob_set,
                                              nonob_rgb_set, ob_rgb_set,
                                              index, fold=5):
    splitdir = "./face_verfication_split/split_{}".format(index)
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    readme_txt = "{}/readme.txt".format(splitdir)
    freadme = open(readme_txt, 'a+')
    IDs = np.arange(1, 64)
    np.random.shuffle(IDs)
    #IDs = np.random.choice(IDs, len(IDs), replace=False)
    num_train_id = int(len(IDs) * (1 / fold * 4))
    num_test_id = len(IDs) - num_train_id
    for f in range(fold):
        # create testset (pairs set)
        IDs_test = IDs[f * num_test_id:(f + 1) * num_test_id]
        IDs_train = [x for x in IDs if x not in IDs_test]
        pairs_txt = "{}/pairs_fold_{}.txt".format(splitdir, f)
        pos_txt = "{}/pos_fold_{}.txt".format(splitdir, f)
        neg_txt = "{}/neg_fold_{}.txt".format(splitdir, f)
        pairs_ob_txt = "{}/pairs_ob_fold_{}.txt".format(splitdir, f)
        pos_ob_txt = "{}/pos_ob_fold_{}.txt".format(splitdir, f)
        neg_ob_txt = "{}/neg_ob_fold_{}.txt".format(splitdir, f)

        pairs_rgb_txt = "{}/pairs_rgb_fold_{}.txt".format(splitdir, f)
        pos_rgb_txt = "{}/pos_rgb_fold_{}.txt".format(splitdir, f)
        neg_rgb_txt = "{}/neg_rgb_fold_{}.txt".format(splitdir, f)
        pairs_ob_rgb_txt = "{}/pairs_ob_rgb_fold_{}.txt".format(splitdir, f)
        pos_ob_rgb_txt = "{}/pos_ob_rgb_fold_{}.txt".format(splitdir, f)
        neg_ob_rgb_txt = "{}/neg_ob_rgb_fold_{}.txt".format(splitdir, f)

        fpairs = open(pairs_txt, 'w')
        fpos = open(pos_txt, 'w')
        fneg = open(neg_txt, 'w')
        fpairs_ob = open(pairs_ob_txt, 'w')
        fpos_ob = open(pos_ob_txt, 'w')
        fneg_ob = open(neg_ob_txt, 'w')

        fpairs_rgb = open(pairs_rgb_txt, 'w')
        fpos_rgb = open(pos_rgb_txt, 'w')
        fneg_rgb = open(neg_rgb_txt, 'w')
        fpairs_ob_rgb = open(pairs_ob_rgb_txt, 'w')
        fpos_ob_rgb = open(pos_ob_rgb_txt, 'w')
        fneg_ob_rgb = open(neg_ob_rgb_txt, 'w')

        IDs_test = sorted(IDs_test)
        labels = dict(zip(IDs_test, range(len(IDs_test))))
        pos, neg = 0, 0
        pos_ob, neg_ob = 0, 0
        pos_rgb, neg_rgb = 0, 0
        pos_ob_rgb, neg_ob_rgb = 0, 0
        # create positive samples
        for i in IDs_test:
            # rgb
            id_set = get_path_from_id(nonob_rgb_set, i)
            pairs = [(x, y) for i, x in enumerate(id_set)
                     for y in id_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs_rgb.write('\t'.join([ii, str(labels[i]),
                                            jj, str(labels[i])]) + '\n')
                fpos_rgb.write('\t'.join([ii, str(labels[i]),
                                          jj, str(labels[i])]) + '\n')
                pos_rgb += 1
            # rgb_ob
            id_set = get_path_from_id(ob_rgb_set, i)
            pairs = [(x, y) for i, x in enumerate(id_set)
                     for y in id_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                               jj, str(labels[i])]) + '\n')
                fpos_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                             jj, str(labels[i])]) + '\n')
                pos_ob_rgb += 1
            # multi
            id_set = get_path_from_id(nonob_set, i)
            id_multi_set = get_multi_dir(id_set)
            # find all the availible pairs
            pairs = [(x, y) for i, x in enumerate(id_multi_set)
                     for y in id_multi_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs.write('\t'.join([ii, str(labels[i]),
                                        jj, str(labels[i])]) + '\n')
                fpos.write('\t'.join([ii, str(labels[i]),
                                      jj, str(labels[i])]) + '\n')
                pos += 1
            # multi_ob
            id_set = get_path_from_id(ob_set, i)
            id_multi_set = get_multi_dir(id_set)
            # find all the availible pairs
            pairs = [(x, y) for i, x in enumerate(id_multi_set)
                     for y in id_multi_set[i + 1:]]
            for (ii, jj) in pairs:
                fpairs_ob.write('\t'.join([ii, str(labels[i]),
                                           jj, str(labels[i])]) + '\n')
                fpos_ob.write('\t'.join([ii, str(labels[i]),
                                         jj, str(labels[i])]) + '\n')
                pos_ob += 1
        # create negtive samples
        pairs_id = [(x, y) for i, x in enumerate(IDs_test)
                    for y in IDs_test[i + 1:]]
        for (i, j) in pairs_id:
            # rgb
            # the path of id_i and id_j
            id_i_set = get_path_from_id(nonob_rgb_set, i)
            id_j_set = get_path_from_id(nonob_rgb_set, j)
            for ii in id_i_set:
                for jj in id_j_set:
                    fpairs_rgb.write('\t'.join([ii, str(labels[i]),
                                                jj, str(labels[j])]) + '\n')
                    fneg_rgb.write('\t'.join([ii, str(labels[i]),
                                              jj, str(labels[j])]) + '\n')
                    neg_rgb += 1
            # rgb_ob
            id_i_set = get_path_from_id(ob_rgb_set, i)
            id_j_set = get_path_from_id(ob_rgb_set, j)
            for ii in id_i_set:
                for jj in id_j_set:
                    fpairs_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                                   jj, str(labels[j])]) + '\n')
                    fneg_ob_rgb.write('\t'.join([ii, str(labels[i]),
                                                 jj, str(labels[j])]) + '\n')
                    neg_ob_rgb += 1
            # multi
            # the path of id_i and id_j
            id_i_set = get_path_from_id(nonob_set, i)
            id_i_multi_set = get_multi_dir(id_i_set)
            id_j_set = get_path_from_id(nonob_set, j)
            id_j_multi_set = get_multi_dir(id_j_set)
            for ii in id_i_multi_set:
                for jj in id_j_multi_set:
                    fpairs.write('\t'.join([ii, str(labels[i]),
                                            jj, str(labels[j])]) + '\n')
                    fneg.write('\t'.join([ii, str(labels[i]),
                                          jj, str(labels[j])]) + '\n')
                    neg += 1
            # multi_ob
            id_i_set = get_path_from_id(ob_set, i)
            id_i_multi_set = get_multi_dir(id_i_set)
            id_j_set = get_path_from_id(ob_set, j)
            id_j_multi_set = get_multi_dir(id_j_set)
            for ii in id_i_multi_set:
                for jj in id_j_multi_set:
                    fpairs_ob.write('\t'.join([ii, str(labels[i]),
                                               jj, str(labels[j])]) + '\n')
                    fneg_ob.write('\t'.join([ii, str(labels[i]),
                                             jj, str(labels[j])]) + '\n')
                    neg_ob += 1
        total = pos + neg
        total_ob = pos_ob + neg_ob
        total_rgb = pos_rgb + neg_rgb
        total_ob_rgb = pos_ob_rgb + neg_ob_rgb
        s1 = '{}th fold, nonob: pos {} neg {} total {}, ob: pos {} neg {} total {} | nonob_rgb: pos {} neg {} total {}, ob_rgb: pos {} neg {} total {}.'.format(
            f, pos, neg, total, pos_ob, neg_ob, total_ob,
            pos_rgb, neg_rgb, total_rgb, pos_ob_rgb, neg_ob_rgb, total_ob_rgb)
        s2 = 'testset id {}'.format(IDs_test)
        print(s1)
        print(s2)
        freadme.write(s1 + '\n')
        freadme.write(s2 + '\n')
        fpairs.close()
        fpos.close()
        fneg.close()
        fpairs_ob.close()
        fpos_ob.close()
        fneg_ob.close()

        fpairs_rgb.close()
        fpos_rgb.close()
        fneg_rgb.close()
        fpairs_ob_rgb.close()
        fpos_ob_rgb.close()
        fneg_ob_rgb.close()

        # create trainset
        train_txt = "{}/train_fold_{}.txt".format(splitdir, f)
        train_ob_txt = "{}/train_ob_fold_{}.txt".format(splitdir, f)
        train_rgb_txt = "{}/train_rgb_fold_{}.txt".format(splitdir, f)
        train_ob_rgb_txt = "{}/train_ob_rgb_fold_{}.txt".format(splitdir, f)

        ftrain = open(train_txt, 'w')
        ftrain_ob = open(train_ob_txt, 'w')
        ftrain_rgb = open(train_rgb_txt, 'w')
        ftrain_ob_rgb = open(train_ob_rgb_txt, 'w')
        n = 0
        n_ob = 0
        n_rgb = 0
        n_ob_rgb = 0

        IDs_train = sorted(IDs_train)
        labels = dict(zip(IDs_train, range(len(IDs_train))))
        # multi
        for i in IDs_train:
            id_set = get_path_from_id(nonob_set, i)
            id_multi_set = get_multi_dir(id_set)
            id_ob_set = get_path_from_id(ob_set, i)
            id_ob_multi_set = get_multi_dir(id_ob_set)
            for sample in id_multi_set:
                ftrain.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n += 1
            for sample in id_ob_multi_set:
                ftrain_ob.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_ob += 1
        # rgb
        for i in IDs_train:
            id_set = get_path_from_id(nonob_rgb_set, i)
            id_ob_set = get_path_from_id(ob_rgb_set, i)
            for sample in id_set:
                ftrain_rgb.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_rgb += 1
            for sample in id_ob_set:
                ftrain_ob_rgb.write(
                    '\t'.join([sample, str(labels[i])]) + '\n')
                n_ob_rgb += 1
        s1 = '{}th fold, nonob {}, ob {} | nonob_rgb {}, ob_rgb {}'.format(
            f, n, n_ob, n_rgb, n_ob_rgb)
        s2 = 'trainset id {}'.format(IDs_train)
        print(s1)
        print(s2)
        freadme.write(s1 + '\n')
        freadme.write(s2 + '\n')
        ftrain.close()
        ftrain_ob.close()
        ftrain_rgb.close()
        ftrain_ob_rgb.close()
    freadme.close()
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
    # split_multi_verification_dataset(nonob_multi_set,
    #                                  index=8, num_exp=5, ratio=0, n_pos=4, n_neg=4)
    # split_rgb_verification_dataset(
    #    nonob_rgb_set, index=6, num_exp=5, n_pos=6, n_neg=6)
    #split_multi_verification_dataset_5fold(nonob_multi_set, index=9, fold=5)
    #split_rgb_verification_dataset(ob_rgb_set, index=12)

    # split_multi_verification_dataset_5fold_ob(nonob_multi_set, ob_multi_set,
    #                                           nonob_rgb_set, ob_rgb_set, 13)

    split_multi_verification_dataset_5fold_MI(nonob_multi_set, ob_multi_set,
                                              nonob_rgb_set, ob_rgb_set, 14)
