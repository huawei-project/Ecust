import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import xlwt
from sklearn import manifold, decomposition
from scipy import interpolate
sys.path.append('../../utils')
from faceverification_utils import getAccuracy, getThreshold, Evaluation_10_fold


def get_id(x): return int(x.split('/')[1])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_pose(x): return int(x[x.find('W') - 2])

#  pose


def is_pose_1(x): return get_pose(x) == 1


def is_pose_2(x): return get_pose(x) == 2


def is_pose_3(x): return get_pose(x) == 3


def is_pose_4(x): return get_pose(x) == 4


def is_pose_5(x): return get_pose(x) == 5


def is_pose_6(x): return get_pose(x) == 6


def is_pose_7(x): return get_pose(x) == 7


def get_pose_mask_(p1, p2, fLs, fRs):
    mask_pose1L = np.array(list(map(globals()['is_pose_{}'.format(p1)], fLs)))
    mask_pose2R = np.array(list(map(globals()['is_pose_{}'.format(p2)], fRs)))
    mask_pose2L = np.array(list(map(globals()['is_pose_{}'.format(p2)], fLs)))
    mask_pose1R = np.array(list(map(globals()['is_pose_{}'.format(p1)], fRs)))
    mask_pose = np.logical_or(np.logical_and(mask_pose1L, mask_pose2R),
                              np.logical_and(mask_pose2L, mask_pose1R))
    return mask_pose


def get_pose_mask(p1, p2, fLs, fRs):
        # eval faster than globals
    mask_pose1L = np.array(list(map(eval('is_pose_{}'.format(p1)), fLs)))
    mask_pose2R = np.array(list(map(eval('is_pose_{}'.format(p2)), fRs)))
    mask_pose2L = np.array(list(map(eval('is_pose_{}'.format(p2)), fLs)))
    mask_pose1R = np.array(list(map(eval('is_pose_{}'.format(p1)), fRs)))
    mask_pose = np.logical_or(np.logical_and(mask_pose1L, mask_pose2R),
                              np.logical_and(mask_pose2L, mask_pose1R))
    return mask_pose


def pose_score_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]

    fold = [0, 1, 2, 3, 4]
    pose = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
    name = ['4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7']
    workbook = xlwt.Workbook(encoding='ascii')
    test_file = '../../workspace/split_11_valid_ob_result_10_fold.mat'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    labels = result['labels'].squeeze()
    predictions = result['predictions'].squeeze()
    n_sample = len(filenameLs)
    image_idLs = np.array(list(map(get_id, filenameLs)))
    image_idRs = np.array(list(map(get_id, filenameRs)))
    image_bandLs = np.array(list(map(get_band, filenameLs)))
    image_bandRs = np.array(list(map(get_band, filenameRs)))
    # ---------------------------------------------------------
    # statistic pos-neg
    mask_pos = labels == 1
    mask_neg = labels == -1
    print('positive samples {}, negative samples {}'.format(mask_pos.sum(),
                                                            mask_neg.sum()))
    # statistic pose
    pos = []
    neg = []
    worksheet = workbook.add_sheet('score')
    # 4_*
    for i, (p1, p2) in enumerate(pose):
        mask_pose = get_pose_mask(p1, p2, filenameLs, filenameRs)
        mask_pose_pos = np.logical_and(mask_pose, mask_pos)
        mask_pose_neg = np.logical_and(mask_pose, mask_neg)
        scores_pos = scores[mask_pose_pos].mean()
        scores_neg = scores[mask_pose_neg].mean()
        pos.append(scores_pos)
        neg.append(scores_neg)
        print('pose {}-{}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(p1, p2, mask_pose.sum(),
                                                                              mask_pose_pos.sum(), scores_pos, mask_pose_neg.sum(), scores_neg))
        worksheet.write(0, i, int(mask_pose.sum()))
        worksheet.write(1, i, int(mask_pose_pos.sum()))
        worksheet.write(2, i, float(scores_pos))
        worksheet.write(3, i, int(mask_pose_neg.sum()))
        worksheet.write(4, i, float(scores_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(pos)), pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(neg)), neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(7), name)
    plt.yticks(np.arange(0, 0.85, 0.05))
    plt.ylabel('cosine')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('./face_verification_pose_score_ob.png', dpi=200)
    plt.show()

    # statistic band
    pos_list = []
    neg_list = []
    worksheet = workbook.add_sheet('score_band')
    for i, band in enumerate(bands):
        mask_band = np.array(list(map(lambda x: get_band(x) == band,
                                      filenameLs)))
        mask_pos_band = np.logical_and(mask_pos, mask_band)
        mask_neg_band = np.logical_and(mask_neg, mask_band)
        pos_band_list = []
        neg_band_list = []
        for j, (p1, p2) in enumerate(pose):
            mask_pose = get_pose_mask(p1, p2, filenameLs, filenameRs)
            mask_pose_band = np.logical_and(mask_pose, mask_band)
            mask_pose_pos_band = np.logical_and(mask_pose, mask_pos_band)
            mask_pose_neg_band = np.logical_and(mask_pose, mask_neg_band)
            scores_pos_band = scores[mask_pose_pos_band].mean()
            scores_neg_band = scores[mask_pose_neg_band].mean()
            pos_band_list.append(scores_pos_band)
            neg_band_list.append(scores_neg_band)
            print('band {}, pose {}-{}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(band, p1, p2, mask_pose_band.sum(),
                                                                                           mask_pose_pos_band.sum(), scores_pos_band,
                                                                                           mask_pose_neg_band.sum(), scores_neg_band,
                                                                                           ))
            worksheet.write(0 + i * 6, j, int(mask_pose_band.sum()))
            worksheet.write(1 + i * 6, j, int(mask_pose_pos_band.sum()))
            worksheet.write(2 + i * 6, j, float(scores_pos_band))
            worksheet.write(3 + i * 6, j, int(mask_pose_neg_band.sum()))
            worksheet.write(4 + i * 6, j, float(scores_neg_band))
        pos_list.append(pos_band_list)
        neg_list.append(neg_band_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    plt.figure(figsize=(5, 5))
    xnew = np.arange(550, 990, 0.1)
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    for i in range(len(pose)):
        ynew = interpolate.interp1d(bands, pos_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
        ynew = interpolate.interp1d(bands, neg_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=name[i] + '_neg')
    plt.xlabel('wavelength')
    plt.ylabel('cosine')
    plt.yticks(np.arange(0, 0.85, 0.05))
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='x-small')
    plt.savefig('./face_verification_pose_score_ob_band.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_pose_ob_score.xls')


def pose_acc_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]

    fold = [0, 1, 2, 3, 4]
    pose = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
    name = ['4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7']
    workbook = xlwt.Workbook(encoding='ascii')
    test_file = '../../workspace/split_11_valid_result_10_fold.mat'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    labels = result['labels'].squeeze()
    predictions = result['predictions'].squeeze()
    n_sample = len(filenameLs)
    image_idLs = np.array(list(map(get_id, filenameLs)))
    image_idRs = np.array(list(map(get_id, filenameRs)))
    image_bandLs = np.array(list(map(get_band, filenameLs)))
    image_bandRs = np.array(list(map(get_band, filenameRs)))
    # ---------------------------------------------------------
    # statistic pos-neg
    mask_pos = labels == 1
    mask_neg = labels == -1
    print('positive samples {}, negative samples {}'.format(mask_pos.sum(),
                                                            mask_neg.sum()))
    # statistic pose
    pos = []
    neg = []
    total = []
    worksheet = workbook.add_sheet('acc')
    # 4_*
    for i, (p1, p2) in enumerate(pose):
        mask_pose = get_pose_mask(p1, p2, filenameLs, filenameRs)
        mask_pose_pos = np.logical_and(mask_pose, mask_pos)
        mask_pose_neg = np.logical_and(mask_pose, mask_neg)
        # pred
        pred_pose = predictions[mask_pose]
        pred_pose_pos = predictions[mask_pose_pos]
        pred_pose_neg = predictions[mask_pose_neg]
        # acc
        acc_total = pred_pose.sum() / mask_pose.sum()
        acc_pos = pred_pose_pos.sum() / mask_pose_pos.sum()
        acc_neg = pred_pose_neg.sum() / mask_pose_neg.sum()
        # append
        total.append(acc_total)
        pos.append(acc_pos)
        neg.append(acc_neg)
        print('pose {}-{}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(p1, p2,
                                                                                      mask_pose.sum(), acc_total,
                                                                                      mask_pose_pos.sum(), acc_pos,
                                                                                      mask_pose_neg.sum(), acc_neg))
        worksheet.write(0, i, int(mask_pose.sum()))
        worksheet.write(1, i, float(acc_total))
        worksheet.write(2, i, int(mask_pose_pos.sum()))
        worksheet.write(3, i, float(acc_pos))
        worksheet.write(4, i, int(mask_pose_neg.sum()))
        worksheet.write(5, i, float(acc_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pose)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(pose)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(pose)), total, label='total sample', color='b',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(total):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)

    plt.xticks(np.arange(7), name)
    plt.ylabel('acc')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_pose_acc.png', dpi=200)
    plt.show()

    # statistic band
    total_list = []
    pos_list = []
    neg_list = []
    worksheet = workbook.add_sheet('acc_band')
    for i, band in enumerate(bands):
        mask_band = np.array(list(map(lambda x: get_band(x) == band,
                                      filenameLs)))
        mask_pos_band = np.logical_and(mask_pos, mask_band)
        mask_neg_band = np.logical_and(mask_neg, mask_band)
        total_band_list = []
        pos_band_list = []
        neg_band_list = []
        for j, (p1, p2) in enumerate(pose):
            mask_pose = get_pose_mask(p1, p2, filenameLs, filenameRs)
            mask_pose_band = np.logical_and(mask_pose, mask_band)
            mask_pose_pos_band = np.logical_and(mask_pose, mask_pos_band)
            mask_pose_neg_band = np.logical_and(mask_pose, mask_neg_band)
            # pred
            pred_pose_band = predictions[mask_pose_band]
            pred_pose_pos_band = predictions[mask_pose_pos_band]
            pred_pose_neg_band = predictions[mask_pose_neg_band]
            # acc
            acc_total = pred_pose_band.sum() / mask_pose_band.sum()
            acc_pos = pred_pose_pos_band.sum() / mask_pose_pos_band.sum()
            acc_neg = pred_pose_neg_band.sum() / mask_pose_neg_band.sum()
            # append
            total_band_list.append(acc_total)
            pos_band_list.append(acc_pos)
            neg_band_list.append(acc_neg)
            print('band {}, pose {}-{}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(band, p1, p2,
                                                                                                   mask_pose_band.sum(), acc_total,
                                                                                                   mask_pose_pos_band.sum(), acc_pos,
                                                                                                   mask_pose_neg_band.sum(), acc_neg,
                                                                                                   ))
            worksheet.write(0 + i * 7, j, int(mask_pose_band.sum()))
            worksheet.write(1 + i * 7, j, float(acc_total))
            worksheet.write(2 + i * 7, j, int(mask_pose_pos_band.sum()))
            worksheet.write(3 + i * 7, j, float(acc_pos))
            worksheet.write(4 + i * 7, j, int(mask_pose_neg_band.sum()))
            worksheet.write(5 + i * 7, j, float(acc_neg))
        total_list.append(total_band_list)
        pos_list.append(pos_band_list)
        neg_list.append(neg_band_list)
    total_list = np.array(total_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)

    xnew = np.arange(550, 990, 0.1)
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    plt.figure(figsize=(5, 5))
    for i in range(len(pose)):
        ynew = interpolate.interp1d(bands, pos_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Positive Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_pose_acc_pos_band.png', dpi=200)
    plt.show()

    plt.figure(figsize=(5, 5))
    for i in range(len(pose)):
        ynew = interpolate.interp1d(bands, neg_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Negative Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_pose_acc_neg_band.png', dpi=200)
    plt.show()

    plt.figure(figsize=(5, 5))
    for i in range(len(pose)):
        ynew = interpolate.interp1d(
            bands, total_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Total Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_pose_acc_total_band.png', dpi=200)
    plt.show()

    # write xls
    workbook.save('./face_verification_pose_acc.xls')


def pose_score_rgb_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]

    fold = [0, 1, 2, 3, 4]
    pose = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
    name = ['4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7']
    workbook = xlwt.Workbook(encoding='ascii')
    test_file = '../../workspace/MobileFacenet_split_6_exp_0/valid_result.mat'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    labels = result['labels'].squeeze()
    predictions = result['predictions'].squeeze()
    n_sample = len(filenameLs)
    image_idLs = np.array(list(map(get_id, filenameLs)))
    image_idRs = np.array(list(map(get_id, filenameRs)))
    image_bandLs = np.array(list(map(get_band, filenameLs)))
    image_bandRs = np.array(list(map(get_band, filenameRs)))
    # ---------------------------------------------------------
    # statistic pos-neg
    mask_pos = labels == 1
    mask_neg = labels == -1
    print('positive samples {}, negative samples {}'.format(mask_pos.sum(),
                                                            mask_neg.sum()))
    # statistic pose
    pos = []
    neg = []
    worksheet = workbook.add_sheet('score_rgb')
    # 4_*
    for i, (p1, p2) in enumerate(pose):
        mask_pose = get_pose_mask(p1, p2, filenameLs, filenameRs)
        mask_pose_pos = np.logical_and(mask_pose, mask_pos)
        mask_pose_neg = np.logical_and(mask_pose, mask_neg)
        scores_pos = scores[mask_pose_pos].mean()
        scores_neg = scores[mask_pose_neg].mean()
        pos.append(scores_pos)
        neg.append(scores_neg)
        print('pose {}-{}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(p1, p2, mask_pose.sum(),
                                                                              mask_pose_pos.sum(), scores_pos, mask_pose_neg.sum(), scores_neg))
        worksheet.write(0, i, int(mask_pose.sum()))
        worksheet.write(1, i, int(mask_pose_pos.sum()))
        worksheet.write(2, i, float(scores_pos))
        worksheet.write(3, i, int(mask_pose_neg.sum()))
        worksheet.write(4, i, float(scores_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(pos)), pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(neg)), neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(7), name)
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.ylabel('cosine')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('./face_verification_pose_score_rgb.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_pose_score_rgb.xls')


def pose_acc_rgb_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]

    fold = [0, 1, 2, 3, 4]
    pose = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
    name = ['4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7']
    workbook = xlwt.Workbook(encoding='ascii')
    test_file = '../../workspace/MobileFacenet_split_6_exp_0/valid_result.mat'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    labels = result['labels'].squeeze()
    predictions = result['predictions'].squeeze()
    n_sample = len(filenameLs)
    image_idLs = np.array(list(map(get_id, filenameLs)))
    image_idRs = np.array(list(map(get_id, filenameRs)))
    image_bandLs = np.array(list(map(get_band, filenameLs)))
    image_bandRs = np.array(list(map(get_band, filenameRs)))
    # ---------------------------------------------------------
    # statistic pos-neg
    mask_pos = labels == 1
    mask_neg = labels == -1
    print('positive samples {}, negative samples {}'.format(mask_pos.sum(),
                                                            mask_neg.sum()))
    # statistic pose
    pos = []
    neg = []
    total = []
    worksheet = workbook.add_sheet('acc')
    # 4_*
    for i, (p1, p2) in enumerate(pose):
        mask_pose = get_pose_mask(p1, p2, filenameLs, filenameRs)
        mask_pose_pos = np.logical_and(mask_pose, mask_pos)
        mask_pose_neg = np.logical_and(mask_pose, mask_neg)
        # pred
        pred_pose = predictions[mask_pose]
        pred_pose_pos = predictions[mask_pose_pos]
        pred_pose_neg = predictions[mask_pose_neg]
        # acc
        acc_total = pred_pose.sum() / mask_pose.sum()
        acc_pos = pred_pose_pos.sum() / mask_pose_pos.sum()
        acc_neg = pred_pose_neg.sum() / mask_pose_neg.sum()
        # append
        total.append(acc_total)
        pos.append(acc_pos)
        neg.append(acc_neg)
        print('pose {}-{}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(p1, p2,
                                                                                      mask_pose.sum(), acc_total,
                                                                                      mask_pose_pos.sum(), acc_pos,
                                                                                      mask_pose_neg.sum(), acc_neg))
        worksheet.write(0, i, int(mask_pose.sum()))
        worksheet.write(1, i, float(acc_total))
        worksheet.write(2, i, int(mask_pose_pos.sum()))
        worksheet.write(3, i, float(acc_pos))
        worksheet.write(4, i, int(mask_pose_neg.sum()))
        worksheet.write(5, i, float(acc_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pose)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(pose)), pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(pose)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(pose)), neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(pose)), total, label='total sample', color='b',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in zip(np.arange(len(pose)), total):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(7), name)
    plt.ylabel('acc')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_pose_acc_rgb.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_pose_acc_rgb.xls')


def main():
    pose_score_analysis()
    # pose_acc_analysis()

    # pose_score_rgb_analysis()
    # pose_acc_rgb_analysis()


if __name__ == '__main__':
    main()
