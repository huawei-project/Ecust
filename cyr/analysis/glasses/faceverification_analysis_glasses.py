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


def plot_embedding(X, y, ax, title=None):
    # maps = [m for m in plt.cm.datad if not m.endswith("_r")]
    # colors = []
    # for m in maps:
    #     if isinstance(plt.cm.datad[m], tuple):
    #         colors += plt.cm.datad[m]
    colors = [plt.cm.tab10(i) for i in range(plt.cm.tab10.N)]
    colors += [plt.cm.Set2(i) for i in range(plt.cm.Set2.N)]
    colors += [plt.cm.Set3(i) for i in range(plt.cm.Set3.N)]
    colors += [plt.cm.Accent(i) for i in range(plt.cm.Accent.N)]
    colors += [plt.cm.Dark2(i) for i in range(plt.cm.Dark2.N)]
    colors += [plt.cm.Paired(i) for i in range(plt.cm.Paired.N)]
    colors += [plt.cm.Pastel1(i) for i in range(plt.cm.Pastel1.N)]
    colors += [plt.cm.Pastel2(i) for i in range(plt.cm.Pastel2.N)]
    np.random.shuffle(colors)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # ax = plt.subplot(1, 2, 2)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=colors[y[i]],
                 fontdict={'weight': 'bold', 'size': 6})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title, fontdict={'fontsize': 10})


def glasses_score_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    glasses = [1, 2, 3, 4]
    workbook = xlwt.Workbook(encoding='ascii')
    fold = [0, 1, 2, 3, 4]
    name = ['glasses', 'sunglasses', 'nonglasses', 'total']
    test_file = '../../workspace/split_11_valid_result_10_fold'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    predictions = result['predictions'].squeeze()
    labels = result['labels'].squeeze()
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
    for i, g in enumerate(glasses):
        if g == 4:
            mask_glasses_pos = mask_pos
            mask_glasses_neg = mask_neg
        else:
            mask_glasses = get_glasses_mask(g, filenameLs, filenameRs)
            mask_glasses_pos = np.logical_and(mask_glasses, mask_pos)
            mask_glasses_neg = np.logical_and(mask_glasses, mask_neg)
        scores_pos = scores[mask_glasses_pos].mean()
        scores_neg = scores[mask_glasses_neg].mean()
        pos.append(scores_pos)
        neg.append(scores_neg)
        print('glasses {}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(g, mask_glasses.sum(),
                                                                              mask_glasses_pos.sum(), scores_pos,
                                                                              mask_glasses_neg.sum(), scores_neg))
        worksheet.write(0, i, int(mask_glasses.sum()))
        worksheet.write(1, i, int(mask_glasses_pos.sum()))
        worksheet.write(2, i, float(scores_pos))
        worksheet.write(3, i, int(mask_glasses_neg.sum()))
        worksheet.write(4, i, float(scores_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(glasses)), name)
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.ylabel('cosine')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper left', fontsize='x-small')
    plt.savefig('./face_verification_glasses_score.png', dpi=200)
    plt.show()
    # statistic band
    pos_list = []
    neg_list = []
    worksheet = workbook.add_sheet('score_band')
    for i, band in enumerate(bands):
        mask_band = np.array(
            list(map(lambda x: get_band(x) == band, filenameLs)))
        mask_pos_band = np.logical_and(mask_pos, mask_band)
        mask_neg_band = np.logical_and(mask_neg, mask_band)
        pos_band_list = []
        neg_band_list = []
        for j, g in enumerate(glasses):
            if g == 4:
                mask_glasses_pos_band = mask_pos_band
                mask_glasses_neg_band = mask_neg_band
            else:
                mask_glasses = get_glasses_mask(g, filenameLs, filenameRs)
                mask_glasses_band = np.logical_and(mask_glasses, mask_band)
                mask_glasses_pos_band = np.logical_and(
                    mask_glasses, mask_pos_band)
                mask_glasses_neg_band = np.logical_and(
                    mask_glasses, mask_neg_band)
            scores_pos_band = scores[mask_glasses_pos_band].mean()
            scores_neg_band = scores[mask_glasses_neg_band].mean()
            pos_band_list.append(scores_pos_band)
            neg_band_list.append(scores_neg_band)
            print('band {}, glasses {}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(band, g, mask_glasses_band.sum(),
                                                                                           mask_glasses_pos_band.sum(), scores_pos_band,
                                                                                           mask_glasses_neg_band.sum(), scores_neg_band,
                                                                                           ))
            worksheet.write(0 + i * 6, j, int(mask_glasses_band.sum()))
            worksheet.write(1 + i * 6, j, int(mask_glasses_pos_band.sum()))
            worksheet.write(2 + i * 6, j, float(scores_pos_band))
            worksheet.write(3 + i * 6, j, int(mask_glasses_neg_band.sum()))
            worksheet.write(4 + i * 6, j, float(scores_neg_band))
        pos_list.append(pos_band_list)
        neg_list.append(neg_band_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    # plot
    plt.figure(figsize=(5, 5))
    xnew = np.arange(550, 990, 0.1)
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    for i in range(len(glasses)):
        ynew = interpolate.interp1d(bands, pos_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
        ynew = interpolate.interp1d(bands, neg_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=name[i] + '_neg')
    plt.xlabel('wavelength')
    plt.ylabel('cosine')
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='x-small')
    plt.savefig('./face_verification_glasses_score_band.png', dpi=200)
    plt.show()

    # plt.figure(figsize=(5, 5))
    # for i in range(len(glasses)):
    #     ynew = interpolate.interp1d(bands, pos_list[:, i], kind='cubic')(xnew)
    #     plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    # plt.xlabel('wavelength')
    # plt.ylabel('cosine')
    # plt.title('Face Verification, Feature Score Analysis',
    #           fontdict={'fontsize': 10})
    # plt.legend(loc='center right', fontsize='x-small')
    # plt.savefig('./face_verification_glasses_score_pos_band_nofintune.png', dpi=200)
    # plt.show()
    # write xls
    workbook.save('./face_verification_glasses_score.xls')


def glasses_acc_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    glasses = [1, 2, 3]
    workbook = xlwt.Workbook(encoding='ascii')
    fold = [0, 1, 2, 3, 4]
    name = ['glasses', 'sunglasses', 'nonglasses']
    test_file = '../../workspace/split_11_valid_result_10_fold'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    predictions = result['predictions'].squeeze()
    labels = result['labels'].squeeze()
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
    # statistic glasses
    total = []
    pos = []
    neg = []
    worksheet = workbook.add_sheet('acc')
    for i, g in enumerate(glasses):
        mask_glasses = get_glasses_mask(g, filenameLs, filenameRs)
        mask_glasses_pos = np.logical_and(mask_glasses, mask_pos)
        mask_glasses_neg = np.logical_and(mask_glasses, mask_neg)
        # pred
        pred_glasses = predictions[mask_glasses]
        pred_glasses_pos = predictions[mask_glasses_pos]
        pred_glasses_neg = predictions[mask_glasses_neg]
        # acc
        acc_total = pred_glasses.sum() / mask_glasses.sum()
        acc_pos = pred_glasses_pos.sum() / mask_glasses_pos.sum()
        acc_neg = pred_glasses_neg.sum() / mask_glasses_neg.sum()
        # append
        total.append(acc_total)
        pos.append(acc_pos)
        neg.append(acc_neg)
        print('glasses {}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(g,
                                                                                      mask_glasses.sum(), acc_total,
                                                                                      mask_glasses_pos.sum(), acc_pos,
                                                                                      mask_glasses_neg.sum(), acc_neg))
        worksheet.write(0, i, int(mask_glasses.sum()))
        worksheet.write(1, i, float(acc_total.sum()))
        worksheet.write(2, i, int(mask_glasses_pos.sum()))
        worksheet.write(3, i, float(acc_pos))
        worksheet.write(4, i, int(mask_glasses_neg.sum()))
        worksheet.write(5, i, float(acc_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(total)), total, label='total sample', color='b',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(total):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(glasses)), name)
    plt.ylabel('acc')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_glasses_acc.png', dpi=200)
    plt.show()
    # statistic band
    total_list = []
    pos_list = []
    neg_list = []
    worksheet = workbook.add_sheet('acc_band')
    for i, band in enumerate(bands):
        mask_band = np.array(
            list(map(lambda x: get_band(x) == band, filenameLs)))
        mask_pos_band = np.logical_and(mask_pos, mask_band)
        mask_neg_band = np.logical_and(mask_neg, mask_band)
        total_band_list = []
        pos_band_list = []
        neg_band_list = []
        for j, g in enumerate(glasses):
            mask_glasses = get_glasses_mask(g, filenameLs, filenameRs)
            mask_glasses_band = np.logical_and(mask_glasses, mask_band)
            mask_glasses_pos_band = np.logical_and(
                mask_glasses, mask_pos_band)
            mask_glasses_neg_band = np.logical_and(
                mask_glasses, mask_neg_band)
            # pred
            pred_glasses_band = predictions[mask_glasses_band]
            pred_glasses_pos_band = predictions[mask_glasses_pos_band]
            pred_glasses_neg_band = predictions[mask_glasses_neg_band]
            # acc
            acc_total = pred_glasses_band.sum() / mask_glasses_band.sum()
            acc_pos = pred_glasses_pos_band.sum() / mask_glasses_pos_band.sum()
            acc_neg = pred_glasses_neg_band.sum() / mask_glasses_neg_band.sum()
            # append
            total_band_list.append(acc_total)
            pos_band_list.append(acc_pos)
            neg_band_list.append(acc_neg)
            print('band {}, glasses {}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(band, g,
                                                                                                   mask_glasses_band.sum(), acc_total,
                                                                                                   pred_glasses_pos_band.sum(), acc_pos,
                                                                                                   pred_glasses_neg_band.sum(), acc_neg,
                                                                                                   ))
            worksheet.write(0 + i * 7, j, int(mask_glasses_band.sum()))
            worksheet.write(1 + i * 7, j, float(acc_total))
            worksheet.write(2 + i * 7, j, int(pred_glasses_pos_band.sum()))
            worksheet.write(3 + i * 7, j, float(acc_pos))
            worksheet.write(4 + i * 7, j, int(pred_glasses_neg_band.sum()))
            worksheet.write(5 + i * 7, j, float(acc_neg))
        total_list.append(total_band_list)
        pos_list.append(pos_band_list)
        neg_list.append(neg_band_list)
    total_list = np.array(total_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    # plot
    plt.figure(figsize=(5, 5))
    xnew = np.arange(550, 990, 0.1)
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    for i in range(len(glasses)):
        ynew = interpolate.interp1d(bands, pos_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Positive Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_glasses_acc_pos_band.png', dpi=200)
    plt.show()

    plt.figure(figsize=(5, 5))
    for i in range(len(glasses)):
        ynew = interpolate.interp1d(bands, neg_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Negative Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_glasses_acc_neg_band.png', dpi=200)
    plt.show()

    print(total_list.shape)
    plt.figure(figsize=(5, 5))
    for i in range(len(glasses)):
        ynew = interpolate.interp1d(
            bands, total_list[:, i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Total Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_glasses_acc_total_band.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_glasses_acc.xls')


def glasses_score_rgb_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    glasses = [1, 2, 3, 4]
    workbook = xlwt.Workbook(encoding='ascii')
    fold = [0, 1, 2, 3, 4]
    name = ['glasses', 'sunglasses', 'nonglasses', 'total']
    test_file = '../../workspace/MobileFacenet_split_6_exp_0/valid_result.mat'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    predictions = result['predictions'].squeeze()
    labels = result['labels'].squeeze()
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
    for i, g in enumerate(glasses):
        if g == 4:
            mask_glasses_pos = mask_pos
            mask_glasses_neg = mask_neg
        else:
            mask_glasses = get_glasses_mask(g, filenameLs, filenameRs)
            mask_glasses_pos = np.logical_and(mask_glasses, mask_pos)
            mask_glasses_neg = np.logical_and(mask_glasses, mask_neg)
        scores_pos = scores[mask_glasses_pos].mean()
        scores_neg = scores[mask_glasses_neg].mean()
        pos.append(scores_pos)
        neg.append(scores_neg)
        print('glasses {}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(g, mask_glasses.sum(),
                                                                              mask_glasses_pos.sum(), scores_pos,
                                                                              mask_glasses_neg.sum(), scores_neg))
        worksheet.write(0, i, int(mask_glasses.sum()))
        worksheet.write(1, i, int(mask_glasses_pos.sum()))
        worksheet.write(2, i, float(scores_pos))
        worksheet.write(3, i, int(mask_glasses_neg.sum()))
        worksheet.write(4, i, float(scores_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(glasses)), name)
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.ylabel('cosine')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper left', fontsize='x-small')
    plt.savefig('./face_verification_glasses_score_rgb.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_glasses_score_rgb.xls')


def glasses_acc_rgb_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    glasses = [1, 2, 3]
    workbook = xlwt.Workbook(encoding='ascii')
    fold = [0, 1, 2, 3, 4]
    name = ['glasses', 'sunglasses', 'nonglasses']
    test_file = '../../workspace/MobileFacenet_split_6_exp_0/valid_result.mat'
    result = scipy.io.loadmat(test_file)
    filenameLs = result['filenameLs'].squeeze()
    filenameRs = result['filenameRs'].squeeze()
    featureLs = result['featureLs'].squeeze()
    featureRs = result['featureRs'].squeeze()
    scores = result['scores'].squeeze()
    predictions = result['predictions'].squeeze()
    labels = result['labels'].squeeze()
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
    # statistic glasses
    total = []
    pos = []
    neg = []
    worksheet = workbook.add_sheet('score_rgb')
    for i, g in enumerate(glasses):
        mask_glasses = get_glasses_mask(g, filenameLs, filenameRs)
        mask_glasses_pos = np.logical_and(mask_glasses, mask_pos)
        mask_glasses_neg = np.logical_and(mask_glasses, mask_neg)
        # pred
        pred_glasses = predictions[mask_glasses]
        pred_glasses_pos = predictions[mask_glasses_pos]
        pred_glasses_neg = predictions[mask_glasses_neg]
        # acc
        acc_total = pred_glasses.sum() / mask_glasses.sum()
        acc_pos = pred_glasses_pos.sum() / mask_glasses_pos.sum()
        acc_neg = pred_glasses_neg.sum() / mask_glasses_neg.sum()
        # append
        total.append(acc_total)
        pos.append(acc_pos)
        neg.append(acc_neg)
        print('glasses {}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(g,
                                                                                      mask_glasses.sum(), acc_total,
                                                                                      mask_glasses_pos.sum(), acc_pos,
                                                                                      mask_glasses_neg.sum(), acc_neg))
        worksheet.write(0, i, int(mask_glasses.sum()))
        worksheet.write(1, i, float(acc_total.sum()))
        worksheet.write(2, i, int(mask_glasses_pos.sum()))
        worksheet.write(3, i, float(acc_pos))
        worksheet.write(4, i, int(mask_glasses_neg.sum()))
        worksheet.write(5, i, float(acc_neg))
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(total)), total, label='total sample', color='b',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(total):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(glasses)), name)
    plt.ylabel('acc')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_glasses_acc_rgb.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_glasses_acc_rgb.xls')


def main():
    glasses_score_analysis()
    # glasses_acc_analysis()
    # glasses_score_rgb_analysis()
    # glasses_acc_rgb_analysis()


if __name__ == '__main__':
    main()