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


def noise_score_analysis():
    nsr = [0, 0.006, 0.008, 0.010, 0.012,
           0.014, 0.016, 0.018, 0.020, 0.04, 0.06]
    snr = [np.inf, 21.51, 20.43, 19.56, 18.84, 18.22,
           17.68, 17.20, 16.76, 13.86, 12.14]
    d = len(nsr)
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    workbook = xlwt.Workbook(encoding='ascii')
    name = ['snr={}'.format(x) for x in snr]

    pos_list = []
    neg_list = []
    total_list = []
    worksheet1 = workbook.add_sheet('score')
    worksheet2 = workbook.add_sheet('score_band')
    total = []
    pos = []
    neg = []
    for i in range(d):
        if nsr[i] == 0:
            subdir = '112x96'
        else:
            subdir = '112x96_noise_{}'.format(nsr[i])
        test_file = '../../workspace/{}/valid_result_fold_10.mat'.format(
            subdir)
        result = scipy.io.loadmat(test_file)
        filenameLs = result['filenameLs'].squeeze()
        filenameRs = result['filenameRs'].squeeze()
        featureLs = result['featureLs'].squeeze()
        featureRs = result['featureRs'].squeeze()
        labels = result['labels'].squeeze()
        scores = result['scores'].squeeze()
        predictions = result['predictions'].squeeze()

        n_sample = len(filenameLs)
        # image_idLs = np.array(list(map(get_id, filenameLs)))
        # image_idRs = np.array(list(map(get_id, filenameRs)))
        # image_bandLs = np.array(list(map(get_band, filenameLs)))
        # image_bandRs = np.array(list(map(get_band, filenameRs)))
        # ---------------------------------------------------------
        # statistic score
        mask_pos = labels == 1
        mask_neg = labels == -1
        scores_pos = scores[mask_pos].mean()
        scores_neg = scores[mask_neg].mean()
        pos.append(scores_pos)
        neg.append(scores_neg)
        print('snr {}: total {}| pos {}, {:.4f} | neg {}, {:.4f}'.format(snr[i],
                                                                         n_sample,
                                                                         mask_pos.sum(), scores_pos,
                                                                         mask_neg.sum(), scores_neg))
        worksheet1.write(0, i, int(n_sample))
        #worksheet1.write(1, i, float(acc_total))
        worksheet1.write(2, i, int(mask_pos.sum()))
        worksheet1.write(3, i, float(scores_pos))
        worksheet1.write(4, i, int(mask_neg.sum()))
        worksheet1.write(5, i, float(scores_neg))
        # ---------------------------------------------------------
        # statistic band
        pos_band_list = []
        neg_band_list = []
        for j, band in enumerate(bands):
            mask_band = np.array(
                list(map(lambda x: get_band(x) == band, filenameLs)))
            mask_pos_band = np.logical_and(mask_pos, mask_band)
            mask_neg_band = np.logical_and(mask_neg, mask_band)
            scores_pos_band = scores[mask_pos_band].mean()
            scores_neg_band = scores[mask_neg_band].mean()
            pos_band_list.append(scores_pos_band)
            neg_band_list.append(scores_neg_band)
            print('band {}, snr {}: total {} | pos {}, {:.4f} | neg {}, {:.4f}'.format(band, snr[i], mask_band.sum(),
                                                                                       mask_pos_band.sum(), scores_pos_band,
                                                                                       mask_neg_band.sum(), scores_neg_band,
                                                                                       ))
            worksheet2.write(0 + i * 6, j, int(mask_band.sum()))
            worksheet2.write(1 + i * 6, j, int(mask_pos_band.sum()))
            worksheet2.write(2 + i * 6, j, float(scores_pos_band))
            worksheet2.write(3 + i * 6, j, int(mask_neg_band.sum()))
            worksheet2.write(4 + i * 6, j, float(scores_neg_band))
        pos_list.append(pos_band_list)
        neg_list.append(neg_band_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    # plot
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c',
              'coral', 'darkred', 'cyan', 'purple', 'pink']
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(d), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(d), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(d), snr)
    plt.xlabel('snr')
    plt.ylabel('cosine')
    plt.title('Face Verification, Noise Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='small')
    plt.savefig('./face_verification_noise_score.png', dpi=200)
    plt.show()

    # plot band
    xnew = np.arange(550, 990, 0.1)
    for i in range(d):
        ynew = interpolate.interp1d(bands, pos_list[i, :], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
        ynew = interpolate.interp1d(bands, neg_list[i, :], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=name[i] + '(neg)')
    plt.xlabel('wavelength')
    plt.ylabel('cosine')
    #plt.yticks(np.arange(0, 0.7, 0.05))
    plt.title('Face Verification, Noise Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='x-small')
    plt.savefig('./face_verification_noise_score_band.png', dpi=200)
    plt.show()

    workbook.save('./face_verification_noise_score.xls')


def noise_acc_analysis():
    nsr = [0, 0.006, 0.008, 0.010, 0.012,
           0.014, 0.016, 0.018, 0.020, 0.04, 0.06]
    snr = [np.inf, 21.51, 20.43, 19.56, 18.84, 18.22,
           17.68, 17.20, 16.76, 13.86, 12.14]
    d = len(snr)
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    workbook = xlwt.Workbook(encoding='ascii')
    name = ['snr={}'.format(x) for x in snr]

    pos_list = []
    neg_list = []
    total_list = []
    worksheet1 = workbook.add_sheet('acc')
    worksheet2 = workbook.add_sheet('acc_band')
    total = []
    pos = []
    neg = []
    for i in range(d):
        if nsr[i] == 0:
            subdir = '112x96'
        else:
            subdir = '112x96_noise_{}'.format(nsr[i])
        test_file = '../../workspace/{}/valid_result_fold_10.mat'.format(
            subdir)
        result = scipy.io.loadmat(test_file)
        filenameLs = result['filenameLs'].squeeze()
        filenameRs = result['filenameRs'].squeeze()
        featureLs = result['featureLs'].squeeze()
        featureRs = result['featureRs'].squeeze()
        labels = result['labels'].squeeze()
        scores = result['scores'].squeeze()
        predictions = result['predictions'].squeeze()

        n_sample = len(filenameLs)
        # image_idLs = np.array(list(map(get_id, filenameLs)))
        # image_idRs = np.array(list(map(get_id, filenameRs)))
        # image_bandLs = np.array(list(map(get_band, filenameLs)))
        # image_bandRs = np.array(list(map(get_band, filenameRs)))
        # ---------------------------------------------------------
        # statistic score
        mask_pos = labels == 1
        mask_neg = labels == -1
        acc_pos = predictions[mask_pos].sum() / mask_pos.sum()
        acc_neg = predictions[mask_neg].sum() / mask_neg.sum()
        acc_total = predictions.mean()
        pos.append(acc_pos)
        neg.append(acc_neg)
        total.append(acc_total)
        print('snr {}: total {}, {:.4f}| pos {}, {:.4f} | neg {}, {:.4f}'.format(snr[i],
                                                                                 n_sample, acc_total,
                                                                                 mask_pos.sum(), acc_pos,
                                                                                 mask_neg.sum(), acc_neg))
        worksheet1.write(0, i, int(n_sample))
        worksheet1.write(1, i, float(acc_total))
        worksheet1.write(2, i, int(mask_pos.sum()))
        worksheet1.write(3, i, float(acc_pos))
        worksheet1.write(4, i, int(mask_neg.sum()))
        worksheet1.write(5, i, float(acc_neg))
        # ---------------------------------------------------------
        # statistic band
        pos_band_list = []
        neg_band_list = []
        total_band_list = []
        for j, band in enumerate(bands):
            mask_band = np.array(
                list(map(lambda x: get_band(x) == band, filenameLs)))
            mask_pos_band = np.logical_and(mask_pos, mask_band)
            mask_neg_band = np.logical_and(mask_neg, mask_band)
            pred_pos_band = predictions[mask_pos_band].sum(
            ) / mask_pos_band.sum()
            pred_neg_band = predictions[mask_neg_band].sum(
            ) / mask_neg_band.sum()
            pred_total_band = predictions[mask_band].sum() / mask_band.sum()
            pos_band_list.append(pred_pos_band)
            neg_band_list.append(pred_neg_band)
            total_band_list.append(pred_total_band)
            print('band {}, snr {}: total {}, {:.4f} | pos {}, {:.4f} | neg {}, {:.4f}'.format(band, snr[i],
                                                                                               mask_band.sum(), pred_total_band,
                                                                                               mask_pos_band.sum(), pred_pos_band,
                                                                                               mask_neg_band.sum(), pred_neg_band,
                                                                                               ))
            worksheet2.write(0 + i * 6, j, int(mask_band.sum()))
            worksheet2.write(1 + i * 6, j, float(pred_total_band))
            worksheet2.write(2 + i * 6, j, int(mask_pos_band.sum()))
            worksheet2.write(3 + i * 6, j, float(pred_pos_band))
            worksheet2.write(4 + i * 6, j, int(mask_neg_band.sum()))
            worksheet2.write(5 + i * 6, j, float(pred_neg_band))
        pos_list.append(pos_band_list)
        neg_list.append(neg_band_list)
        total_list.append(total_band_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    total_list = np.array(total_list)
    # plot
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c',
              'coral', 'darkred', 'cyan', 'purple', 'pink']
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(d), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(d), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(d), snr)
    plt.xlabel('snr')
    plt.ylabel('acc')
    plt.title('Face Verification, Noise Acc Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig('./face_verification_noise_acc.png', dpi=200)
    plt.show()

    # plot band
    xnew = np.arange(550, 990, 0.1)
    for i in range(d):
        ynew = interpolate.interp1d(bands, pos_list[i, :], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.', linewidth=1, label=name[i])
        ynew = interpolate.interp1d(bands, neg_list[i, :], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=name[i] + '(neg)')
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    #plt.yticks(np.arange(0, 0.7, 0.05))
    plt.title('Face Verification, Noise Acc Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='x-small')
    plt.savefig('./face_verification_noise_acc_band.png', dpi=200)
    plt.show()

    workbook.save('./face_verification_noise_acc.xls')


def main():
    noise_score_analysis()
    # noise_acc_analysis()


if __name__ == '__main__':
    main()
