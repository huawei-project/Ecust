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


def illumination_score_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    split = [1, 2, 3, 4, 5, 6, 7]
    workbook = xlwt.Workbook(encoding='ascii')
    data_name = ['0%', '30%', '50%', '70%', 'illum', 'rgb', 'rgb_illum']
    pos_list = []
    neg_list = []
    pos_band_list = []
    neg_band_list = []
    worksheet1 = workbook.add_sheet('score')
    worksheet2 = workbook.add_sheet('score_band')
    for i in range(len(split)):
        test_file = '../../workspace/MobileFacenet_split_{}_exp_{}/valid_result.mat'.format(
            split[i], 0)
        result = scipy.io.loadmat(test_file)
        filenameLs = result['filenameLs'].squeeze()
        filenameRs = result['filenameRs'].squeeze()
        featureLs = result['featureLs'].squeeze()
        featureRs = result['featureRs'].squeeze()
        scores = result['scores'].squeeze()
        Thresholds = result['Thresholds'].squeeze()
        predictions = result['predictions'].squeeze()
        labels = result['labels'].squeeze()
        n_sample = len(filenameLs)
        mask_pos = labels == 1
        mask_neg = labels == -1
        scores_pos = scores[mask_pos].mean()
        scores_neg = scores[mask_neg].mean()
        pos_list.append(scores_pos)
        neg_list.append(scores_neg)
        print('split {}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(i, n_sample,
                                                                            mask_pos.sum(), scores_pos,
                                                                            mask_neg.sum(), scores_neg))
        worksheet1.write(0, i, int(n_sample))
        worksheet1.write(1, i, int(mask_pos.sum()))
        worksheet1.write(2, i, float(scores_pos))
        worksheet1.write(3, i, int(mask_neg.sum()))
        worksheet1.write(4, i, float(scores_neg))
        # ------------------
        # statistic band
        pos_band = []
        neg_band = []
        if i < 5:
            for j, band in enumerate(bands):
                mask_band = np.array(
                    list(map(lambda x: get_band(x) == band, filenameLs)))
                mask_pos_band = np.logical_and(mask_pos, mask_band)
                mask_neg_band = np.logical_and(mask_neg, mask_band)
                scores_pos_band = scores[mask_pos_band].mean()
                scores_neg_band = scores[mask_neg_band].mean()
                print('band {}, split {}: total {} | pos {}, {:.2f} | neg {}, {:.2f}'.format(band, i, mask_band.sum(),
                                                                                             mask_pos_band.sum(), scores_pos_band,
                                                                                             mask_neg_band.sum(), scores_neg_band,
                                                                                             ))
                worksheet2.write(0 + i * 6, j, int(mask_band.sum()))
                worksheet2.write(1 + i * 6, j, int(mask_pos_band.sum()))
                worksheet2.write(2 + i * 6, j, float(scores_pos_band))
                worksheet2.write(3 + i * 6, j, int(mask_neg_band.sum()))
                worksheet2.write(4 + i * 6, j, float(scores_neg_band))
                pos_band.append(scores_pos_band)
                neg_band.append(scores_neg_band)
        pos_band_list.append(pos_band)
        neg_band_list.append(neg_band)
    # plot
    plt.plot(np.arange(len(pos_list)), pos_list, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos_list):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg_list)), neg_list, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg_list):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(data_name)), data_name)
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.ylabel('cosine')
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper left', fontsize='x-small')
    plt.savefig('./face_verification_illumination_score.png', dpi=200)
    plt.show()
    # -----------
    plt.figure(figsize=(5, 5))
    xnew = np.arange(550, 990, 0.1)
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    for i in range(5):
        ynew = interpolate.interp1d(
            bands, pos_band_list[i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=data_name[i])
        ynew = interpolate.interp1d(
            bands, neg_band_list[i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=data_name[i]+'_neg')
    plt.xlabel('wavelength')
    plt.ylabel('cosine')
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.title('Face Verification, Feature Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='x-small')
    plt.savefig('./face_verification_illumination_score_band.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_illumination_score.xls')


def illumination_acc_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    split = [1, 2, 3, 4, 5, 6, 7]
    workbook = xlwt.Workbook(encoding='ascii')
    data_name = ['0%', '30%', '50%', '70%', 'illum', 'rgb', 'rgb_illum']
    pos_list = []
    neg_list = []
    total_list = []
    pos_band_list = []
    neg_band_list = []
    total_band_list = []
    worksheet1 = workbook.add_sheet('acc')
    worksheet2 = workbook.add_sheet('acc_band')
    for i in range(len(split)):
        test_file = '../../workspace/MobileFacenet_split_{}_exp_{}/valid_result.mat'.format(
            split[i], 0)
        result = scipy.io.loadmat(test_file)
        filenameLs = result['filenameLs'].squeeze()
        filenameRs = result['filenameRs'].squeeze()
        featureLs = result['featureLs'].squeeze()
        featureRs = result['featureRs'].squeeze()
        scores = result['scores'].squeeze()
        Thresholds = result['Thresholds'].squeeze()
        predictions = result['predictions'].squeeze()
        labels = result['labels'].squeeze()
        n_sample = len(filenameLs)
        # image_idLs = np.array(list(map(get_id, filenameLs)))
        # image_idRs = np.array(list(map(get_id, filenameRs)))
        # image_bandLs = np.array(list(map(get_band, filenameLs)))
        # image_bandRs = np.array(list(map(get_band, filenameRs)))
        # acc
        mask_pos = labels == 1
        mask_neg = labels == -1
        pred_pos = predictions[mask_pos].sum() / (mask_pos).sum()
        pred_neg = predictions[mask_neg].sum() / (mask_neg).sum()
        pred_total = predictions.mean()
        pos_list.append(pred_pos)
        neg_list.append(pred_neg)
        total_list.append(pred_total)
        print('split {}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(i,
                                                                                    n_sample, pred_total,
                                                                                    mask_pos.sum(), pred_pos,
                                                                                    mask_neg.sum(), pred_neg))
        worksheet1.write(0, i, int(n_sample))
        worksheet1.write(1, i, float(pred_total))
        worksheet1.write(2, i, int(mask_pos.sum()))
        worksheet1.write(3, i, float(pred_pos))
        worksheet1.write(4, i, int(mask_neg.sum()))
        worksheet1.write(5, i, float(pred_neg))
        # ------------------
        # statistic band
        pos_band = []
        neg_band = []
        total_band = []
        if i < 5:
            for j, band in enumerate(bands):
                mask_band = np.array(
                    list(map(lambda x: get_band(x) == band, filenameLs)))
                mask_pos_band = np.logical_and(mask_pos, mask_band)
                mask_neg_band = np.logical_and(mask_neg, mask_band)
                pred_pos = predictions[mask_pos_band]
                pred_neg = predictions[mask_neg_band]
                pred_total = predictions[mask_band]

                acc_pos_band = pred_pos.sum() / mask_pos_band.sum()
                acc_neg_band = pred_neg.sum() / mask_neg_band.sum()
                acc_total_band = pred_total.sum() / mask_band.sum()
                pos_band.append(acc_pos_band)
                neg_band.append(acc_neg_band)
                total_band.append(acc_total_band)
                print('band {}, split {}: total {}, {:.2f} | pos {}, {:.2f} | neg {}, {:.2f}'.format(band, i,
                                                                                                     mask_band.sum(), acc_total_band,
                                                                                                     mask_pos_band.sum(), acc_pos_band,
                                                                                                     mask_neg_band.sum(), acc_neg_band,
                                                                                                     ))
                worksheet2.write(0 + i * 7, j, int(mask_band.sum()))
                worksheet2.write(1 + i * 7, j, float(acc_total_band))
                worksheet2.write(2 + i * 7, j, int(mask_pos_band.sum()))
                worksheet2.write(3 + i * 7, j, float(acc_pos_band))
                worksheet2.write(4 + i * 7, j, int(mask_neg_band.sum()))
                worksheet2.write(5 + i * 7, j, float(acc_neg_band))
        pos_band_list.append(pos_band)
        neg_band_list.append(neg_band)
        total_band_list.append(total_band)
    # plot
    plt.figure(figsize=(5, 5))
    x = np.arange(len(data_name), dtype=np.float) * 2
    total_width, n = 1.5, 3
    width = total_width / n
    plt.bar(x, total_list, width=width,
            label='total', tick_label=data_name, fc='y')
    for a, b in zip(x, total_list):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=8)
    x += width
    plt.bar(x, pos_list, width=width, label='positive', fc='r')
    for a, b in zip(x, pos_list):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=8)
    x += width
    plt.bar(x, neg_list, width=width, label='negatve', fc='g')
    for a, b in zip(x, neg_list):
        plt.text(a, b, '{:.3f}'.format(b),
                 ha='center', va='bottom', fontsize=8)
    plt.ylabel('acc')
    plt.title('Face Verification Analysis', fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('./face_verification_illumination_acc.png', dpi=200)
    plt.show()
    # --------------------------------
    colors = ['g', 'y', 'k', 'c', 'r', 'b', 'c', 'm']
    xnew = np.arange(550, 990, 0.1)

    plt.figure(figsize=(5, 5))
    for i in range(5):
        ynew = interpolate.interp1d(
            bands, total_band_list[i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=data_name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Total Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_illumination_acc_total_band.png', dpi=200)
    plt.show()

    plt.figure(figsize=(5, 5))
    for i in range(5):
        ynew = interpolate.interp1d(
            bands, pos_band_list[i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=data_name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Positive Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_illumination_acc_pos_band.png', dpi=200)
    plt.show()

    plt.figure(figsize=(5, 5))
    for i in range(5):
        ynew = interpolate.interp1d(
            bands, neg_band_list[i], kind='cubic')(xnew)
        plt.plot(xnew, ynew, colors[i], marker='.',
                 linewidth=1, label=data_name[i])
    plt.xlabel('wavelength')
    plt.ylabel('acc')
    plt.title('Face Verification, Negative Sample Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./face_verification_illumination_acc_neg_band.png', dpi=200)
    plt.show()
    # write xls
    workbook.save('./face_verification_illumination_acc.xls')


def illumination_tsne_analysis():
    num_band = 23
    bands = [550 + 20 * i for i in range(num_band)]
    split = [1, 2, 3, 4, 5, 6, 7]
    data_name = ['0%', '30%', '50%', '70%', 'illum', 'rgb', 'rgb_illum']
    acc_pos_list = []
    acc_neg_list = []
    acc_total_list = []
    acc_pos_band_list = []
    acc_neg_band_list = []
    acc_total_band_list = []

    H, W = 2, 4
    for i in range(len(split)):
        test_file = '../../workspace/MobileFacenet_split_{}_exp_{}/valid_result.mat'.format(
            split[i], 0)
        result = scipy.io.loadmat(test_file)
        filenameLs = result['filenameLs'].squeeze()
        filenameRs = result['filenameRs'].squeeze()
        featureLs = result['featureLs'].squeeze()
        featureRs = result['featureRs'].squeeze()
        scores = result['scores'].squeeze()
        Thresholds = result['Thresholds'].squeeze()
        predictions = result['predictions'].squeeze()
        labels = result['labels'].squeeze()
        n_sample = len(filenameLs)
        image_idLs = np.array(list(map(get_id, filenameLs)))
        image_idRs = np.array(list(map(get_id, filenameRs)))
        # image_bandLs = np.array(list(map(get_band, filenameLs)))
        # image_bandRs = np.array(list(map(get_band, filenameRs)))
        # ----------------------------
        # statistic feature statibility
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X = np.concatenate((featureLs, featureRs), 0)
        y = np.concatenate((image_idLs, image_idRs), 0)
        n_id = 15
        n = sorted(np.unique(y))[n_id - 1]
        X = X[y <= n]
        y = y[y <= n]
        X_tsne = tsne.fit_transform(X)
        ax = plt.figure(figsize=(5, 5))
        plot_embedding(X_tsne, y, ax=ax,
                       title='t-SNE dimension reduction({}, dim=2, n_id={})'.format(data_name[i], n_id))
        plt.savefig(
            './face_verification_illumination_tsne_{}.png'.format(data_name[i]), dpi=200)


def main():
    illumination_score_analysis()
    illumination_acc_analysis()
    illumination_tsne_analysis()


if __name__ == '__main__':
    main()
