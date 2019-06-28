import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import xlwt
from sklearn import manifold, decomposition
from scipy import interpolate
from sklearn.metrics import roc_curve, roc_auc_score
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


def spectrum_score_analysis():
    bands1 = np.arange(550, 991, 20)
    bands2 = np.arange(550, 991, 40)
    bands3 = np.arange(550, 991, 60)
    bands4 = np.arange(550, 991, 80)
    bands5 = np.arange(550, 991, 100)
    choice = [bands1, bands2, bands3, bands4]
    num_c = [23, 12, 8, 6]
    d = len(choice)
    workbook = xlwt.Workbook(encoding='ascii')
    name = ['20', '40', '60', '80']

    pos_list = []
    neg_list = []
    total_list = []
    worksheet1 = workbook.add_sheet('score')
    total = []
    pos = []
    neg = []
    for i in range(d):
        bands = choice[i]
        subdir = '112x96_{}'.format(num_c[i])
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
        print('c {}: total {}| pos {}, {:.4f} | neg {}, {:.4f}'.format(num_c[i],
                                                                       n_sample,
                                                                       mask_pos.sum(), scores_pos,
                                                                       mask_neg.sum(), scores_neg))
        worksheet1.write(0, i, int(n_sample))
        #worksheet1.write(1, i, float(acc_total))
        worksheet1.write(2, i, int(mask_pos.sum()))
        worksheet1.write(3, i, float(scores_pos))
        worksheet1.write(4, i, int(mask_neg.sum()))
        worksheet1.write(5, i, float(scores_neg))

    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    # plot
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    plt.figure(figsize=(5, 5))
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
    plt.xticks(np.arange(d), name)
    plt.xlabel('spectrum resolution')
    plt.ylabel('cosine')
    plt.title('Face Verification, Spectrum Score Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='small')
    #plt.savefig('./face_verification_spectrum_score.png', dpi=200)
    plt.show()

    workbook.save('./face_verification_spectrum_score.xls')


def spectrum_acc_analysis():
    bands1 = np.arange(550, 991, 20)
    bands2 = np.arange(550, 991, 40)
    bands3 = np.arange(550, 991, 60)
    bands4 = np.arange(550, 991, 80)
    bands5 = np.arange(550, 991, 100)
    choice = [bands1, bands2, bands3, bands4]
    num_c = [23, 12, 8, 6]
    d = len(choice)
    workbook = xlwt.Workbook(encoding='ascii')
    name = ['20', '40', '60', '80']

    pos_list = []
    neg_list = []
    total_list = []
    worksheet1 = workbook.add_sheet('acc')
    total = []
    pos = []
    neg = []
    for i in range(d):
        bands = choice[i]
        subdir = '112x96_{}'.format(num_c[i])
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
        print('c {}: total {}, {:.4f}| pos {}, {:.4f} | neg {}, {:.4f}'.format(num_c[i],
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
    # plot
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
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
    # plt.plot(np.arange(d), total, label='total sample', color='b',
    #          linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    # for a, b in enumerate(total):
    #     plt.text(a, b, '{:.4f}'.format(b),
    #              ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(d), name)
    plt.xlabel('spectrum resolution')
    plt.ylabel('acc')
    plt.title('Face Verification, Noise Acc Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='small')
    #plt.savefig('./face_verification_spectrum_acc.png', dpi=200)
    plt.show()
    workbook.save('./face_verification_spectrum_acc.xls')


def spectrum_tsne_analysis():
    bands1 = np.arange(550, 991, 20)
    bands2 = np.arange(550, 991, 40)
    bands3 = np.arange(550, 991, 60)
    bands4 = np.arange(550, 991, 80)
    bands5 = np.arange(550, 991, 100)
    choice = [bands1, bands2, bands3, bands4]
    num_c = [23, 12, 8, 6]
    d = len(choice)
    workbook = xlwt.Workbook(encoding='ascii')
    name = ['20', '40', '60', '80', '100']

    for i in range(d):
        bands = choice[i]
        subdir = '112x96_{}'.format(num_c[i])
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
                       title='t-SNE dimension reduction({}, dim=2, n_id={})'.format(name[i], n_id))
        plt.savefig(
            './face_verification_spectrum_tsne_{}.png'.format(name[i]), dpi=200)


def spectrum_roc():
    bands1 = np.arange(550, 991, 20)
    bands2 = np.arange(550, 991, 40)
    bands3 = np.arange(550, 991, 60)
    bands4 = np.arange(550, 991, 80)
    bands5 = np.arange(550, 991, 100)
    choice = [bands1, bands2, bands3, bands4]
    num_c = [23, 12, 8, 6]
    d = len(choice)
    workbook = xlwt.Workbook(encoding='ascii')
    name = ['20', '40', '60', '80']

    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    plt.figure(figsize=(10, 10))
    for i in range(d):
        bands = choice[i]
        subdir = '112x96_{}'.format(num_c[i])
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
        fpr, tpr, thresholds = roc_curve(labels, scores)
        mask = tpr > 0.6
        fpr = fpr[mask]
        tpr = tpr[mask]
        auc = roc_auc_score(labels, scores)
        #plt.subplot(2, 2, i + 1)
        plt.plot(fpr, tpr, label='SR={}, AUC={:.4f}'.format(name[i], auc),
                 color=colors[i], linewidth=5)
        plt.xlabel('False Positive Rate')
        #plt.yticks(np.arange(0.4, 1.0, 0.01))
        plt.ylabel('True Positive Rate')
        plt.title('Face Verification, ROC Curve', fontdict={'fontsize': 10})
        plt.legend(loc='lower right', fontsize='large')
    #plt.savefig('./face_verification_spectrum_roc.png', dpi=200)
    plt.show()


def main():
    # spectrum_score_analysis()
    spectrum_acc_analysis()
    spectrum_roc()
    # spectrum_tsne_analysis()


if __name__ == '__main__':
    main()
