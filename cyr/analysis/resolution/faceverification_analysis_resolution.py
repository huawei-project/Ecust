import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import xlwt
from sklearn import manifold, decomposition
from sklearn.metrics import roc_curve, roc_auc_score
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


def resolution_pretrained_model_acc():
    resolutions = ['112x96', '96x80', '80x64', '64x48']
    accs = [0.992833, 0.990333, 0.988667, 0.982333]
    n_sample = 22793
    fps = [243.27, 237.31, 240.31, 238.36]
    params = [999552, 993408, 988288, 984192]
    params_memory = [x * 4 / (1024**2) for x in params]  # MB
    memory = [29.99, 21.42, 14.28, 8.57]  # MB
    MAdd = [387.06, 276.51, 184.38, 110.68]  # MMAdd
    Flops = [197.27, 140.93, 93.97, 56.41]  # MFlops
    MemRW = [63.91, 46.72, 32.39, 20.93]  # MB

    # plot
    # acc
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(resolutions)), accs, color='r', linewidth=5,
             marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(accs):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(resolutions)), resolutions)
    plt.ylabel('acc')
    plt.title('Face Verification, Pretrained in CASIA_WebFace, Tested in LFW',
              fontdict={'fontsize': 10})
    # plt.legend(loc='upper left', fontsize='x-small')
    plt.savefig('./face_verification_resolution_pretrained_acc.png', dpi=200)
    plt.show()

    # memory
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(resolutions)), params_memory, label='parameter',
             color='r', linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(params_memory):
        plt.text(a, b, '{:.2f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(resolutions)), memory, label='inference',
             color='g', linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(memory):
        plt.text(a, b, '{:.2f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(resolutions)), resolutions)
    plt.ylabel('MB')
    plt.title('Face Verification, Model Memory',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig('./face_verification_resolution_memory.png', dpi=200)
    plt.show()

    # MAdd and Flops
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(resolutions)), MAdd, label='MAdd',
             color='r', linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(MAdd):
        plt.text(a, b, '{:.2f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(resolutions)), Flops, label='Flops',
             color='g', linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(Flops):
        plt.text(a, b, '{:.2f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(len(resolutions)), resolutions)
    plt.ylabel('M')
    plt.title('Face Verification, Model Computation',
              fontdict={'fontsize': 10})
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig('./face_verification_resolution_computation.png', dpi=200)
    plt.show()


def resolution_finetune_model_acc():
    resolutions = ['112x96', '96x80', '80x64', '64x48']
    d = len(resolutions)
    workbook = xlwt.Workbook(encoding='ascii')

    pos_list = []
    neg_list = []
    total_list = []
    pos_band_list = []
    neg_band_list = []
    total_band_list = []
    worksheet1 = workbook.add_sheet('acc')
    #worksheet2 = workbook.add_sheet('acc_band')
    total = []
    pos = []
    neg = []
    for i in range(d):
        test_file = '../../workspace/{}/valid_result_fold_10.mat'.format(
            resolutions[i])
        result = scipy.io.loadmat(test_file)
        filenameLs = result['filenameLs'].squeeze()
        filenameRs = result['filenameRs'].squeeze()
        featureLs = result['featureLs'].squeeze()
        featureRs = result['featureRs'].squeeze()
        labels = result['labels'].squeeze()
        scores = result['scores'].squeeze()
        predictions = result['predictions'].squeeze()

        n_sample = len(filenameLs)
        mask_pos = labels == 1
        mask_neg = labels == -1
        acc_pos = predictions[mask_pos].sum() / (mask_pos).sum()
        acc_neg = predictions[mask_neg].sum() / (mask_neg).sum()
        acc_total = predictions.mean()
        pos.append(acc_pos)
        neg.append(acc_neg)
        total.append(acc_total)
        print('resolution {}: total {}, {:.4f} | pos {}, {:.4f} | neg {}, {:.4f}'.format(resolutions[i],
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
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(pos)), pos, label='positive sample', color='r',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(pos):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    plt.plot(np.arange(len(neg)), neg, label='negative sample', color='g',
             linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    for a, b in enumerate(neg):
        plt.text(a, b, '{:.4f}'.format(b),
                 ha='center', va='bottom', fontsize=10)
    # plt.plot(np.arange(len(total)), total, label='total sample', color='b',
    #          linewidth=5, marker='o', markerfacecolor='blue', markersize=10)
    # for a, b in enumerate(total):
    #     plt.text(a, b, '{:.4f}'.format(b),
    #              ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(d), resolutions)
    plt.ylabel('acc')
    plt.title('Face Verification, Image Resolutoin Analysis',
              fontdict={'fontsize': 10})
    plt.legend(loc='center right', fontsize='x-small')
    plt.savefig('./face_verification_resolution_acc.png', dpi=200)
    plt.show()
    workbook.save('./face_verification_resolution_acc.xls')


def resolution_finetune_model_roc():
    resolutions = ['112x96', '96x80', '80x64', '64x48']
    d = len(resolutions)
    colors = ['r', 'b', 'c', 'm', 'g', 'y', 'k', 'c']
    plt.figure(figsize=(10, 10))
    for i in range(d):
        test_file = '../../workspace/{}/valid_result_fold_10.mat'.format(
            resolutions[i])
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
        plt.plot(fpr, tpr, label='Res={}, AUC={:.4f}'.format(resolutions[i], auc),
                 color=colors[i], linewidth=5)
        plt.xlabel('False Positive Rate')
        #plt.yticks(np.arange(0.4, 1.0, 0.01))
        plt.ylabel('True Positive Rate')
        plt.title('Face Verification, ROC Curve', fontdict={'fontsize': 10})
        plt.legend(loc='lower right', fontsize='large')
    plt.savefig('./face_verification_resolution_roc.png', dpi=200)
    plt.show()


def main():
    # resolution_pretrained_model_acc()
    resolution_finetune_model_acc()
    # resolution_finetune_model_roc()


if __name__ == '__main__':
    main()
