#from __future__ import print_function
import os
import numpy as np
import scipy.io


def getAccuracy(scores, labels, threshold, weighted=False):
    """
        params:
            scores: shape (n_sample, )
            labels: shape (n_sample, )
            threshold: scalar
            weighted: assign the weights of positive and negative acc according to their number of samples
    """
    w_p = 2 / len(scores)
    w_n = 2 / len(scores)
    if weighted:
        w_p = 1 / len(scores[labels == 1])
        w_n = 1 / len(scores[labels == -1])
    p = np.sum(scores[labels == 1] > threshold)
    n = np.sum(scores[labels == -1] < threshold)
    return (p * w_p + n * w_n) / 2


def getThreshold(scores, labels, thrNum, weighted):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, labels, thresholds[i], weighted)
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def Evaluation_10_fold(fLs, fRs, labels, weighted=False, arc=True):
    """
        params: fLs, fRs: shape (n_sample, feature_dim)
        params: labels: shape (n_sample,)
        params: arc: scores = cos(fLs, fRs), else scores = norm(fLs, fRs)
    """
    if arc:
        fLs = fLs / np.sqrt(np.sum(fLs**2, 1, keepdims=True))
        fRs = fRs / np.sqrt(np.sum(fRs**2, 1, keepdims=True))
        scores = np.sum(np.multiply(fLs, fRs), 1)
    else:
        dim = fLs.shape[1]
        scores = ((fLs - fRs)**2).sum(-1) / dim
    ACCs = np.zeros(10)
    Thresholds = np.zeros(10)
    predictions = np.zeros_like(labels)

    n_pos = (labels == 1).sum()
    n_neg = (labels == -1).sum()

    print("Total samples {}, positive samples {}, negative samples {}".format(
        len(fLs), n_pos, n_neg))
    n_pos_f = int(np.ceil(n_pos / 10))
    n_neg_f = int(np.ceil(n_neg / 10))
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == -1)[0]
    predictions = np.zeros_like(labels)
    Thresholds = np.zeros(10)
    ACCs = np.zeros(10)
    Folds = np.zeros_like(scores)
    for i in range(10):
        idx_pos_val = idx_pos[n_pos_f:]
        idx_pos_test = idx_pos[:n_pos_f]
        idx_neg_val = idx_neg[n_neg_f:]
        idx_neg_test = idx_neg[:n_neg_f]
        valFold = np.concatenate((idx_pos_val, idx_neg_val), 0)
        testFold = np.concatenate((idx_pos_test, idx_neg_test), 0)
        Folds[testFold] = i

        scores_val = scores[valFold]
        scores_test = scores[testFold]
        labels_val = labels[valFold]
        labels_test = labels[testFold]
        Thresholds[i] = getThreshold(scores_val, labels_val, 1000, weighted)
        ACCs[i] = getAccuracy(scores_test, labels_test, Thresholds[i])
        predictions[idx_pos_test] = scores[idx_pos_test] > Thresholds[i]
        predictions[idx_neg_test] = scores[idx_neg_test] < Thresholds[i]
        print('---{}---'.format(i))
        print('pos val {} pos test {}, neg val {} neg test {}.'.format(
            len(idx_pos_val), len(idx_pos_test), len(idx_neg_val), len(idx_neg_test)))
        print('fold {}, acc {:.4f}, threshold {:.4f}'.format(
            i, ACCs[i], Thresholds[i]))
        idx_pos = np.concatenate((idx_pos[n_pos_f:], idx_pos[:n_pos_f]), 0)
        idx_neg = np.concatenate((idx_neg[n_neg_f:], idx_neg[:n_neg_f]), 0)

    return ACCs, Thresholds, scores, predictions, Folds


def EvaluationLDA_10_fold(fLs, fRs, labels, weighted=False):
    """
        params: fLs, fRs: shape (n_sample, feature_dim)
        params: labels: shape (n_sample,)
    """
    N = fLs.shape[0]
    scores = (fLs - fRs).sum(-1)
    ACCs = np.zeros(10)
    Thresholds = np.zeros(10)
    predictions = np.zeros_like(labels)

    n_pos = (labels == 1).sum()
    n_neg = (labels == -1).sum()
    n_pos_f = int(np.ceil(n_pos / 10))
    n_neg_f = int(np.ceil(n_neg / 10))
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == -1)[0]
    predictions = np.zeros_like(labels)
    Thresholds = np.zeros(10)
    ACCs = np.zeros(10)
    Folds = np.zeros_like(scores)
    for i in range(10):
        idx_pos_val = idx_pos[n_pos_f:]
        idx_pos_test = idx_pos[:n_pos_f]
        idx_neg_val = idx_neg[n_neg_f:]
        idx_neg_test = idx_neg[:n_neg_f]
        valFold = np.concatenate((idx_pos_val, idx_neg_val), 0)
        testFold = np.concatenate((idx_pos_test, idx_neg_test), 0)
        Folds[testFold] = i

        scores_val = scores[valFold]
        scores_test = scores[testFold]
        labels_val = labels[valFold]
        labels_test = labels[testFold]
        Thresholds[i] = getThreshold(scores_val, labels_val, 1000, weighted)
        ACCs[i] = getAccuracy(scores_test, labels_test, Thresholds[i])
        predictions[idx_pos_test] = scores[idx_pos_test] > 0.3
        predictions[idx_neg_test] = scores[idx_neg_test] < 0.3
        print('---{}---'.format(i))
        print('pos val {} pos test {}, neg val {} neg test {}.'.format(
            len(idx_pos_val), len(idx_pos_test), len(idx_neg_val), len(idx_neg_test)))
        print('fold {}, acc {:.4f}, threshold {:.4f}'.format(
            i, ACCs[i], Thresholds[i]))
        idx_pos = np.concatenate((idx_pos[n_pos_f:], idx_pos[:n_pos_f]), 0)
        idx_neg = np.concatenate((idx_neg[n_neg_f:], idx_neg[:n_neg_f]), 0)

    return ACCs, Thresholds, scores, predictions, Folds


if __name__ == '__main__':
    pass
