import os
import scipy.io
import numpy as np
from utils.faceverification_utils import getAccuracy, getThreshold, Evaluation_10_fold

noise = [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.04, 0.06]
#resolutions = ['112x96_noise_{}'.format(x) for x in noise]
resolutions = ['112x96_21', '112x96_5']
for s in resolutions:
    filenameLs = None
    filenameRs = None
    featureLs = None
    featureRs = None
    scores = None
    labels = None
    for fold in range(5):
        #subdir = '112x96_noise_{}'.format(s)
        test_file = './workspace/{}/MobileFacenet_HyperECUST_fold_{}/valid_result.mat'.format(
            s, fold)
        result = scipy.io.loadmat(test_file)
        filenameLs_ = result['filenameLs'].squeeze()
        filenameRs_ = result['filenameRs'].squeeze()
        featureLs_ = result['featureLs'].squeeze()
        featureRs_ = result['featureRs'].squeeze()
        labels_ = result['labels'].squeeze()
        #scores = result['scores'].squeeze()
        #Thresholds = result['Thresholds'].squeeze()
        #predictions = result['predictions'].squeeze()
        if filenameLs is None:
            filenameLs = filenameLs_
            filenameRs = filenameRs_
            featureLs = featureLs_
            featureRs = featureRs_
            labels = labels_
        else:
            filenameLs = np.concatenate((filenameLs, filenameLs_), 0)
            filenameRs = np.concatenate((filenameRs, filenameRs_), 0)
            featureLs = np.concatenate((featureLs, featureLs_), 0)
            featureRs = np.concatenate((featureRs, featureRs_), 0)
            labels = np.concatenate((labels, labels_), 0)

    Accs, Thresholds, scores, predictions, folds = Evaluation_10_fold(
        featureLs, featureRs, labels, True)
    result = {'filenameLs': filenameLs, 'filenameRs': filenameRs,
              'featureLs': featureLs, 'featureRs': featureRs,
              'labels': labels, 'Accs': Accs, 'Thresholds': Thresholds,
              'scores': scores, 'predictions': predictions, 'folds': folds}
    save_path = './workspace/{}/valid_result_fold_10.mat'.format(s)
    scipy.io.savemat(save_path, result)
    print('save to {}'.format(save_path))
