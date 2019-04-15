import os
import numpy as np
import matplotlib.pyplot as plt

from config import configer
from utiles import get_labels_from_pathlist, getLabel, getWavelen, getVol, getDicts, getPos

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

def softmax_multioutput(array):
    """
    Params:
        array:  {ndarray(N, C, cls)}
    """
    n_samples, n_channels, n_classes = array.shape

    for i_samples in range(n_samples):
        for i_channels in range(n_channels):
            x = array[i_samples, i_channels]
            array[i_samples, i_channels] = softmax(x)

    return array


def get_sub_filenames(filenames, substr=None):
    """
    Params:
        substr: {str} 
    """
    subfiles = []
    indexes  = []
    if substr is not None:
        for filename in filenames:
            if filename.find(substr)!=-1:
                subfiles += [filename]
                indexes  += [filenames.index(filename)]
        return subfiles, indexes
    else:
        indexes = [i for i in range(len(filenames))]
        return filenames, indexes


def analysis(subset=None):
    # get test files and labels
    txtfile = './split/{}/test.txt'.format(configer.splitmode)
    with open(txtfile, 'r') as f:
        testfiles = f.readlines()

    testfiles_sub, indexes_sub = get_sub_filenames(testfiles, subset)
    labels = get_labels_from_pathlist(testfiles_sub)
    labels = [i-1 for i in labels]

    # get test outputs
    npyfile = os.path.join(configer.logspath, configer.modelname, 'test_out.npy')
    testout = np.load(npyfile)
    for i in range(testout.shape[0]):              
        testout[i] = softmax(testout[i])
    testout_sub = testout[indexes_sub]                      # ndarray(N, cls)

    N, cls = testout_sub.shape; C = 23
    WAVELEN = [550+i*20 for i in range(C)]




    ## 20190226
    # N, cls = testout_sub.shape
    # C = 46
    # testout_sub_ch = np.zeros(shape=(N, C, cls))
    # WaveLen = [550 + 10*i for i in range(C)]
    # for c in range(C):
    #     for n in range(N):
    #         if (WaveLen[c] == getWavelen(testfiles_sub[n])):
    #             testout_sub_ch[n, c] = testout_sub[n]




    ## 20190227 - AUC
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OneHotEncoder
    auc_scores = np.zeros(C)
    acc_scores = np.zeros(C)
    for c in range(C):
        y_true = []
        y_pred_prob = []
        for n in range(N):
            testfile = testfiles_sub[n]
            testout  = list(testout_sub[n])
            idxch    = WAVELEN.index(getWavelen(testfile))
            if idxch == c:                                          # 对应波长
                y_true += [getLabel(testfile)-1]
                y_pred_prob += [testout]


        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        equ = (y_true==np.argmax(y_pred_prob, axis=1)).astype('int')# 正确记1，错误记0

        y_pred_prob_bin = np.zeros(shape=y_true.shape[0])
        for i in range(y_true.shape[0]):
            prob = y_pred_prob[i, y_true[i]]
            y_pred_prob_bin[i] = prob if equ[i] else (1 - prob)     # 统计判断正确的概率，即从多分类变为二分类
        



        # AUC
        if len(set(equ)) == 1:
            auc_scores[c] = 1.
        else:
            auc_scores[c] = roc_auc_score(equ, y_pred_prob_bin)

        # ACC
        acc_scores[c] = np.mean(y_pred_prob_bin, axis=0)


    subset = 'all' if subset is None else subset
    npyfile = os.path.join(configer.logspath, configer.modelname, 'analy_{}_auc.npy'.format(subset))
    np.save(npyfile, auc_scores)
    npyfile = os.path.join(configer.logspath, configer.modelname, 'analy_{}_acc.npy'.format(subset))
    np.save(npyfile, acc_scores)

    plt.figure("testout_sub_{}_auc".format(subset))
    plt.bar(np.arange(C),  auc_scores, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white')
    plt.figure("testout_sub_{}_acc".format(subset))
    plt.bar(np.arange(C),  acc_scores, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white')
    
    # plt.show()



def analysis_pos4(subset=None):
        # get test files and labels
    txtfile = './dataset/{}/test_pos4.txt'.format(configer.splitmode)
    with open(txtfile, 'r') as f:
        testfiles = f.readlines()
    testfiles_sub, indexes_sub = get_sub_filenames(testfiles, subset)
    labels = get_labels_from_pathlist(testfiles_sub)
    used = [i for i in range(1, 34) if (i not in notUsedSubjects)]
    labels = [used.index(i) for i in labels]

    # get test outputs
    npyfile = os.path.join(configer.logspath, configer.modelname, 
                    'test_output_{}_pos4.npy'.format(configer.splitmode))
    testout = np.load(npyfile)
    for i in range(testout.shape[0]):              
        testout[i] = softmax(testout[i])
    testout_sub = testout[indexes_sub]                      # ndarray(N, cls)

    N, cls = testout_sub.shape; C = 46
    WAVELEN = [550+i*10 for i in range(C)]

    ## 20190227 - AUC
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OneHotEncoder
    auc_scores = np.zeros(C)
    acc_scores = np.zeros(C)
    for c in range(C):
        y_true = []
        y_pred_prob = []
        for n in range(N):
            testfile = testfiles_sub[n]
            testout  = list(testout_sub[n])
            idxch    = WAVELEN.index(getWavelen(testfile))
            if idxch == c:                                          # 对应波长
                y_true += [used.index(get_label_from_path(testfile))]
                y_pred_prob += [testout]

        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        equ = (y_true==np.argmax(y_pred_prob, axis=1)).astype('int')# 正确记1，错误记0

        y_pred_prob_bin = np.zeros(shape=y_true.shape[0])
        for i in range(y_true.shape[0]):
            prob = y_pred_prob[i, y_true[i]]
            y_pred_prob_bin[i] = prob if equ[i] else (1 - prob)     # 统计判断正确的概率，即从多分类变为二分类
        



        # AUC
        if len(set(equ)) == 1:
            auc_scores[c] = 1.
        else:
            auc_scores[c] = roc_auc_score(equ, y_pred_prob_bin)

        # ACC
        acc_scores[c] = np.mean(y_pred_prob_bin, axis=0)


    subset = 'all' if subset is None else subset
    npyfile = os.path.join(configer.logspath, configer.modelname, 'test_output_{}_{}_pos4_auc.npy'.format(configer.splitmode, subset))
    np.save(npyfile, auc_scores)
    npyfile = os.path.join(configer.logspath, configer.modelname, 'test_output_{}_{}_pos4_acc.npy'.format(configer.splitmode, subset))
    np.save(npyfile, acc_scores)

    plt.figure("testout_sub_{}_pos4_auc".format(subset))
    plt.bar(np.arange(C),  auc_scores, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white')
    plt.figure("testout_sub_{}_pos4_acc".format(subset))
    plt.bar(np.arange(C),  acc_scores, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white')
    
    plt.show()

def analysis_position(subset=None):
    # get test files and labels
    txtfile = './split/{}/test.txt'.format(configer.splitmode)
    with open(txtfile, 'r') as f:
        testfiles = f.readlines()
    testfiles_sub, indexes_sub = get_sub_filenames(testfiles, subset)
    labels = get_labels_from_pathlist(testfiles_sub)
    labels = list(map(lambda x: x-1, labels))

    # get test outputs
    npyfile = os.path.join(configer.logspath, configer.modelname, 'test_out.npy')
    testout = np.load(npyfile)
    for i in range(testout.shape[0]):              
        testout[i] = softmax(testout[i])
    testout_sub = testout[indexes_sub]                      # ndarray(N, cls)
    testout_sub = np.argmax(testout_sub, axis=1)

    n_position = np.zeros(shape=7)  # 统计各角度的样本数量
    n_accuracy = np.zeros(shape=7)  # 统计各角度的正确数量

    for i in range(len(testfiles_sub)):
        testfile = testfiles_sub[i]
        y_true   = labels[i]
        y_pred   = testout_sub[i]

        posidx = getPos(testfile) - 1
        n_position[posidx] += 1
        if y_pred == y_true: n_accuracy[posidx] += 1
    
    acc = n_accuracy / n_position
    print(list(acc))
    
def analysis_similarity(subset=None, mode='euc'):
    # get test files and labels
    txtfile = './split/{}/test.txt'.format(configer.splitmode)
    with open(txtfile, 'r') as f:
        testfiles = f.readlines()
    testfiles_sub, indexes_sub = get_sub_filenames(testfiles, subset)
    labels = get_labels_from_pathlist(testfiles_sub)
    labels = list(map(lambda x: x-1, labels))

    # get test outputs
    npyfile = os.path.join(configer.logspath, configer.modelname, 'test_out.npy')
    testout = np.load(npyfile)
    for i in range(testout.shape[0]):              
        testout[i] = softmax(testout[i])
    testout_sub = testout[indexes_sub]                      # ndarray(N, cls)

    if mode == 'euc':       # 欧氏距离
        f = lambda x, y: np.linalg.norm(x - y)
    elif mode == 'cos':     # 余弦相似度
        f = lambda x, y: x.dot(y) / (np.linalg.norm(x)*np.linalg.norm(y))

    # get similarity_matirx, 每两个样本间的相似度
    subset = 'all' if subset is None else subset
    matrixfile = os.path.join(configer.logspath, configer.modelname, 
                    'similarity_sample_{}_{}.npy'.format(configer.splitmode, subset, mode))
    if os.path.exists(matrixfile):
        similarity_matrix = np.load(matrixfile)
    else:
        similarity_matrix = np.zeros(shape=(testout_sub.shape[0], testout_sub.shape[0]))
        for i in range(testout_sub.shape[0]):
            out_i = testout_sub[i]
            out_i /= np.linalg.norm(out_i)          # 单位化
            for j in range(i, testout_sub.shape[0]):
                out_j = testout_sub[j]
                out_j /= np.linalg.norm(out_j)      # 单位化
                similarity_matrix[i, j] = f(out_i, out_j)
        np.save(matrixfile, similarity_matrix)

    # 计算类间相似度, 每两个类间的相似度
    matrixfile = os.path.join(configer.logspath, configer.modelname, 
                    'similarity_class_{}_{}.npy'.format(configer.splitmode, subset, mode))
    if os.path.exists(matrixfile):
        similarity = np.load(matrixfile)
    else:
        similarity = np.zeros(shape=(63, 63))
        for i in range(similarity.shape[0]):    # 类 i
            idxi = [l for l in range(len(labels)) if labels[l]==i]
            for j in range(similarity.shape[1]):# 类 j
                idxj = [l for l in range(len(labels)) if labels[l]==j]
                sum_similarity = 0; cnt = 0
                for ii in idxi:
                    for jj in idxj:
                        cnt += 1
                        sum_similarity += similarity_matrix[ii, jj]
                similarity[i, j] = sum_similarity / cnt
        np.save(matrixfile, similarity)
    print(similarity)
    
    # draw figure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(similarity.shape[0])
    y = np.arange(similarity.shape[1])
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, similarity, cmap=plt.cm.winter)
    plt.show()

if __name__ == "__main__":
    # analysis()
    # analysis('normal')
    # analysis('illum1')
    # analysis('illum2')

    # analysis_pos4()
    # analysis_pos4('normal')
    # analysis_pos4('illum1')
    # analysis_pos4('illum2')

    # analysis_position()
    # analysis_position('normal')
    # analysis_position('illum1')
    # analysis_position('illum2')

    analysis_similarity()
    analysis_similarity('normal')
    analysis_similarity('illum1')
    analysis_similarity('illum2')
