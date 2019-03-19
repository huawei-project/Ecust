import os
import numpy as np
import matplotlib.pyplot as plt

from config import configer
from utiles import get_labels_from_pathlist

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
    txtfile = './dataset/{}/test.txt'.format(configer.splitmode)
    with open(txtfile, 'r') as f:
        testfiles = f.readlines()
    testfiles_sub, indexes_sub = get_sub_filenames(testfiles, subset)
    labels = get_labels_from_pathlist(testfiles_sub)

    notUsed = [16, 17, 24, 28, 30, 32]
    used = [i for i in range(1, 34) if (i not in notUsed)]
    labels = [used.index(i) for i in labels]

    npyfile = os.path.join(configer.logspath, configer.modelname, 
                    'test_output_{}.npy'.format(configer.splitmode))
    testout = np.load(npyfile)              
    testout = softmax_multioutput(testout)
    testout_sub = testout[indexes_sub]                      # ndarray(N, C, cls)

    N, C, cls = testout_sub.shape

    # 分类层特征输出统计    
    testout_sub_true = np.zeros(shape=(N, C))               # ndarray(N, C)
    for n in range(N):
        testout_sub_true[n] = testout_sub[n, :, labels[n]]
    testout_sub_true_mean = np.mean(testout_sub_true, axis=0)
    testout_sub_true_std = np.std(testout_sub_true, axis=0)

    # 通道准确率统计
    testout_sub_y_pred = np.argmax(testout_sub, axis=2)
    testout_sub_true   = np.zeros(shape=(N, C), dtype=bool)
    for n in range(N):
        y_true = labels[n]
        y_pred = testout_sub_y_pred[n]
        testout_sub_true[n] = (y_true==y_pred)
    test_out_sub_true_acc = np.mean(testout_sub_true, axis=0)


    # testout_sub_true_mean_index = np.argsort(testout_sub_true_mean)[::-1]   # 均值越大越好
    # testout_sub_true_mean_sorted = testout_sub_true_mean[testout_sub_true_mean_index]
    # testout_sub_true_std_index = np.argsort(testout_sub_true_std)           # 方差越小越好
    # testout_sub_true_std_sorted = testout_sub_true_std[testout_sub_true_std_index]

    plt.figure("testout_sub_true_mean")
    plt.bar(
        np.arange(C),  testout_sub_true_mean, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white'
    )
    plt.figure("testout_sub_true_std")
    plt.bar(
        np.arange(C),  testout_sub_true_std, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white'
    )
    plt.figure("test_out_sub_true_acc")
    plt.bar(
        np.arange(C),  test_out_sub_true_acc, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white'
    )
    plt.show()

    npyfile = os.path.join(configer.logspath, configer.modelname, 
                    'testout_sub_true_mean_{}.npy'.format(configer.splitmode))
    np.save(npyfile, testout_sub_true_mean)
    npyfile = os.path.join(configer.logspath, configer.modelname, 
                    'testout_sub_true_std_{}.npy'.format(configer.splitmode))
    np.save(npyfile, testout_sub_true_std)
    npyfile = os.path.join(configer.logspath, configer.modelname, 
                    'test_out_sub_true_acc_{}.npy'.format(configer.splitmode))
    np.save(npyfile, test_out_sub_true_acc)

    print(" ")


if __name__ == "__main__":
    analysis()
    # analysis('ob1')