# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-12 11:54:13
@LastEditTime: 2019-09-17 11:32:56
@Update: 
'''
import os
import cv2
import scipy
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding
from tensorboardX import SummaryWriter

def _condition(filename):
    """ 
    Params:
        filename: {str}
    Returns:
        isTrue: {bool}
    Notes:
    -   室内`Indoordetect`；
    -   人员标签1~10；
    -   多光谱`multi`；
    -   无光照`normal`；
    -   所有位置`Multi_[1 ~ 7]_W1_*`
    -   所有眼镜`Multi_*_W1_[1, 5, 6]`；
    -   连续拍摄4组，选择第1组；
    -   第1个通道；
    """
    filename = filename.strip().split('/')

    isIndoor = filename[5] == 'Indoordetect'
    isUser   = int(filename[6]) < 30
    isMulti  = filename[7] == 'multi'
    isNormal = filename[8] == 'normal'

    filename[9] = filename[9].split('_')
    pos   = filename[9][1]
    glass = filename[9][3]
    
    isPos    = int(pos)   in [1, 4, 7]
    isGlass  = int(glass) in [1, 5, 6]

    isFirstChannel = int(filename[-1].split('.')[0]) == 1

    return isIndoor and isUser and isMulti and \
            isNormal and isPos and isGlass and isFirstChannel

def fetchEmbeddings(matfile, condition=None):
    """ 读取`.mat`，获取embedding向量
    
    Params:
        matfile: {str} path of `.mat` file
        condition: {callable function or None}
    Returns:
        filenames: {ndarray(n_samples), str}
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
    """
    mat = scipy.io.loadmat(matfile)
    filenameLs, featureLs = mat['filenameLs'], mat['featureLs']
    filenameRs, featureRs = mat['filenameRs'], mat['featureRs']
    filenames = np.r_[filenameLs, filenameRs]; X = np.r_[featureLs, featureRs]
    filenames = list(map(lambda x: x.strip(), filenames))

    if condition is not None:
        index = list(map(lambda x: condition(x), filenames))
        X  = X [index]
        filenames = filenames[index]

    y = np.array(list(map(lambda x: int(x.split('/')[6]), filenames)))
    print("Data fetched! ", X.shape, y.shape)

    filename, index = np.unique(filenames, return_index=True)
    X = X[index]; y = y[index]
    
    return filenames, X, y

def fetchEmbeddings2(matfile, condition=None):
    """ 读取`.mat`，获取embedding向量
    
    Params:
        matfile: {str} path of `.mat` file
        condition: {callable function or None}
    Returns:
        filenames: {ndarray(n_samples), str}
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
    """
    mat = scipy.io.loadmat(matfile)
    filenames, X = mat['filename'], mat['feature']
    filenames = list(map(lambda x: x.strip(), filenames))

    if condition is not None:
        index = list(map(lambda x: condition(x), filenames))
        X = X [index]
        filenames = filenames[index]

    y = np.array(list(map(lambda x: int(x.split('/')[6]), filenames)))
    print("Data fetched! ", X.shape, y.shape)

    filename, index = np.unique(filenames, return_index=True)
    X = X[index]; y = y[index]
    
    return filenames, X, y

def plotEmbeddings(X, y, filenames=None, num=-1, logdir='plots'):
    """ 绘制embedding, tensorboard
    
    Params:
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
        filenames: {ndarray(n_samples), str}
        logdir: {str}
    """
    print("Randomly choose...")
    if num != -1:
        index = np.random.choice(list(range(X.shape[0])), num, replace=False)
        filenames = np.array(filenames)[index] if filenames is not None else None
        X = X[index]; y = y[index]

    if filenames is None:
        images = None
    else:
        filenames = list(map(lambda x: x if os.path.isfile(x) else '{}/{}'.format(x, '1.jpg'), filenames))
        images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), filenames))
        images = torch.ByteTensor(np.array(list(map(lambda x: np.transpose(x, axes=[2, 0, 1]), images))))

    print("Ploting...")
    with SummaryWriter(logdir) as writer:
        writer.add_embedding(mat=X, metadata=y, label_img=images)
    print("------ Done ------")

    return X, y

def plotTsneEmbeddings(X, y, filenames=None, dim=3, num=-1, savefile='tsne.npy', logdir='plots'):
    """ 绘制embedding, tensorboard
    
    Params:
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
        filenames: {ndarray(n_samples), str}
        logdir: {str}
    """
    print("dir: ", logdir)
    # print("TSNE...")
    # if os.path.exists(savefile):
    #     X = np.load(savefile)
    # else:
    #     X = TSNE(n_components=dim).fit_transform(X)
    #     np.save(savefile, X)

    print("Randomly choose...")
    if num != -1:
        index = np.random.choice(list(range(X.shape[0])), num, replace=False)
        filenames = np.array(filenames)[index] if filenames is not None else None
        X = X[index]; y = y[index]

    print("TSNE...")
    X = TSNE(n_components=dim).fit_transform(X)

    if filenames is None:
        images = None
    else:
        filenames = list(map(lambda x: x if os.path.isfile(x) else '{}/{}'.format(x, '1.jpg'), filenames))
        images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), filenames))
        images = torch.ByteTensor(np.array(list(map(lambda x: np.transpose(x, axes=[2, 0, 1]), images))))

    print("Ploting...")
    with SummaryWriter(logdir) as writer:
        writer.add_embedding(mat=X, metadata=y, label_img=images)
    print("------ Done ------")

    return X, y

def plotSpectralEmbedding(X, y, filenames=None, dim=3, num=-1, savefile='se.npy', logdir='plots'):
    """ 绘制embedding, tensorboard
    
    Params:
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
        logdir: {str}
    """
    print("dir: ", logdir)

    print("Randomly choose...")
    if num != -1:
        index = np.random.choice(list(range(X.shape[0])), num, replace=False)
        filenames = np.array(filenames)[index] if filenames is not None else None
        X = X[index]; y = y[index]

    print("SE...")
    X = SpectralEmbedding(n_components=dim).fit_transform(X)

    if filenames is None:
        images = None
    else:
        filenames = list(map(lambda x: x if os.path.isfile(x) else '{}/{}'.format(x, '1.jpg'), filenames))
        images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), filenames))
        images = torch.ByteTensor(np.array(list(map(lambda x: np.transpose(x, axes=[2, 0, 1]), images))))

    print("Ploting...")
    with SummaryWriter(logdir) as writer:
        writer.add_embedding(mat=X, metadata=y, label_img=images)
    print("------ Done ------")

    return X, y

def plotLocallyLinearEmbedding(X, y, filenames=None, dim=3, num=-1, savefile='se.npy', logdir='plots'):
    """ 绘制embedding, tensorboard
    
    Params:
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
        logdir: {str}
    """
    print("dir: ", logdir)

    print("Randomly choose...")
    if num != -1:
        index = np.random.choice(list(range(X.shape[0])), num, replace=False)
        filenames = np.array(filenames)[index] if filenames is not None else None
        X = X[index]; y = y[index]

    print("SE...")
    X = LocallyLinearEmbedding(n_neighbors=50, n_components=dim).fit_transform(X)

    if filenames is None:
        images = None
    else:
        filenames = list(map(lambda x: x if os.path.isfile(x) else '{}/{}'.format(x, '1.jpg'), filenames))
        images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), filenames))
        images = torch.ByteTensor(np.array(list(map(lambda x: np.transpose(x, axes=[2, 0, 1]), images))))

    print("Ploting...")
    with SummaryWriter(logdir) as writer:
        writer.add_embedding(mat=X, metadata=y, label_img=images)
    print("------ Done ------")

    return X, y

if __name__ == "__main__":
    
    datapath = '/datasets/Indoordetect'
    featurepath = '/home/louishsu/Work/Workspace/features'
    
    # ## -------------- Multi out 3d --------------
    dirname = 'Casia+HyperECUST_HyperECUST'
    filenames, X, y = fetchEmbeddings('{}/{}/train_result.mat'.format(featurepath, dirname), None)
    
    # plotEmbeddings(X=X[:, :3], y=y, filenames=list(map(lambda x: '{}/{}'.format(datapath, '/'.join(x.split('/')[6:])), filenames)), 
    #                   logdir='{}/plots/{}_plots_with_fig/'.format(featurepath, dirname))
    plotEmbeddings(X=X[:, :3], y=y,                      logdir='{}/plots/{}_plots/'.format(featurepath, dirname))

    ## -------------- Multi tsne(256 -> 3) --------------
    # dirname = 'workspace_mi/Casia+HyperECUSTMI_HyperECUSTMI'
    # filenames, X, y = fetchEmbeddings('{}/{}/val_result.mat'.format(featurepath, dirname), None)
    dirname = 'test_result_unique/test_result_unique.mat'
    filenames, X, y = fetchEmbeddings2('{}/{}'.format(featurepath, dirname), None)

    dim=2
    plotTsneEmbeddings(X=X, y=y, filenames=None,      dim=dim, 
                savefile='{}/{}/tsne{}d.mat'.format(featurepath, dirname, dim), 
                logdir='{}/plots/{}_plots_tsne_{}d/'.format(featurepath, dirname, dim))
    # plotSpectralEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/se{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_se_{}d/'.format(featurepath, dirname, dim))
    # plotLocallyLinearEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/lle{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_lle_{}d/'.format(featurepath, dirname, dim))
                
    dim=3
    plotTsneEmbeddings(X=X, y=y, filenames=None,      dim=dim, 
                savefile='{}/{}/tsne{}d.mat'.format(featurepath, dirname, dim), 
                logdir='{}/plots/{}_plots_tsne_{}d/'.format(featurepath, dirname, dim))
    # X, y = plotSpectralEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/se{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_se_{}d/'.format(featurepath, dirname, dim))
    # X, y = plotLocallyLinearEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/lle{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_lle_{}d/'.format(featurepath, dirname, dim))

    ## -------------- RGB    tsne(256 -> 3) --------------
    # dirname = 'workspace_new/Casia+HyperECUSTRGB_HyperECUSTRGB'
    # filenames, X, y = fetchEmbeddings('{}/{}/val_result.mat'.format(featurepath, dirname), None)
    dirname = 'test_result_unique/test_result_rgb_unique.mat'
    filenames, X, y = fetchEmbeddings2('{}/{}'.format(featurepath, dirname), None)
    
    dim=2
    plotTsneEmbeddings(X=X, y=y, filenames=None,      dim=dim, 
                savefile='{}/{}/tsne{}d.mat'.format(featurepath, dirname, dim), 
                logdir='{}/plots/{}_plots_tsne_{}d/'.format(featurepath, dirname, dim))
    # plotSpectralEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/se{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_se_{}d/'.format(featurepath, dirname, dim))
    # plotLocallyLinearEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/lle{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_lle_{}d/'.format(featurepath, dirname, dim))

    plt.figure(); plt.scatter(X[:, 0], X[:, 1], c=y); plt.show()
                
    dim=3
    X, y = plotTsneEmbeddings(X=X, y=y, filenames=None,      dim=dim, 
                savefile='{}/{}/tsne{}d.mat'.format(featurepath, dirname, dim), 
                logdir='{}/plots/{}_plots_tsne_{}d/'.format(featurepath, dirname, dim))
    # X, y = plotSpectralEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/se{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_se_{}d/'.format(featurepath, dirname, dim))
    # X, y = plotLocallyLinearEmbedding(X, y, filenames=None, dim=dim, 
    #             savefile='{}/{}/lle{}d.mat'.format(featurepath, dirname, dim), 
    #             logdir='{}/plots/{}_plots_lle_{}d/'.format(featurepath, dirname, dim))

    fig = plt.figure(); ax = Axes3D(fig); ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y); plt.show()
