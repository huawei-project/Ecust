# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-12 11:54:13
@LastEditTime: 2019-09-12 19:00:19
@Update: 
'''
import os
import cv2
import scipy
import numpy as np
from sklearn.manifold import TSNE
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
    filenames, X = mat['filenameLs'], mat['featureLs']

    if condition is not None:
        index = list(map(lambda x: condition(x), filenames))
        X  = X [index]
        filenames = filenames[index]

    y = np.array(list(map(lambda x: int(x.split('/')[6]), filenames)))

    return filenames, X, y

def plotTsne3dEmbeddings(X, y, filenames=None, savename='tsne3d.npy', logdir='plots'):
    """ 绘制embedding, tensorboard
    
    Params:
        X: {ndarray(n_samples, n_features)}
        y: {ndarray(n_samples)}
        filenames: {ndarray(n_samples), str}
        logdir: {str}
    """
    X = TSNE(n_components=3).fit_transform(X)
    np.save(savename, X)

    images = np.array(list(map(lambda x: cv2.imread(
            x if os.isfile(x) else '{}/{}'.format(x, '1.jpg'), cv2.IMREAD_GRAYSCALE), filenames)))\
            if filenames is not None else None
    
    with SummaryWriter(logdir) as writer:
        writer.add_embedding(mat=X, metadata=y, label_img=images)

if __name__ == "__main__":
    
    import platform


    if platform.system() == 'Windows':
    
        datapath = '/datasets/Indoordetect' # TODO:
        filenames, X, y = fetchEmbeddings(
                    'C:/Work/Github/facerecognition/facerecognition/workspace_mi/Casia_HyperECUSTMI/val_result.mat', None)
        filenames = list(map(lambda x: '{}/{}'.format(datapath, '/'.join(x.split('/')[6:])), filenames))
        plotTsne3dEmbeddings(X=X, y=y, filenames=None,  # TODO: filenames
                    savename='C:/Work/Github/facerecognition/facerecognition/workspace_mi/Casia_HyperECUSTMI/val_result_tsne3d.npy',
                    logdir='C:/Work/Github/facerecognition/facerecognition/workspace_mi/Casia_HyperECUSTMI/plots/')
    else:
    
        datapath = '/datasets/Indoordetect'
        filenames, X, y = fetchEmbeddings(
                    '/home/louishsu/Work/Workspace/features/workspace_mi/Casia_HyperECUSTMI/val_result.mat', None)
        filenames = list(map(lambda x: '{}/{}'.format(datapath, '/'.join(x.split('/')[6:])), filenames))
        plotTsne3dEmbeddings(X=X, y=y, filenames=filenames, 
                    savename='/home/louishsu/Work/Workspace/features/workspace_mi/Casia_HyperECUSTMI/val_result_tsne3d.npy',
                    logdir='/home/louishsu/Work/Workspace/features/workspace_mi/Casia_HyperECUSTMI/plots/')
