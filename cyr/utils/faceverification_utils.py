#from __future__ import print_function
import os
import numpy as np
import scipy.io


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def Evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    Thresholds = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate(
            (featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0, keepdims=True)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.sqrt(np.sum(featureLs**2, 1, keepdims=True))
        featureRs = featureRs / np.sqrt(np.sum(featureRs**2, 1, keepdims=True))

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        Thresholds[i] = getThreshold(
            scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]],
                              flags[testFold[0]], Thresholds[i])
        #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
        # print('--------')
        # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs, Thresholds


def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model.MobileFacenet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])
        net.eval()
        nl, nr, flods, flags = parseList(lfw_dir)
        lfw_dataset = LFW(nl, nr)
        lfw_loader = torch.utils.data.DataLoader(
            lfw_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    for data in lfw_loader:
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
    count += data[0].size(0)
    print('extracing deep features from the face pair {}...'.format(count))
    res = [net(d).data.cpu().numpy()for d in data]
    featureL = np.concatenate((res[0], res[1]), 1)
    featureR = np.concatenate((res[2], res[3]), 1)
    if featureLs is None:
        featureLs = featureL
    else:
        featureLs = np.concatenate((featureLs, featureL), 0)
    if featureRs is None:
        featureRs = featureR
    else:
        featureRs = np.concatenate((featureRs, featureR), 0)
    # featureLs.append(featureL)
    # featureRs.append(featureR)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)


if __name__ == '__main__':
    pass
