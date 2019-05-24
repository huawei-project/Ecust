import os
from torch import nn
from models.mobilefacenet import MobileFacenet
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from dataloader.HyperECUST_loader import HyperECUST_FV, HyperECUST_FI, HyperECUST_FI_MI, HyperECUST_FV_MI
from trainers.faceverification_trainer import MobileFacenetTrainer
from utils.faceverification_utils import Evaluation_10_fold
from torch.optim import lr_scheduler
import torch
import torch.optim as optim
import numpy as np
# gpu init
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'
# -----------------------------------------------------
CASIA_path = '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96'
CASIA_txt = '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96.txt'
LFW_path = '~/yrc/myFile/sphereface/test/data/lfw-112X96'
LFW_txt = '~/yrc/myFile/sphereface/test/data/pairs.txt'
ECUST_path = '~/myDataset/ECUST_112x96'


def main(split=14, fold=0, c=12, bands=None):
    params = {
        'trainset_path': ECUST_path,
        'trainset_txt': '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verification_split/split_{}/train_fold_{}.txt'.format(split, fold),
        'validset_path': ECUST_path,
        'validset_txt': '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verification_split/split_{}/pairs_fold_{}.txt'.format(split, fold),
        # '~/yrc/myFile/sphereface/test/data/pairs.txt',
        # '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verfication_split/split_1/pairs.txt',
        'testset_path': ECUST_path,
        'testset_txt': '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verification_split/split_{}/pairs_fold_{}.txt'.format(split, fold),
        #'~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verfication_split/split_{}/pairs_exp_{}.txt'.format(split, 0),
        'workspace_dir': '~/yrc/myFile/huaweiproj/code/cyr/workspace/112x96_{}'.format(c),
        'log_dir': 'MobileFacenet_HyperECUST_fold_{}'.format(fold),
        #'MobileFacenet_split_{}_exp_{}'.format(split, 0),
        'batch_size': 21,
        'batch_size_valid': 64,
        'max_epochs': 40,
        'num_classes': 33,
        'use_gpu': True,
        'height': 112,
        'width': 96,
        'test_freq': 1,
        'resume': './workspace/MobileFacenet_CASIA_Face/MobileFacenet_best.pkl'
        #'./workspace/'  + '/MobileFacenet_HyperECUST_fold_{}/MobileFacenet_best.pkl'.format(fold),
        #'./workspace/' + resolution + '/MobileFacenet/MobileFacenet_best.pkl'
        #'./workspace/MobileFacenet_split_{}_exp_{}/MobileFacenet_best.pkl'.format(split, 0)
    }
    # Dataset
    trainset = HyperECUST_FI_MI(params['trainset_path'], params['trainset_txt'],
                                bands=bands)
    validset = HyperECUST_FV_MI(params['validset_path'], params['validset_txt'],
                                bands=bands)
    testset = HyperECUST_FV_MI(params['testset_path'], params['testset_txt'],
                               bands=bands)
    datasets = {'train': trainset, 'valid': validset, 'test': testset}
    # Define model
    net = MobileFacenet(trainset.class_nums)
    # Define optimizers
    ignored_params = list(map(id, net.linear1.parameters()))
    ignored_params += [id(net.weight)]
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params,
                         net.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': net.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.01, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[12], gamma=0.1)
    # Define criterion
    criterion = net.LossFunc()
    # Define trainer
    Trainer = MobileFacenetTrainer(params=params,
                                   net=net,
                                   datasets=datasets,
                                   optimizer=optimizer,
                                   lr_scheduler=exp_lr_scheduler,
                                   criterion=criterion,
                                   workspace_dir=params['workspace_dir'],
                                   log_dir=params['log_dir'],
                                   sets=list(datasets.keys()),
                                   eval_func=Evaluation_10_fold,
                                   )

    # Train model
    # Trainer.train()
    # return

    # Evaluate one epoch
    # Trainer.reload()
    # acc, threshold, _ = Trainer.eval_epoch(filename='valid_result.mat')
    # print("acc {:.4f}, threshold {:.4f}".format(acc, threshold))
    # return

    # Test model
    # Trainer.load_checkpoint(params['resume'])
    # Trainer.test(0.4)
    # return

    # Finetune
    # Load the pretrained model
    # @Note: the optimizer and lr_scheduler should be redefined if execute Trainer.reload(True),
    Trainer.reload(finetune=True)
    net = Trainer.net
    # modify the weigth of conv1
    w = net.conv1.conv.weight
    new_w = w.repeat(1, c // 3, 1, 1)
    cc = c % 3
    if cc:
        new_w = torch.cat((new_w, w[:, :cc, :, :]), 1)
    net.conv1.conv.weight.data = new_w
    print(net.conv1.conv.weight.shape)
    # Define optimizers
    ignored_params = list(map(id, net.linear1.parameters()))
    ignored_params += [id(net.weight)]
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params,
                         net.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': net.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.01, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[24, 36], gamma=0.1)

    Trainer.optimizer = optimizer
    Trainer.lr_scheduler = exp_lr_scheduler
    Trainer.train()
    del Trainer
    return


# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    # main()
    bands1 = np.arange(550, 991, 20)
    bands2 = np.arange(550, 991, 40)
    bands3 = np.arange(550, 991, 60)
    bands4 = np.arange(550, 991, 80)
    bands5 = np.arange(550, 991, 100)
    choice = [bands3]
    channel = [8]
    for j in range(len(choice)):
        bands = choice[j]
        c = channel[j]
        for i in range(5):
            print('---------fold_{}_c_{}---------'.format(i, c))
            main(fold=i, c=c, bands=bands)
