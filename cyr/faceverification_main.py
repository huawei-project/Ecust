import os
from torch import nn
from models.mobilefacenet import MobileFacenet
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from dataloader.HyperECUST_loader import HyperECUST_FV
from trainers.faceverification_trainer import MobileFacenetTrainer
from utils.faceverification_utils import Evaluation_10_fold
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
# gpu init
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'
# -----------------------------------------------------


def main():
    params = {
        'trainset_path': '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96',
        'trainset_txt': '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96.txt',
        'validset_path': '~/yrc/myFile/sphereface/test/data/lfw-112X96',
        'validset_txt': '~/yrc/myFile/sphereface/test/data/pairs.txt',
        'testset_path': '~/myDataset/ECUST',
        'testset_txt': '~/yrc/myFile/huaweiproj/code/datasets/face_verfication_split/split_1/pairs.txt',
        'workspace_dir': '~/yrc/myFile/huaweiproj/code/cyr/workspace',
        'batch_size': 256,
        'batch_size_valid': 32,
        'max_epochs': 50,
        'num_classes': 10572,
        'use_gpu': True,
        'height': 112,
        'width': 96,
        'test_freq': 1,
        'resume': None
    }
    # Dataset
    trainset = CASIA_Face(params['trainset_path'], params['trainset_txt'])
    validset = LFW(params['validset_path'], params['validset_txt'])
    datasets = {'train': trainset, 'valid': validset}
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
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[36, 52, 58], gamma=0.1)
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
                                   sets=list(datasets.keys()),
                                   eval_func=Evaluation_10_fold)
    # Training the model
    Trainer.train()
    return


# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
