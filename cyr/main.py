"""
# Author: Yuru Chen
# Time: 2019 03 25
"""
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import scipy.io as sio
import math
from tensorboardX import SummaryWriter
from track_mem_use import gpu_profile, modelsize
sys.path.append('../datasets')
from create_dataset import HyperECUST, split_dataset
sys.path.append('../models')
from vgg import myVGG
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'


def write_to_log(path, str):
    with open(path, 'a+') as f:
        f.write(str + '\n')


def generate_all_params(modules):
    generator_ = []
    for module in modules:
        list_ = list(module.parameters())
        generator_ = generator_ + list_
    generator = (i for i in generator_)
    return generator


def generate_weight_or_bias_params(modules, Type='weight', mode='conv_and_fc'):
    """ 
    Params:
    #    mode
    #       options: conv_and_fc, bn, all
    """
    generator_ = []
    for module in modules:
        params_dict = dict(module.named_parameters())
        for name, params in params_dict.items():
            if mode == 'bn':
                if 'bn' in name and Type in name:
                    generator_ += [params]
            elif mode == 'all':
                if Type in name:
                    generator_ += [params]
            else:
                if 'bn' not in name and Type in name:
                    generator_ += [params]
    generator = (i for i in generator_)
    return generator


def fine_tune(Pretrained, myModel):
    my_model_dict = myModel.state_dict()
    new_model_dict = {key: value for key, value in Pretrained.state_dict(
    ).items() if key in my_model_dict}
    my_model_dict.update(new_model_dict)
    myModel.load_state_dict(my_model_dict)
    return myModel


def adjust_learning_rate(optimizer, step, total_step, epoch, num_epochs, mode='step'):
    # step  [1,total_step]
    # epoch [1, num_epochs]
    if mode == 'lr_finder':
        start_lr = 1e-8   # 6e-5
        end_lr = 1e-3     # 1e-4
        new_lr = (end_lr - start_lr) / (num_epochs - 1) * \
            (epoch - 1) + start_lr
    if mode == 'leslie':
        a1 = 0.4 * num_epochs
        a2 = 2 * a1
        low_lr = 1e-8  # 1e-8 Adam
        high_lr = 1e-4  # 1e-5 Adam
        if epoch <= a1:
            new_lr = (high_lr - low_lr) / (a1 - 1) * (epoch - 1) + low_lr
        elif epoch <= a2:
            new_lr = (low_lr - high_lr) / (a2 - a1) * (epoch - a1) + high_lr
        else:
            new_lr = low_lr
    if mode == 'poly':
        start_lr = 2e-4
        decay_rate = 0.9
        new_lr = start_lr * (1 - (step - 1) / total_step)**decay_rate
    if mode == 'step':
        start_lr = 1e-4
        start_epoch = 2
        divider = 0.1
        interval = 1
        if epoch < start_epoch:
            new_lr = start_lr
        else:
            new_lr = start_lr * \
                (divider**((epoch - start_epoch) // interval + 1))
    if mode == 'constant':
        new_lr = 1e-5
    optimizer.param_groups[0]['lr'] = new_lr
    optimizer.param_groups[1]['lr'] = new_lr * 10
    optimizer.param_groups[2]['lr'] = new_lr * 20
    return new_lr
# ----------------------------------------------------------------------------


def main():
    mode = 'train'
    test_index = 10
    dataset_name_list = ['HyperECUST']
    model_name_list = ['vgg']
    dataset_dict = {'HyperECUST': HyperECUST}
    model_dict = {'vgg': myVGG}
    dataset_name = dataset_name_list[0]
    model_name = model_name_list[0]
    params = get_parameters(dataset_name, model_name,
                            dataset_dict[dataset_name], model_dict[model_name])

    myModel = myVGG(config='A', num_classes=params.num_classes)
    # Fine Tune
    vgg = models.vgg11(pretrained=True)
    myModel = fine_tune(vgg, myModel)
    del vgg
    # Loss Function
    criterion = myModel.LossFunc(params.num_classes)
    # dataset
    #train_list, valid_list = split_dataset(params.home_image)
    train_path = '/home/lilium/yrc/myFile/Ecust/louishsu/spectral_analysis_sample_channels/split_23chs/split_1/train.txt'
    valid_path = '/home/lilium/yrc/myFile/Ecust/louishsu/spectral_analysis_sample_channels/split_23chs/split_1/test.txt'
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(valid_path, 'r') as f:
        valid_list = f.readlines()

    if mode == 'train':
        train(params, myModel, criterion, train_list, valid_list)
    if mode == 'test':
        test(params, myModel, test_index, train_list, valid_list)
    return


def train(params, myModel, criterion, train_list, valid_list):
    torch.backends.cudnn.benchmark = True
    trainset = params.dataset_loader(params.home_image, train_list,
                                     facesize=(params.height, params.width), cropped_by_bbox=False, mode='train')
    validset = params.dataset_loader(params.home_image, valid_list,
                                     facesize=(params.height, params.width), cropped_by_bbox=False, mode='valid')
    trainloader = DataLoader(
        trainset, batch_size=params.batch_size, shuffle=True)
    validloader = DataLoader(
        validset, batch_size=params.batch_size_valid, shuffle=True)
    # calculate total step
    num_training_samples = len(trainset)
    num_validation_samples = len(validset)
    steps_per_epoch = np.ceil(
        num_training_samples / params.batch_size).astype(np.int32)
    steps_per_epoch_valid = np.ceil(
        num_validation_samples / params.batch_size_valid).astype(np.int32)

    num_total_steps = params.num_epochs * steps_per_epoch
    num_total_steps_valid = params.num_epochs / \
        params.interval_test * steps_per_epoch_valid
    print("training set: %d " % num_training_samples)
    print("validation set: %d" % num_validation_samples)
    print("train total steps: %d" % num_total_steps)
    print("valid total steps: %d" % num_total_steps_valid)
    print("Dataset {}".format(params.dataset_name))
    # calculate model parameters memory
    dummy_input = Variable(torch.rand(
        params.batch_size, 3, params.height, params.width)).cuda()
    modelsize(myModel, dummy_input)
    del dummy_input

    myModel.cuda()

    # filter manully
    base_modules = list(myModel.children())[0]
    base_params = generate_all_params(base_modules)
    base_params = filter(lambda p: p.requires_grad, base_params)
    add_modules = list(myModel.children())[1:]
    add_weight_params = generate_weight_or_bias_params(
        add_modules, 'weight', 'all')
    add_bias_params = generate_weight_or_bias_params(
        add_modules, 'bias', 'all')
    # optimizer
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': add_weight_params, 'lr': 1e-4 * 10},
        {'params': add_bias_params, 'lr': 1e-4 * 20},
    ], betas=(0.9, 0.95), weight_decay=0.0005)
    # optimizer = optim.SGD([
    #     {'params': base_params, 'lr': 1e-4},
    #     {'params': add_weight_params, 'lr': 1e-4 * 10},
    #     {'params': add_bias_params, 'lr': 1e-4 * 20},
    # ], momentum=0.9, weight_decay=0.0005)

    WRITER_DIR = params.log_path + \
        '{}_{}/'.format(params.model_name, params.dataset_name)
    LOG_DIR = params.log_path + '{0}_{1}/{0}_{1}_loss.txt'.format(params.model_name,
                                                                  params.dataset_name)
    CHECKPOINT_DIR = params.checkpoint_path + \
        '{}_{}/'.format(params.model_name, params.dataset_name)
    if os.path.isdir(CHECKPOINT_DIR) == 0:
        os.mkdir(CHECKPOINT_DIR)
    writer = SummaryWriter(WRITER_DIR)
    # GO!!!!!!!
    start_time = time.time()
    train_total_time = 0
    global_step = 0
    best_epoch = 0
    best_acc = 0
    for epoch in range(params.num_epochs):
        torch.cuda.empty_cache()
        for step, (images, labels, _) in enumerate(trainloader):
            # adjust learning rate
            before_op_time = time.time()
            # input data
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            # zero the grad
            optimizer.zero_grad()
            # forward
            # set the training mode, bn and dropout will be updated
            myModel.train()
            predict = myModel(images)
            # compute loss
            loss = criterion(predict, labels)
            # backward
            loss.backward()
            # update parameters
            optimizer.step()
            # calculate time
            duration = time.time() - before_op_time
            fps = images.shape[0] / duration
            time_sofar = train_total_time / 60
            time_left = (num_total_steps /
                         (global_step + 1) - 1.0) * time_sofar
            # print loss
            if (step + 1) % (steps_per_epoch // 10) == 0:
                print_str = 'Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | fps {:4.2f} | Loss: {:7.3f} | Time elapsed {:.2f}min | Time_left {:.2f}min'. \
                    format(epoch + 1, params.num_epochs, step + 1,
                           steps_per_epoch, fps, loss, time_sofar, time_left)
                print(print_str)
                write_to_log(LOG_DIR, print_str)
            global_step += 1
            train_total_time += time.time() - before_op_time

        torch.cuda.empty_cache()
        if (epoch + 1) % params.interval_test == 0:
            print("<-------------Testing the model-------------->")
            test_total_time = 0
            with torch.no_grad():
                for index, (images, labels, _) in enumerate(validloader):
                    before_op_time = time.time()
                    # input data
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()
                    # forward
                    # set the evaluation mode, bn and dropout will be fixed
                    myModel.eval()
                    predict = myModel(images)
                    pred_label = myModel.inference(predict)
                    duration = time.time() - before_op_time
                    test_total_time += duration
                    # accuracy
                    acc = torch.sum(labels == pred_label).float(
                    ) / torch.numel(labels)
                fps = num_validation_samples / test_total_time
                print_str = 'The {}th epoch, fps {:4.2f} | acc {:.4f}'.format(
                    epoch + 1, fps, acc)
                print(print_str)
                write_to_log(LOG_DIR, print_str)
                # saving the model
                if epoch + 1 - params.interval_test != best_epoch:
                    last = CHECKPOINT_DIR + \
                        '{}.pkl'.format(epoch + 1 - params.interval_test)
                    if os.path.isfile(last):
                        os.remove(last)
                print("<+++++++++++++saving the model++++++++++++++>")
                now = CHECKPOINT_DIR + '{}.pkl'.format(epoch + 1)
                torch.save(myModel.state_dict(), now)
                if acc >= best_acc:
                    last_best = CHECKPOINT_DIR + '{}.pkl'.format(best_epoch)
                    if os.path.isfile(last_best):
                        os.remove(last_best)
                    best_epoch = epoch + 1
                    best_acc = acc

    print("Finished training!")
    end_time = time.time()
    print("Spend time: %.2fh" % ((end_time - start_time) / 3600))
    # gpu_profile(frame=sys._getframe(), event='line', arg=None)
    return myModel

# ------------------------------------------------------------------------------------


def test(params, myModel, test_index, train_list, test_list):
    torch.backends.cudnn.benchmark = True
    validset = params.dataset_loader(params.home_image, test_list,
                                     facesize=(params.height, params.width), cropped_by_bbox=False, mode='valid')
    validloader = DataLoader(validset, batch_size=1, shuffle=True)

    num_validation_samples = len(validset)
    steps_per_epoch_valid = np.ceil(
        num_validation_samples / params.batch_size_valid).astype(np.int32)
    print("validation set: %d" % num_validation_samples)
    # load parameters
    CHECKPOINT_DIR = params.checkpoint_path + \
        '{}_{}/'.format(params.model_name, params.dataset_name)
    myModel.load_state_dict(torch.load(
        CHECKPOINT_DIR + '{}.pkl'.format(test_index)))
    myModel.cuda()
    print("<-------------Testing the model-------------->")
    test_total_time = 0
    with torch.no_grad():
        for index, (images, labels, _) in enumerate(validloader):
            before_op_time = time.time()
            # input data
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            # forward
            # set the evaluation mode, bn and dropout will be fixed
            myModel.eval()
            predict = myModel(images)
            pred_label = myModel.inference(predict)
            duration = time.time() - before_op_time
            test_total_time += duration
            # accuracy
            acc = torch.sum(labels == pred_label).float() / torch.numel(labels)

        fps = num_validation_samples / test_total_time
        print_str = 'Testing the {} examples, fps {:4.2f} | acc {:.4f}'.format(
            num_validation_samples, fps, acc)
        print(print_str)
        #write_to_log(LOG_DIR, print_str)
    return


if __name__ == '__main__':
    # sys.settrace(gpu_profile)
    main()
