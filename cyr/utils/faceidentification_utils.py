"""
# Author: Yuru Chen
# Time: 2019 03 25
"""
import sys
import os
import torch
from collections import namedtuple
import logging


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


model_parameters = namedtuple('parameters',
                              'model_name,'
                              'dataset_name,'
                              'dataset_loader,'
                              'network,'
                              'home_image,'
                              'log_path, '
                              'batch_size, batch_size_valid, '
                              'height, width,'
                              'learning_rate_mode,'
                              'num_epochs,'
                              'num_classes,'
                              'interval_test'
                              )

HOME = os.environ['HOME']


def get_parameters(dataset_name, model_name, dataset, model):
    if dataset_name == 'HyperECUST':
        params = model_parameters(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_loader=dataset,
            network=model,
            home_image=HOME + '/myDataset/ECUST/',
            log_path='../log/',
            batch_size=128,
            batch_size_valid=64,
            learning_rate_mode='constant',
            num_epochs=50,
            num_classes=33,
            height=128,
            width=128,
            interval_test=1
        )
    else:
        params = model_parameters(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_loader=dataset,
            network=model,
            home_image=HOME + '/myDataset/ECUST/',
            log_path='../log/',
            batch_size=64,
            batch_size_valid=64,
            learning_rate_mode='poly',
            num_epochs=200,
            num_classes=50,
            height=128,
            width=128,
            interval_test=1
        )
    return params


def safe_log(x):
    x = torch.clamp(x, 1e-6, 1e6)
    return torch.log(x)


def safe_log10(x):
    x = torch.clamp(x, 1e-6, 1e6)
    return torch.log10(x)


def vis_square(data):
    """Take an array of shape (n, height, width), or (n, height, width, 3)
            and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    data = (data - data.min()) / (data.max() - data.min())
    m = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, m**2 - data.shape[0]), (0, 1), (0, 1))  # add some space between filters
               + ((0, 0), ) * (data.ndim - 3))  # don't pad the last dimension
    data = np.pad(data, padding, mode='constant',
                  constant_values=1)  # pad with ones
    # tile the filters into an image
    data = data.reshape((m, m) + data.shape[1:]).transpose((0, 2, 1, 3) +
                                                           tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (m * data.shape[1], m * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')


def print_model_params(model):
    params = dict(model.named_parameters())
    for k, v in params.items():
        print(k.ljust(40), str(v.shape).ljust(30), 'req_grad', v.requires_grad)


def print_optim_strategy(optimizer):
    for index, p in enumerate(optimizer.param_groups):
        outputs = ''
        string = ''
        for k, v in p.items():
            if k is 'params':
                params = v
            else:
                string += (k + ':' + str(v).ljust(7) + ' ')
        for i in range(len(params)):
            outputs += ('params' + ':' +
                        str(params[i].shape).ljust(30) + ' ') + string
        print('---------{}-----------'.format(index))
        print(outputs)
