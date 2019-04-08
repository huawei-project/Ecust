import os
import glob
import torch
import time
import datetime
import logging
from torch.utils.data import DataLoader


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


class Trainer(object):
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, net, datasets, optimizer, lr_scheduler, criterion,
                 batch_size, batch_size_valid,
                 max_epochs, test_freq,
                 use_gpu, resume,
                 sets, workspace_dir, log_dir):
        self.net = net
        self.datasets = datasets
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.batch_size = batch_size
        self.batch_size_valid = batch_size_valid
        self.max_epochs = max_epochs
        self.test_freq = test_freq  # 0 means don't test during training
        self.use_gpu = use_gpu
        self.resume = resume
        self.sets = sets
        self.workspace_dir = workspace_dir
        self.log_dir = log_dir
        # Init
        self.best_epoch = 0
        self.best_acc = 0
        self.start_epoch = 1
        self.global_step = 1
        self.stats = {}
        # Dataloader
        self.trainset = self.datasets[self.sets[0]]
        self.trainloader = DataLoader(self.trainset, self.batch_size,
                                      shuffle=True, num_workers=8, drop_last=False)
        self.validset = self.datasets[self.sets[1]]
        self.validloader = DataLoader(self.validset, self.batch_size_valid,
                                      shuffle=False, num_workers=8, drop_last=False)
        if 'test' in self.sets:
            self.testset = self.datasets[self.sets[2]]
            self.testloader = DataLoader(self.testset, 1, shuffle=False)
        # Workspace and log dir
        if self.workspace_dir is not None:
            if not os.path.exists(workspace_dir):
                os.makedirs(workspace_dir)
        else:
            raise Exception("Workspace dir doesn't exist!")
        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            logging = init_log(self.log_dir)
            self.print = logging.info
            self.print("Log dir: {}".format(self.log_dir))
        else:
            self.print = print

    def train(self):
        """Do training, you can overload this function according to your need."""
        torch.backends.cudnn.benchmark = True
        # Calculate total step
        self.n_train = len(self.trainset)
        self.steps_per_epoch = np.ceil(
            self.n_train / self.batch_size).astype(np.int32)
        self.verbose = min(self.verbose, self.steps_per_epoch)
        self.n_steps = self.max_epochs * self.steps_per_epoch
        # calculate model parameters memory
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        memory = para * 4 / 1000 / 1000
        self.print('Model {} : params: {:4f}M'.format(
            self.net._get_name(), memory))
        self.print('###### Experiment Parameters ######')
        for k, v in self.params.items():
            self.print('{0:<22s} : {1:}'.format(k, v))
        self.print("{0:<22s} : {1:} ".format('trainset sample', self.n_train))
        # GO!!!!!!!!!
        start_time = time.time()
        self.train_total_time = 0
        self.time_sofar = 0
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Decay Learning Rate
            self.lr_scheduler.step()
            # Train one epoch
            total_loss = self.train_epoch(epoch)
            torch.cuda.empty_cache()
            # Evaluate the model
            if self.test_freq and epoch % self.test_freq == 0:
                acc = self.eval(epoch)
        self.print("Finished training! Best epoch {} best acc {}".format(
            self.best_epoch, self.best_acc))
        self.print("Spend time: {:.2f}h".format(
            (time.time() - start_time) / 3600))

    def train_epoch(self):
        """Train one epoch, you can overload this function according to your need."""
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        self.net.train()
        # Iterate over data.
        for step, data in enumerate(self.trainloader):
            image, label = data[0].to(device), data[1].to(device)
            before_op_time = time.time()
            self.optimizer.zero_grad()
            output = self.net(image)
            total_loss = self.criterion(output, label)
            total_loss.backward()
            self.optimizer.step()
            fps = image.shape[0] / (time.time() - before_op_time)
            time_sofar = self.train_total_time / 3600
            time_left = (self.n_steps / self.global_step - 1.0) * time_sofar
            if self.verbose > 0 and (step + 1) % (self.steps_per_epoch // self.verbose) == 0:
                print_str = 'Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | fps {:4.2f} | Loss: {:7.3f} | Time elapsed {:.2f}h | Time left {:.2f}h'. \
                    format(epoch, self.max_epochs, step + 1, self.steps_per_epoch,
                           fps, total_loss, time_sofar, time_left)
                self.print(print_str)
            self.global_step += 1
            self.train_total_time += time.time() - before_op_time
        return total_loss

    def eval(self, epoch):
        """Do evaluating, you can overload this function according to your need."""
        torch.backends.cudnn.benchmark = True
        self.n_valid = len(self.validset)
        self.print("{0:<22s} : {1:} ".format('validset sample', self.n_valid))
        self.print("<-------------Evaluate the model-------------->")
        # Evaluate one epoch
        acc = self.eval_epoch()
        self.print('The {}th epoch | acc: {:.4f}%'.format(epoch, acc * 100))
        # Save the checkpoint
        if self.log_dir:
            self.save_checkpoint(epoch, acc)
            self.print('=> Checkpoint was saved successfully!')
        else:
            if acc >= self.best_acc:
                self.best_epoch, self.acc = epoch, acc
        return acc

    def eval_epoch(self):
        """Evaluate one epoch, you can overload this function according to your need."""
        if self.test_freq:
            raise NotImplementedError

    def save_checkpoint(self, epoch, acc):
        """Saves a checkpoint of the network and other variables.
           Only save the best and latest epoch.
        """
        net_type = type(self.net).__name__
        if epoch - self.test_freq != self.best_epoch:
            pre_save = os.path.join(self.log_dir, '{}_{:03d}.pkl'.format(
                net_type, epoch - self.test_freq))
            if os.path.isfile(pre_save):
                os.remove(pre_save)
        cur_save = os.path.join(
            self.log_dir, '{}_{:03d}.pkl'.format(net_type, epoch))
        state = {
            'epoch': epoch,
            'acc': acc,
            'net_type': net_type,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'stats': self.stats,
            'use_gpu': self.use_gpu,
            'save_time': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        torch.save(state, cur_save)
        if acc >= self.best_acc:
            last_best = os.path.join(
                self.log_dir, '{}_{:03d}.pkl'.format(net_type, self.best_epoch))
            if os.path.isfile(last_best):
                os.remove(last_best)
            self.best_epoch = epoch
            self.best_acc = acc

    def reload(self, finetune=False):
        if self.resume:
            if self.load_checkpoint(self.resume, finetune):
                self.print('Checkpoint was loaded successfully!')
        else:
            raise Exception("Resume doesn't exist!")

    def load_checkpoint(self, checkpoint=None, finetune=False):
        """Loads a network checkpoint file.
        Can be called in two different ways:
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        net_type = type(self.net).__name__
        if isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            if self.log_dir is not None:
                checkpoint_path = os.path.join(
                    self.log_dir, '{}_{:03d}.pkl'.format(net_type, checkpoint))
            else:
                raise Exception("Log dir doesn't exist!")
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        checkpoint_dict = torch.load(
            checkpoint_path, map_location={'cuda:1': 'cuda:0'})
        #assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'
        if finetune:
            start_epoch = 1
            best_acc = 0
        else:
            start_epoch = checkpoint_dict['epoch'] + 1
            best_acc = checkpoint_dict['acc']
        self.start_epoch = start_epoch
        self.best_acc = best_acc
        self.net.load_state_dict(checkpoint_dict['net'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if 'lr_scheduler' in checkpoint_dict:
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
            self.lr_scheduler.last_epoch = start_epoch - 1
        self.stats = checkpoint_dict['stats']
        self.use_gpu = checkpoint_dict['use_gpu']
        return True


if __name__ == '__main__':
    pass
