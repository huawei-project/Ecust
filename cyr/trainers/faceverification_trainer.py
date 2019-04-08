import torch
import os
import sys
import time
import datetime
import numpy as np
import scipy.io
sys.path.append(os.path.dirname(__file__))
from tensorboardX import SummaryWriter
from trainer import Trainer


class MobileFacenetTrainer(Trainer):
    def __init__(self, params, net, datasets, optimizer, lr_scheduler, criterion, workspace_dir,
                 sets=['train', 'valid', 'test'], verbose=50, eval_func=None):
        self.params = params
        self.verbose = verbose
        self.eval_func = eval_func
        # Init dir
        if workspace_dir is not None:
            workspace_dir = os.path.expanduser(workspace_dir)
            log_dir = os.path.join(workspace_dir,
                                   '{}_{}'.format(type(net).__name__, type(datasets[sets[0]]).__name__))
        # Call the constructor of the parent class (Trainer)
        super().__init__(net, datasets, optimizer, lr_scheduler, criterion,
                         batch_size=params['batch_size'], batch_size_valid=params['batch_size_valid'],
                         max_epochs=params['max_epochs'], test_freq=params['test_freq'],
                         use_gpu=params['use_gpu'], resume=params['resume'],
                         sets=sets, workspace_dir=workspace_dir, log_dir=log_dir)

    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.log_dir:
            self.writer = SummaryWriter(self.log_dir)
        else:
            raise Exception("Log dir doesn't exist!")
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
            self.writer.add_scalar(
                'lr', self.optimizer.param_groups[0]['lr'], epoch)
            # Train one epoch
            total_loss = self.train_epoch(epoch)
            self.writer.add_scalar('loss', total_loss, self.global_step)
            torch.cuda.empty_cache()
            # Evaluate the model
            if self.test_freq and epoch % self.test_freq == 0:
                acc, threshold = self.eval(epoch)
                self.writer.add_scalar('acc', acc, epoch)
                self.writer.add_scalar('threshold', threshold, epoch)
        self.print("Finished training! Best epoch {} best acc {}".format(
            self.best_epoch, self.best_acc))
        self.print("Spend time: {:.2f}h".format(
            (time.time() - start_time) / 3600))
        return

    def train_epoch(self, epoch):
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
        torch.backends.cudnn.benchmark = True
        self.n_valid = len(self.validset)
        self.print("{0:<22s} : {1:} ".format('validset sample', self.n_valid))
        self.print("<-------------Evaluate the model-------------->")
        # Evaluate one epoch
        acc, threshold, fps = self.eval_epoch()
        self.print('The {}th epoch, fps {:4.2f} | 10 fold, avg acc: {:.4f}%, avg threshold {:.4f}'.format(
            epoch, fps, acc * 100, threshold))
        # Save the checkpoint
        if self.log_dir:
            self.save_checkpoint(epoch, acc)
            self.print('=> Checkpoint was saved successfully!')
        else:
            if acc >= self.best_acc:
                self.best_epoch, self.acc = epoch, acc
        return acc, threshold

    def eval_epoch(self):
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        self.net.eval()
        featureLs = None
        featureRs = None
        valid_total_time = 0
        count = 0
        with torch.no_grad():
            for step, data in enumerate(self.validloader):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                # forward
                before_op_time = time.time()
                res = [self.net.get_feature(d).data.cpu().numpy()
                       for d in data]
                duration = time.time() - before_op_time
                valid_total_time += duration
                count += len(data[0])
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
        fps = count / valid_total_time
        result = {'fl': featureLs, 'fr': featureRs,
                  'fold': self.validset.folds, 'flag': self.validset.flags}
        # save tmp_result
        save_dir = os.path.join(self.log_dir, 'tmp_result.mat')
        scipy.io.savemat(save_dir, result)
        # accuracy
        Accs, Thresholds = self.eval_func(save_dir)
        acc, threshold = np.mean(Accs), np.mean(Thresholds)
        return acc, threshold, fps

    def test(self, threshold=0.3):
        torch.backends.cudnn.benchmark = True
        n_test = len(self.testset)
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.net.eval()
        print("<-------------Test the model-------------->")
        test_total_time = 0
        count = 0
        with torch.no_grad():
            for step, (data, label) in enumerate(self.testloader):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                before_op_time = time.time()
                res = [self.net.get_feature(
                    d).data.cpu().numpy() for d in data]
                duration = time.time() - before_op_time
                test_total_time += duration
                featureL = np.concatenate((res[0], res[1]), 1)
                featureR = np.concatenate((res[2], res[3]), 1)
                mu = np.mean(np.concatenate(
                    (featureL, featureR), 0), 0, keepdims=True)
                featureL = featureL - mu
                featureR = featureR - mu
                featureL = featureL / \
                    np.sqrt(np.sum(featureL**2, 1, keepdims=True))
                featureR = featureR / \
                    np.sqrt(np.sum(featureR**2, 1, keepdims=True))
                score = np.sum(np.multiply(featureL, featureR), 1)
                label = np.squeeze(label)
                p = np.sum(score[label == 1] > threshold)
                n = np.sum(score[label == -1] < threshold)
                count = count + p + n
            acc = 1.0 * count / n_test
            fps = self.n_test / test_total_time
            print('Test samples {}, fps {}, accuracy {}'.format(n_test, fps, acc))
        return


if __name__ == '__main__':
    pass
