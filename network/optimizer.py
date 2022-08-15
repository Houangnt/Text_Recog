import torch
import os
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR


class TransformerLossCompute(nn.Module):

    def __init__(self, criterion, optimizer, opt=None):
        """
        :param criterion: loss function
        :param optimizer: optimization function
        """

        super(TransformerLossCompute, self).__init__()
        self.criterion = criterion
        self.opt = opt
        self.optimizer = optimizer

        # self._step = 1
        # self.warmup = opt.warmup
        # self.factor = opt.factor
        # self.model_size = opt.model_size
        # self._rate = 0
        self.loss_avg = LossesAverager()

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=opt.lr_milestones, gamma=opt.lr_gamma)

    def forward(self, x, y, tgt_mask=None):
        """

        :param x: model output
        :param y: targets
        :param tgt_mask: loss mask, to ignore the loss of the padding symbols
        :return: loss value
        """
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1, y.size(-1)).float()
        )

        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(tgt_mask)
            loss = tgt_mask.view(-1, tgt_mask.shape[2]).float() * loss
            loss = loss.sum() / x.shape[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self._step += 1
        return loss

    def lr_rate_step(self, step=None):
        """
        Learning rate according to the Attetnion is all you need paper
        :param step: current step
        :return:
        """
        if step is not None:
            for _ in range(step):
                self.lr_scheduler.step()
        self.lr_scheduler.step()

    def load_optimizer(self, path, file, device):
        """
        :param path: path where checkpoint is stored
        :param file: checkpoint file name
        :param device CPU or GPU
        :return:
        """
        self.optimizer.load_state_dict((torch.load(os.path.join(path, file), map_location=device)))


class LossCompute(nn.Module):
    def __init__(self, loss_type, criterion, optimizer):
        """
        :param criterion: loss function
        :param optimizer: optimization function
        """

        super(LossCompute, self).__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_avg = LossesAverager()
        self.loss_type = loss_type

    def forward(self, preds, text, batch_size=None, length=None):
        if 'ctc' in self.loss_type:
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if 'baidu_ctc' in self.loss_type:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = self.criterion(preds, text, preds_size, length) / self.batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = self.criterion(preds, text, preds_size, length)
        else:
            target = text[:, 1:]  # without [GO] Symbol
            cost = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
        self.loss_avg.add(cost)
        train_loss = self.loss_avg.val()
        self.loss_avg.reset()
        return train_loss

    def load_optimizer(self, path, file, device):
        """
        :param path: path where checkpoint is stored
        :param file: checkpoint file name
        :param device CPU or GPU
        :return:
        """

        self.optimizer.load_state_dict((torch.load(os.path.join(path, file), map_location=device)))


class LossesAverager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def baidu_ctc(opt):
    from warpctc_pytorch import CTCLoss
    return CTCLoss()


def ctc(opt):
    return nn.CTCLoss(zero_infinity=True).to(opt.device)


def attn(opt):
    return nn.CrossEntropyLoss(ignore_index=0).to(opt.device)


def transformer(opt):
    return nn.KLDivLoss(size_average=opt.train.size_average, reduce=opt.train.reduce_loss)
