import torch
import torch.nn.functional as F

import os
import time
import numpy as np

import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
from utils.utils import get_model

import horovod
import horovod.torch as hvd


def train_mixed_precision(
    trainloader, model, epoch, num_epochs, optimizer, trainer,
    criterion, cfg, logger, verbose, scaler, **kwargs
):
    if cfg.eval_mode:
        model.eval()
    else:
        model.train()

    start_time = time.time()

    num_batches = len(trainloader) if 'num_batches' not in kwargs else kwargs['num_batches']
    tr_loss = AverageMeter()
    tr_acc = AverageMeter()

    trainer.reset_epoch(epoch)
    for i, (data, targets) in enumerate(trainloader):
        if i > num_batches:
            break
        cnt = targets.shape[0] if not isinstance(targets, list) else targets[0].shape[0]

        optimizer.zero_grad()
        loss, acc = trainer.forward(model, criterion, data, targets, **kwargs)

        tr_loss.update(loss.data.item(), cnt)
        tr_acc.update(acc, cnt)

        scaler.scale(loss).backward()
        optimizer.synchronize()
        scaler.unscale_(optimizer)
        with optimizer.skip_synchronize():
            scaler.step(optimizer)
        scaler.update()

        if (i & cfg.show_step == 0) and verbose:
            pbar_str = "Epoch:{:>3d} [{:>3d}/{}] loss:{:>5.3f} acc:{:>5.2f}%".format(
                epoch, i, num_batches, tr_loss.val, tr_acc.val * 100)
            logger.info(pbar_str)
    end_time = time.time()
    if verbose:
        pbar_str = "---Epoch:{:>3d}/{}".format(epoch, num_epochs) \
            + " tr_loss:{:>5.3f}".format(tr_loss.avg) \
            + " tr_acc:{:>5.2f}%".format(tr_acc.avg * 100) \
            + " elapsed_time:{:>5.2f}m---".format((end_time - start_time)/60)
        logger.info(pbar_str)

    return tr_acc.avg, tr_loss.avg


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def valid_hvd(
    dataloader, model, epoch,
    criterion, cfg, logger, verbose, rank, **kwargs
):
    model.eval()

    with torch.no_grad():
        val_loss = 0.
        val_acc = 0.
        
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.cuda(rank), targets.cuda(rank)

            features = model(data, feature_flag=True, **kwargs)
            output = model(features, classifier_flag=True, **kwargs)

            loss = criterion(output, targets)
            val_loss += loss.item()

            pred = torch.argmax(output, 1)
            acc, cnt = accuracy(pred.cpu().numpy(), targets.cpu().numpy())
            val_acc += acc

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

        val_loss = metric_average(val_loss, 'avg_loss')
        val_acc = metric_average(val_acc, 'avg_acc')
 
    if verbose:
        pbar_str = "------Valid: Epoch:{:>3d}".format(epoch) \
            + " val_loss:{:>5.3f}".format(val_loss) \
            + " val_acc:{:>5.2f}%".format(val_acc * 100)
        logger.info(pbar_str)

    return val_acc, val_loss

