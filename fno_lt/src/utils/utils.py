import torch

import os
import time
import logging

from builder import Network
from utils.lr_scheduler import WarmupMultiStepLR


def create_logger(cfg):
    dataset = cfg.dataset.dataset
    net_type = cfg.backbone.type
    pooling_type = cfg.pooling.type
    scaling_type = cfg.scaling.type
    seed_num = cfg.seed_num
    log_dir = os.path.join(cfg.output_dir, cfg.name, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = "{}_{}_{}_{}_{}.log".format(
        dataset, net_type, pooling_type, scaling_type, seed_num)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    print(cfg)

    logger.info("--------------------Cfg is set as follow---------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):
    base_lr = cfg.train.optimizer.base_lr
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.train.optimizer.type == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=cfg.train.optimizer.momentum,
            weight_decay=cfg.train.optimizer.weight_decay,
#            nesterov=True,
        )
    elif cfg.train.optimizer.type == 'ADAM':
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.train.optimizer.weight_decay,
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.train.lr_scheduler.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.train.lr_scheduler.lr_step,
            gamma=cfg.train.lr_scheduler.lr_factor,
        )
    elif cfg.train.lr_scheduler.type == 'cosine':
        if cfg.train.lr_scheduler.cosine_decay_end > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=cfg.train.lr_scheduler.cosine_decay_end, 
                eta_min=cfg.train.lr_scheduler.eta_min,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=cfg.train.num_epochs, 
                eta_min=cfg.train.lr_scheduler.eta_min,
            )
    elif cfg.train.lr_scheduler.type == 'warmup':
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.train.lr_scheduler.lr_step,
            gamma=cfg.train.lr_scheduler.lr_factor,
            warmup_epochs=cfg.train.lr_scheduler.warm_epoch,
        )
    elif cfg.train.lr_scheduler.type == 'none':
        scheduler = None
    else:
        raise NotImplementedError(
            "Unsupported LR Scheduler: {}".format(cfg.train.lr_scheduler.type)
        )

    return scheduler


def get_model(cfg, num_classes, rank):
    model = Network(cfg, num_classes=num_classes)

    if not cfg.cpu_mode:
        model = model.cuda(rank)

    if cfg.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=False,
#            find_unused_parameters=True,
        )
    
    if cfg.dp:
        model = torch.nn.DataParallel(model)

    return model


def get_category_list(targets, num_classes, cfg):
    num_list = [0] * num_classes
    ctgy_list = []
    for tgt in targets:
        ctgy_id = tgt.item() if isinstance(tgt, torch.Tensor) else tgt
        num_list[ctgy_id] += 1
        ctgy_list.append(ctgy_id)
    return num_list, ctgy_list

