import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import os
import sys
import ast
import click
import shutil
import random
import argparse
import warnings
import numpy as np

import _init_paths
import loss as custom_loss
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
)
from utils.nc_utils import *
from core.function import train_model, valid_model, nc_model, test_model
from core.trainer import Trainer
from utils.reprod import fix_seed
from utils.dist import setup, cleanup

milestone = [
      1,   2,   3,   4,   5,   7,   9,  11,  14,  18, 
     23,  28,  34,  41,  49,  59,  70,  82,  95, 110, 
    127, 145, 165, 187, 211, 237, 265, 295, 327, 350
]


def parse_args():
    parser = argparse.ArgumentParser(description="Codes for Orth")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main_worker(rank, world_size, args):
    # ----- BEGIN basic setting -----
    update_config(cfg, args)
    logger = None

    verbose = (not cfg.ddp) or (rank == 0)
    if verbose:
        logger, log_file = create_logger(cfg)
        warnings.filterwarnings("ignore")

    fix_seed(cfg.seed_num)

    if cfg.ddp:
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size, port=cfg.port)

    torch.cuda.set_device(rank)
    # ----- END basic setting -----

    # ----- BEGIN dataset setting -----
    transform_tr = get_transform(cfg, mode='train')
    transform_ts = get_transform(cfg, mode='test')

    train_set = getattr(custom_dataset, cfg.dataset.dataset)(
        cfg, train=True, download=True, transform=transform_tr)

    if not isinstance(train_set.targets, torch.Tensor):
        train_set.targets = torch.tensor(train_set.targets, dtype=torch.long)
    num_classes = len(torch.unique(train_set.targets))
    num_class_list, ctgy_list = get_category_list(train_set.targets, num_classes, cfg)

    param_dict = {
        'num_classes': num_classes,
        'num_class_list': num_class_list,
        'cfg': cfg,
        'rank': rank,
    }

    class_map = train_set.class_map if cfg.dataset.dataset in ['ImageNetLT', 'Places'] else None
    valid_set = getattr(custom_dataset, cfg.dataset.dataset)(
        cfg, train=False, download=True, transform=transform_ts, class_map=class_map)
    
    if cfg.train.sampler.type == 'CAS':
        trainsampler = custom_dataset.ClassAwareSampler(train_set)
    else:
        trainsampler = DistributedSampler(train_set) if cfg.ddp else None
    validsampler = DistributedSampler(valid_set) if cfg.ddp else None
    
    if cfg.ddp:
        batch_size = int(cfg.train.batch_size / world_size)
        num_workers = int((cfg.train.num_workers+world_size-1)/world_size)
    else:
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers

    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(trainsampler is None),
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        sampler=trainsampler,
    )
    validloader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        sampler=validsampler,
    )
    # ----- END dataset setting -----

    # ----- BEGIN model builder -----
    num_epochs = cfg.train.num_epochs

    model = get_model(cfg, num_classes, rank)
    if cfg.pretrained: # load pretrained model
        if os.path.isfile(cfg.pretrained):
            print("=> loading checkpoint '{}'".format(cfg.pretrained))
            checkpoint = torch.load(cfg.pretrained, map_location='cuda:{}'.format(rank))
            model.load_state_dict(checkpoint['state_dict'])
    mm = model.module if cfg.ddp or cfg.dp else model
    trainer = Trainer(cfg, rank)
    criterion = getattr(custom_loss, cfg.loss.loss_type)(param_dict=param_dict).cuda(rank)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    # ----- END model builder -----

    # ----- BEGIN recording setting -----
    if verbose:
        model_dir = os.path.join(cfg.output_dir, cfg.name, 'seed{:03d}'.format(cfg.seed_num), "models")
        code_dir = os.path.join(cfg.output_dir, cfg.name, 'seed{:03d}'.format(cfg.seed_num), "codes")
        tensorboard_dir = (
            os.path.join(cfg.output_dir, cfg.name, 'seed{:03d}'.format(cfg.seed_num), "tensorboard")
            if cfg.train.tensorboard.enable else None
        )
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            shutil.rmtree(code_dir)
            if (tensorboard_dir is not None) and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        current_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            '*.pyc', '*.so', '*.out', '*pycache*', '*.pth', '*build*', '*output*', '*datasets*'
        )
        shutil.copytree(os.path.join(current_dir, '..'), code_dir, ignore=ignore)

        if tensorboard_dir is not None:
            dummy_input = torch.rand((1, 3) + cfg.input_size).cuda(rank)
            writer = SummaryWriter(log_dir=tensorboard_dir)
            pooling_module = mm.pooling
            writer.add_graph(pooling_module, (dummy_input,))
        else:
            writer = None
    # ----- END recording setting -----

    # ----- START train & valid -----
    best_result, best_epoch, start_epoch = 0, 0, 1
    save_step = cfg.save_step if cfg.save_step != -1 else num_epochs

    if verbose:
        logger.info(
            "-------------------Train start: {} {} {} | {} {} | {} / {}--------------------".format(
                cfg.backbone.type, cfg.pooling.type, cfg.reshape.type, 
                cfg.classifier.type, cfg.scaling.type, 
                cfg.loss.loss_type,
                cfg.train.trainer.type,
            )
        )

    kwargs_tr, kwargs_val = {}, {}
    # for Imbalanced Learning
    if cfg.dataset.type == 'imbalanced':
        kwargs_tr['lt'], kwargs_val['lt'] = True, True
        if cfg.train.sampler.type == 'CAS':
            kwargs_tr['num_batches'] = int(np.ceil(float(len(trainloader.dataset))/cfg.train.batch_size))
    if cfg.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        kwargs_tr['scaler'] = scaler

    threshold_zero_err = 0.996 if 'ImageNet' in cfg.dataset.dataset else 0.999
    zero_err_check = True
    nc_dict = {
        'zero_error_epoch': 0, 
        'train_acc': 0., 
        'threshold': threshold_zero_err,
        'equinorm': [],
        'equiangularity': [],
        'maximal_angle': [],
        'class_mean_convergence': [],
        'within_class_variation': [],
        'ncc': [],
    }

    for epoch in range(start_epoch, num_epochs + 1):
        if (epoch > start_epoch) and (scheduler is not None):
            scheduler.step()
        if cfg.ddp:
            trainsampler.set_epoch(epoch)
        if cfg.reshape.sph.lb_decay:
            mm.reshape.lb_decay(epoch, num_epochs)
            print("**LB Decay** phi_L: {:.4f}".format(mm.reshape.phi_L))
        # train
        train_acc, train_loss = train_model(
            trainloader, model, epoch, num_epochs, optimizer, trainer, 
            criterion, cfg, logger, verbose, **kwargs_tr
        )
        if verbose:
            model_save_path = os.path.join(
                model_dir,
                'epoch_{}.pth'.format(epoch),
            )
            if epoch % save_step == 0:
                save_dict = {
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                }
                if not cfg.save_only_result:
                    save_dict['epoch'] = epoch
                    save_dict['state_dict'] = model.state_dict()
                    save_dict['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                    save_dict['optimizer'] = optimizer.state_dict()

                torch.save(save_dict, model_save_path)

        loss_dict, acc_dict = {'train_loss': train_loss}, {'train_acc': train_acc}

        # valid
        if (cfg.valid_step != -1) and (epoch % cfg.valid_step == 0):
            valid_acc, valid_loss, lt_acc = valid_model(
                validloader, model, epoch, 
                criterion, cfg, logger, verbose, rank, **kwargs_val
            )
            loss_dict['valid_loss'], acc_dict['valid_acc'] = valid_loss, valid_acc
            if verbose:
                if valid_acc >= best_result:
                    best_result, best_epoch = valid_acc, epoch
                    best_lt_result = lt_acc
                    save_dict = {
                        'best_result': best_result,
                        'best_lt_result': best_lt_result,
                        'best_epoch': best_epoch,
                    }
                    if not cfg.save_only_result:
                        save_dict['epoch'] = epoch
                        save_dict['state_dict'] = model.state_dict()
                        save_dict['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                        save_dict['optimizer'] = optimizer.state_dict()

                    torch.save(save_dict, os.path.join(model_dir, 'best_model.pth'))
                logger.info(
                    "-------------Best Epoch:{:>3d}   Best Acc:{:>5.2f}%------------".format(
                        best_epoch, best_result * 100
                    )
                )
        # get nc results
        if verbose:
            if zero_err_check and train_acc >= threshold_zero_err:
                zero_err_check = False
                nc_dict['zero_error_epoch'] = epoch
                nc_dict['train_acc'] = train_acc
                nc_dict['test_acc'] = valid_acc
            if epoch in milestone:
                total_features, total_labels = nc_model(trainloader, model, rank)
                # neural collapse
                feat, lbl = total_features.detach().cpu(), total_labels.detach().cpu()
                weights = mm.classifier.weight.detach().cpu()
                global_mean, class_mean = get_mean(feat, lbl)
                nc_dict['equinorm'].append(equinorm(
                    feat, lbl, weights, global_mean=global_mean, class_mean=class_mean))
                nc_dict['equiangularity'].append(equiangularity(
                    feat, lbl, weights, global_mean=global_mean, class_mean=class_mean))
                nc_dict['maximal_angle'].append(maximal_angle(
                    feat, lbl, weights, global_mean=global_mean, class_mean=class_mean))
                nc_dict['class_mean_convergence'].append(class_mean_convergence(
                    feat, lbl, weights, global_mean=global_mean, class_mean=class_mean))
                nc_dict['within_class_variation'].append(within_class_variation(
                    feat, lbl, weights, global_mean=global_mean, class_mean=class_mean))
                nc_dict['ncc'].append(ncc(
                    feat, lbl, weights, global_mean=global_mean, class_mean=class_mean))

        if cfg.train.tensorboard.enable and verbose:
            writer.add_scalars('scalar/acc', acc_dict, epoch)
            writer.add_scalars('scalar/loss', loss_dict, epoch)
    if cfg.train.tensorboard.enable and verbose:
        writer.close()
    if verbose:
        logger.info(
            "-------------------Train Finished: {} (seed:{})-------------------".format(cfg.name, cfg.seed_num)
        )
        if not cfg.ddp and not cfg.save_only_result:
            test_model(
                validloader, cfg, rank, verbose,
                num_classes=num_classes, pretrained=os.path.join(model_dir, 'best_model.pth')
            )
        torch.save(nc_dict, os.path.join(model_dir, 'nc_info.pth'))

    if cfg.ddp:
        cleanup()
    # ----- END train & valid -----

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    if cfg.ddp:
        ngpus_per_node = torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        rank = cfg.rank if cfg.rank != -1 else 0
        main_worker(rank, cfg.world_size, args)

