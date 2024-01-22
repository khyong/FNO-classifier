import torch
import torch.nn.functional as F

import math
import numpy as np
from copy import deepcopy
from collections import Counter
from contextlib import nullcontext

import utils.mixup_utils as mixup_utils
from core.evaluate import accuracy


class Trainer:
    def __init__(self, cfg, rank):
        self.cfg = cfg
        self.type = cfg.train.trainer.type
        self.rank = rank
        self.num_epochs = cfg.train.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.init_all_params()

    def init_all_params(self):
        self.mixup_alpha = self.cfg.train.trainer.mixup_alpha
        if 'mixup' in self.type:  # mixup, arc_mixup, arc_mixup_cls, arc_mixup_cls_cbs
            if self.type.startswith('manifold'):  # manifold_*
                self.orig_type = deepcopy(self.type[9:])
                self.type = 'manifold_variants'
            else:
                self.orig_type = deepcopy(self.type)
                self.type = 'mixup_variants'
            self.mixup_func = getattr(mixup_utils, self.orig_type + '_data')
            self.mixup_criterion = getattr(mixup_utils, self.orig_type + '_criterion')
        elif 'mix_index' in self.type:
            self.orig_type = deepcopy(self.type)
            self.type = 'mix_index_variants'
    
    def reset_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, data, targets, **kwargs):
        return getattr(Trainer, self.type)(
            self, model, criterion, data, targets, **kwargs
        )

    def _with_autocast(self):
        return torch.cuda.amp.autocast() if self.cfg.mixed_precision else nullcontext()

    def _with_freeze(self):
        return torch.no_grad() if self.cfg.backbone.backbone_freeze else nullcontext()

    def default(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)

        with self._with_autocast():
            with self._with_freeze():
                features = model(data, feature_flag=True)
            output = model(features, classifier_flag=True)
            loss = criterion(output, targets)
        
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mixup_variants(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = self.mixup_func(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )
        
        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            if 'cls' in self.orig_type:
                mm = model.module if self.cfg.ddp else model
                loss = self.mixup_criterion(
                    mm.classifier.base, mixed_features, y_a, y_b, lam)
            else:
                mixed_output = model(mixed_features, classifier_flag=True)
                loss = self.mixup_criterion(
                    criterion, mixed_output, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def manifold_variants(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        
        with self._with_autocast():
            with self._with_freeze():
                mixed_features, y_a, y_b, lam = model.backbone(
                    data, y=targets, rank=self.rank,
                    mixup_hidden=True,
                    mixup_func=self.mixup_func,
                    mixup_alpha=self.mixup_alpha)
                x = model.pooling(mixed_features)
                x = model.reshape(x)
            if 'cls' in self.orig_type:
                mm = model.module if self.cfg.ddp else model
                loss = self.mixup_criterion(
                    mm.classifier.base, x, y_a, y_b, lam)
            else:
                mixed_output = model(x, classifier_flag=True)
                loss = self.mixup_criterion(
                    criterion, mixed_output, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mix_index_variants(self, model, criterion, data, targets, **kwargs):
        data_a, data_b = data[0].cuda(self.rank), data[1].cuda(self.rank)
        y_a, y_b = targets[0].cuda(self.rank), targets[1].cuda(self.rank)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        if self.orig_type.startswith('arc'):
            pi = 3.141592
            arc_lam = np.sin(pi/2 * lam)
            mixed_x = arc_lam * data_a + np.sqrt(1 - arc_lam**2) * data_b
        else:
            mixed_x = lam * data_a + (1 - lam) * data_b

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            if self.orig_type.startswith('arc'):
                mm = model.module if self.cfg.ddp else model
                loss = mixup_utils.arc_mixup_cls_criterion(
                    mm.classifier.base, mixed_features, y_a, y_b, arc_lam)
            else:
                mixed_output = model(x, classifier_flag=True)
                loss = mixup_utils.mixup_criterion(
                    criterion, mixed_output, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mask(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = mixup_utils.arc_mixup_cls_cbs_sync_data(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )

        mm = model.module if self.cfg.ddp else model
        classes_in_batch = torch.unique(targets)
        feat_indices_in_batch = []
        feat_indices = mm.classifier.get_feat_indices()
        for cls_num, feat_idx in enumerate(feat_indices):
            if cls_num in classes_in_batch:
                feat_indices_in_batch.append(feat_idx)
        feat_indices_in_batch = np.concatenate(feat_indices_in_batch)

        with self._with_autocast():
            with self._with_freeze():
                x = mm.backbone(mixed_x)
                x = mm.pooling(x)
                x = mm.reshape(x)   # no normalization
                feat_mask = torch.zeros(x.shape[1]).float()
                feat_mask[feat_indices_in_batch] = 1.
                feat_mask = feat_mask.unsqueeze(0).cuda(self.rank)
                x = feat_mask * x
                mixed_features = F.normalize(x, dim=1)

            loss = mixup_utils.arc_mixup_cls_cbs_sync_criterion(
                mm.classifier.base, mixed_features, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mask_softmax(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = mixup_utils.arc_mixup_cbs_sync_data(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )

        mm = model.module if self.cfg.ddp else model
        classes_in_batch = torch.unique(targets)
        cls_mask = torch.zeros(self.num_classes).float()
        cls_mask[classes_in_batch] = 1.
        cls_mask = cls_mask.unsqueeze(0).cuda(self.rank)

        tgts_a = F.one_hot(y_a, num_classes=self.num_classes).float()
        tgts_b = F.one_hot(y_b, num_classes=self.num_classes).float()

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            mixed_output = model(mixed_features, classifier_flag=True)
            logits = F.normalize(cls_mask * torch.exp(mixed_output), dim=1)
            loss = lam * torch.mean(torch.sum(tgts_a * torch.log(logits + 1e-18), dim=1)) + \
                np.sqrt(1 - lam**2) * torch.mean(tgts_b * torch.log(logits + 1e-18))
            loss = (-1.) * loss

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def sphmask(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = mixup_utils.arc_mixup_cls_cbs_sync_data(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )

        mm = model.module if self.cfg.ddp else model
        classes_in_batch = torch.unique(targets)
        feat_indices_in_batch = []
        feat_indices = mm.classifier.get_feat_indices()
        for cls_num, feat_idx in enumerate(feat_indices):
            if cls_num in classes_in_batch:
                feat_indices_in_batch.append(feat_idx)
        feat_indices_in_batch = np.concatenate(feat_indices_in_batch)

        with self._with_autocast():
            with self._with_freeze():
                x = mm.backbone(mixed_x)
                x = mm.pooling(x)
                if x.dim() > 2:
                    x = x.flatten(1)
                feat_mask = torch.zeros(x.shape[1]).float()
                feat_mask[feat_indices_in_batch] = 1.
                feat_mask = feat_mask.unsqueeze(0).cuda(self.rank)
                x = feat_mask * x
                x = mm.reshape(x) 
                mixed_features = F.normalize(x, dim=1)

            loss = mixup_utils.arc_mixup_cls_cbs_sync_criterion(
                mm.classifier.base, mixed_features, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def sphmask_v2(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = mixup_utils.arc_mixup_cls_cbs_sync_data(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )

        mm = model.module if self.cfg.ddp else model
        classes_in_batch = torch.unique(targets)
        feat_indices_in_batch = []
        feat_indices = mm.classifier.get_feat_indices()
        for cls_num, feat_idx in enumerate(feat_indices):
            if cls_num in classes_in_batch:
                feat_indices_in_batch.append(feat_idx)
        feat_indices_in_batch = np.concatenate(feat_indices_in_batch)

        with self._with_autocast():
            with self._with_freeze():
                x = mm.backbone(mixed_x)
                x = mm.pooling(x)
                x = mm.reshape(x)   # no normalization
                feat_mask = torch.zeros(x.shape[1]).float()
                feat_mask[feat_indices_in_batch] = 1.
                feat_mask = feat_mask.unsqueeze(0).cuda(self.rank)
                mixed_features = feat_mask * x

            loss = mixup_utils.arc_mixup_cls_cbs_sync_criterion(
                mm.classifier.base, mixed_features, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def sphmask_v3(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = mixup_utils.arc_mixup_cls_cbs_sync_data(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )

        mm = model.module if self.cfg.ddp else model
        classes_in_batch = torch.unique(targets)
        feat_indices_in_batch = []
        feat_indices = mm.classifier.get_feat_indices()
        for cls_num, feat_idx in enumerate(feat_indices):
            if cls_num in classes_in_batch:
                feat_indices_in_batch.append(feat_idx)
        feat_indices_in_batch = np.concatenate(feat_indices_in_batch)

        with self._with_autocast():
            with self._with_freeze():
                x = mm.backbone(mixed_x)
                x = mm.pooling(x)
                if x.dim() > 2:
                    x = x.flatten(1)
                feat_mask = torch.zeros(x.shape[1]).long()
                feat_mask[feat_indices_in_batch] = 1
                pi_mask = torch.zeros(x.shape[1]).float()
                pi_mask[feat_mask==0] = 3.141592 / 2
                feat_mask = feat_mask.unsqueeze(0).float().cuda(self.rank)
                pi_mask = pi_mask.unsqueeze(0).cuda(self.rank)
                mixed_features = mm.reshape(x, feat_mask=feat_mask, pi_mask=pi_mask)

            loss = mixup_utils.arc_mixup_cls_cbs_sync_criterion(
                mm.classifier.base, mixed_features, y_a, y_b, lam)

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mask_no_normalize(self, model, criterion, data, targets, **kwargs):
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)
        mixed_x, y_a, y_b, lam = mixup_utils.arc_mixup_cls_cbs_sync_data(
            data, targets, 
            alpha=self.mixup_alpha, rank=self.rank
        )

        mm = model.module if self.cfg.ddp else model
        classes_in_batch = torch.unique(targets)
        feat_indices_in_batch = []
        feat_indices = mm.classifier.get_feat_indices()
        for cls_num, feat_idx in enumerate(feat_indices):
            if cls_num in classes_in_batch:
                feat_indices_in_batch.append(feat_idx)
        feat_indices_in_batch = np.concatenate(feat_indices_in_batch)

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
                #feat_mask = torch.zeros(mixed_features.shape[1]).float()
                #feat_mask[feat_indices_in_batch] = 1.
                #feat_mask = feat_mask.unsqueeze(0).cuda(self.rank)
                #mixed_features = feat_mask * mixed_features

            cls_a = mm.classifier.base[y_a]
            cls_b = mm.classifier.base[y_b]
            mixed_cls = lam * cls_a + np.sqrt(1 - lam**2) * cls_b
            #prob = torch.sum(mixed_features * mixed_cls, dim=1)
            #loss = 10. * torch.mean((1 - prob) ** 2)
            loss = torch.mean(torch.sum((mixed_cls - mixed_features)**2, dim=1))

        with torch.no_grad():
            output = model(data)
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

