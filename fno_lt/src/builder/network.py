import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone
import modules


class Network(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(Network, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_features = self.get_num_features()
    
        self.backbone = getattr(backbone, cfg.backbone.type)(cfg)
        self.pooling = getattr(modules, cfg.pooling.type)()
        self.reshape = getattr(modules, cfg.reshape.type)(
            cfg, num_features=self.num_features)
        self.classifier = self._get_classifier()
        self.scaling = getattr(modules, cfg.scaling.type)(
            self.num_classes)

    def forward(self, input, **kwargs):
        if ('feature_flag' in kwargs):
            return self.extract_feature(input)
        elif ('classifier_flag' in kwargs):
            return self.classify(input)

        x = self.backbone(input)
        x = self.pooling(x)
        x = self.reshape(x)
        x = self.classifier(x)
        x = self.scaling(x)
        return x

    def extract_feature(self, input):
        x = self.backbone(input)
        x = self.pooling(x)
        x = self.reshape(x)
        return x

    def classify(self, input):
        x = self.classifier(input)
        x = self.scaling(x)
        return x

    def get_num_features(self):
        dict_num_features = {
            'SimpleFNN': 300,
            'LeNet5': 84,
        }

        if self.cfg.backbone.type in dict_num_features:
            num_features = dict_num_features[self.cfg.backbone.type]
        elif 'vgg' in self.cfg.backbone.type:
            num_features = 4096
        elif 'resnet' in self.cfg.backbone.type:
            basic_list = [
                'resnet18_z', 'resnet34_z',
                'resnetcifar18_z', 'resnetcifar34_z',
            ]
            num_features = 512 if self.cfg.backbone.type in basic_list else 2048
            if self.cfg.backbone.type.startswith('resnetcifar32'):
                if self.cfg.dataset.dataset == 'IMBALANCECIFAR10' or self.cfg.dataset.dataset == 'CIFAR10':
                    num_features = 64
                elif self.cfg.dataset.dataset == 'IMBALANCECIFAR100' or self.cfg.dataset.dataset == 'CIFAR100':
                    num_features = 128
        else:
            raise NotImplementedError
        return num_features

    def _get_classifier(self):
        bias_flag = self.cfg.classifier.bias

        if self.cfg.classifier.type == 'FC':
            classifier = nn.Linear(self.num_features, self.num_classes, bias=bias_flag)
        else:
            classifier = getattr(modules, self.cfg.classifier.type)(
                self.num_features, self.num_classes, cfg=self.cfg)
        return classifier
    
