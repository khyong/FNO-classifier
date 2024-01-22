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
                'resnets18_z', 'resnets34_z',
            ]
            num_features = 512 if self.cfg.backbone.type in basic_list else 2048
        elif 'densenet' in self.cfg.backbone.type:
            if '40' in self.cfg.backbone.type:
                # growth_rate: 12, layers: [6, 6, 6]
                # 1. (64 + 6 x 12) / 2 = 68
                # 2. (68 + 6 x 12) / 2 = 70
                # 3. 70 + 6 * 12 = 142
                num_features = 142
            elif '250' in self.cfg.backbone.type:
                # growth_rate: 24, layers: [41, 41, 41]
                # 1. (64 + 41 x 24) / 2 = 524
                # 2. (524 + 41 x 24) / 2 = 754
                # 3. 754 + 41 x 24 = 1738
                num_features = 1738
            elif '201' in self.cfg.backbone.type:
                # growth_rate: 32, layers: [6, 12, 48, 32]
                # 1. (64 + 6 x 32) / 2 = 128
                # 2. (128 + 12 x 32) / 2 = 256
                # 3. (256 + 48 x 32) / 2 = 896
                # 4. 896 + 32 x 32 = 1920
                num_features = 1920
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
    
