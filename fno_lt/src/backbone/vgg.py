"""
    ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
from torchvision.models.vgg import make_layers, cfgs, VGG
from typing import Union, List, cast

__all__=[
    'vgg11f_bn_no_avg_pooling',
    'vgg13f_bn_no_avg_pooling',
    'vgg16f_bn_no_avg_pooling',
    'vgg19f_bn_no_avg_pooling',
    'vgg11f_bn_one',
    'vgg13f_bn_one',
    'vgg16f_bn_one',
    'vgg19f_bn_one',
]


class VGGFNoAvgPooling(VGG):
    def __init__(self, cfg, **kwargs):
        super(VGGFNoAvgPooling, self).__init__(**kwargs)
        self.no_actv = cfg.backbone.no_actv
        self.no_dropout = cfg.backbone.no_dropout
        self.features_cls = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
        )
        self.actv = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.features_cls(x)
        if not self.no_actv:
            x = self.actv(x)
        if not self.no_dropout:
            x = self.dropout(x)
        return x


def _vggf_no_avg_pooling(arch, vgg_type, batch_norm, cfg, **kwargs):
    return VGGFNoAvgPooling(cfg, features=make_layers(cfgs[vgg_type], batch_norm=batch_norm), **kwargs)


def vgg11f_bn_no_avg_pooling(cfg, **kwargs):
    return _vggf_no_avg_pooling('vgg11f_bn', 'A', True, cfg, **kwargs)

def vgg13f_bn_no_avg_pooling(cfg, **kwargs):
    return _vggf_no_avg_pooling('vgg13f_bn', 'B', True, cfg, **kwargs)

def vgg16f_bn_no_avg_pooling(cfg, **kwargs):
    return _vggf_no_avg_pooling('vgg16f_bn', 'D', True, cfg, **kwargs)

def vgg19f_bn_no_avg_pooling(cfg, **kwargs):
    return _vggf_no_avg_pooling('vgg19f_bn', 'E', True, cfg, **kwargs)

    
def make_one_channel_layers(vgg_type: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
	layers: List[nn.Module] = []
	in_channels = 1
	for v in vgg_type:
		if v == "M":
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			v = cast(int, v)
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


def _vggf_one(arch, vgg_type, batch_norm, cfg, **kwargs):
    return VGGFNoAvgPooling(cfg, features=make_one_channel_layers(cfgs[vgg_type], batch_norm=batch_norm), **kwargs)


def vgg11f_bn_one(cfg, **kwargs):
	return _vggf_one('vgg11f_bn_one', 'A', True, cfg, **kwargs)

def vgg13f_bn_one(cfg, **kwargs):
	return _vggf_one('vgg13f_bn_one', 'B', True, cfg, **kwargs)

def vgg16f_bn_one(cfg, **kwargs):
	return _vggf_one('vgg16f_bn_one', 'D', True, cfg, **kwargs)

def vgg19f_bn_one(cfg, **kwargs):
	return _vggf_one('vgg19f_bn_one', 'E', True, cfg, **kwargs)

