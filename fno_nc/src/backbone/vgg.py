"""
    ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
from torchvision.models.vgg import make_layers, cfgs, VGG
from typing import Union, List, cast

__all__=[
    'vggs11_z',
    'vggs13_z',
    'vggl13_z',
    'vgg19_z',
]


class VGGSmall_z(VGG):
    def __init__(self, cfg, **kwargs):
        super(VGGSmall_z, self).__init__(**kwargs)
        self.features_cls = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.features_cls(x)
        return x


def make_n_channels_layers(vgg_type: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
	layers: List[nn.Module] = []
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

def _vggs_z(arch, vgg_type, batch_norm, cfg, **kwargs):
    return VGGSmall_z(
        cfg, 
        features=make_n_channels_layers(
            cfgs[vgg_type], 
            batch_norm=batch_norm, 
            in_channels=cfg.backbone.in_channels), 
        **kwargs)

def vggs11_z(cfg, **kwargs):
    return _vggs_z('vggs11_z', 'A', True, cfg, **kwargs)

def vggs13_z(cfg, **kwargs):
    return _vggs_z('vggs13_z', 'B', True, cfg, **kwargs)


class VGGLarge_z(VGG):
    def __init__(self, cfg, **kwargs):
        super(VGGLarge_z, self).__init__(**kwargs)
        self.features_cls = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.features_cls(x)
        return x

def _vggl_z(arch, vgg_type, batch_norm, cfg, **kwargs):
    return VGGLarge_z(
        cfg, 
        features=make_n_channels_layers(
            cfgs[vgg_type], 
            batch_norm=batch_norm, 
            in_channels=cfg.backbone.in_channels), 
        **kwargs)

def vggl13_z(cfg, **kwargs):
    return _vggl_z('vggl13_z', 'B', True, cfg, **kwargs)
    

class VGG_z(VGG):
    def __init__(self, cfg, **kwargs):
        super(VGG_z, self).__init__(**kwargs)
        self.features_cls = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.features_cls(x)
        return x

def _vgg_z(arch, vgg_type, batch_norm, cfg, **kwargs):
    return VGG_z(cfg, features=make_layers(cfgs[vgg_type], batch_norm=batch_norm), **kwargs)

def vgg19_z(cfg, **kwargs):
    return _vgg_z('vgg19_z', 'E', True, cfg, **kwargs)
    
