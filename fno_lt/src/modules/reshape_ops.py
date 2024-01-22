import torch
import torch.nn as nn

import math

__all__ = [
    'FlattenCustom',
    'FlattenNorm',
    'Spherization',
    'SphFC',
    'SphMask',
    'SphNoGrad',
]

PI = 3.141592


class FlattenCustom(nn.Flatten):
    def __init__(self, cfg, **kwargs):
        start_dim = kwargs['start_dim'] if 'start_dim' in kwargs else 1
        end_dim = kwargs['end_dim'] if 'end_dim' in kwargs else -1
        super(FlattenCustom, self).__init__(start_dim=start_dim, end_dim=end_dim)


class FlattenNorm(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = nn.functional.normalize(x, dim=1)
        return x


class FlattenWNorm(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.dataset.num_classes))

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = nn.functional.normalize(x, dim=1)
        return x


class Spherization(nn.Module):
    def __init__(self, cfg, num_features=None, **kwargs):
        super(Spherization, self).__init__()

        if num_features is None:
            raise Exception("'num_features' is None. You have to initialize 'num_features'.")
        
        self.cfg = cfg
        self.eps = cfg.reshape.sph.eps

        radius = torch.tensor(cfg.reshape.sph.radius, dtype=torch.float32)
        scaling = torch.tensor(cfg.reshape.sph.scaling, dtype=torch.float32)
        delta = torch.tensor(cfg.reshape.sph.delta, dtype=torch.float32)

        if cfg.reshape.sph.lowerbound:
            L = 0.01
            upper_bound = torch.tensor((PI / 2) * (1. - L), dtype=torch.float32)
            phi_L = torch.tensor(math.asin(delta ** (1. / num_features)), dtype=torch.float32)
            phi_L = phi_L if phi_L < upper_bound else upper_bound
        else:
            phi_L = torch.tensor(0.).float()

        W_theta = torch.diag(torch.ones(num_features))
        W_theta = torch.cat((W_theta, W_theta[-1].unsqueeze(0)))

        W_phi = torch.ones((num_features+1, num_features+1))
        W_phi = torch.triu(W_phi, diagonal=1)
        W_phi[-2][-1] = 0.

        b_phi = torch.zeros(num_features+1)
        b_phi[-1] = -PI / 2.

        self.register_buffer('start_phi_L', phi_L)
        self.register_buffer('phi_L', phi_L)
        self.register_buffer('W_theta', W_theta)
        self.register_buffer('W_phi', W_phi)
        self.register_buffer('b_phi', b_phi)
        if cfg.reshape.sph.lrable[0]:
            self.scaling = nn.Parameter(scaling)
        else:
            self.register_buffer('scaling', scaling)
        if cfg.reshape.sph.lrable[1]:
            self.radius = nn.Parameter(radius)
        else:
            self.register_buffer('radius', radius)
    
        self.register_buffer('delta', delta)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.spherize(x)
        return x

    def spherize(self, x):
        x = self.scaling * x
        x = self.angularize(x)
        x = torch.matmul(x, self.W_theta.T)

        v_sin = torch.sin(x)
        v_cos = torch.cos(x + self.b_phi)
        
        x = torch.matmul(torch.log(torch.abs(v_sin)+self.eps), self.W_phi) \
            + torch.log(torch.abs(v_cos)+self.eps)
        x = self.radius * torch.exp(x)
 
        x = self.sign(x, v_sin, v_cos)

        return x

    def angularize(self, x):
        if self.cfg.reshape.sph.angle_type == 'half':
            return (PI - 2 * self.phi_L) * torch.sigmoid(x) + self.phi_L
        else:
            return (PI / 2 - self.phi_L) * torch.sigmoid(x) + self.phi_L

    def sign(self, x, v_sin, v_cos):
        x = self.get_sign_only_cos(x, v_cos) * x
        return x

    def get_sign_only_cos(self, x, v_cos):
        v_sign_cos = torch.zeros(x.shape, dtype=torch.int16).to(x.device)
        v_sign_cos[v_cos < 0.] = 1
        v_sign_cos[:, -1] = 0

        return (-1.) ** (v_sign_cos)

    def lb_decay(self, epoch, num_epochs):
        # linear
        self.phi_L = self.start_phi_L * (1 - float(epoch)/num_epochs)


class SphFC(nn.Module):
    def __init__(self, cfg, num_features=None, **kwargs):
        super(SphFC, self).__init__()

        if num_features is None:
            raise Exception("'num_features' is None. You have to initialize 'num_features'.")
        
        self.cfg = cfg
        self.eps = cfg.reshape.sph.eps

        radius = torch.tensor(cfg.reshape.sph.radius, dtype=torch.float32)
        scaling = torch.tensor(cfg.reshape.sph.scaling, dtype=torch.float32)
        delta = torch.tensor(cfg.reshape.sph.delta, dtype=torch.float32)

        if cfg.reshape.sph.lowerbound:
            L = 0.01
            upper_bound = torch.tensor((PI / 2) * (1. - L), dtype=torch.float32)
            phi_L = torch.tensor(math.asin(delta ** (1. / num_features)), dtype=torch.float32)
            phi_L = phi_L if phi_L < upper_bound else upper_bound
        else:
            phi_L = torch.tensor(0.).float()

        W_theta = torch.diag(torch.ones(num_features))
        W_theta = torch.cat((W_theta, W_theta[-1].unsqueeze(0)))

        W_phi = torch.ones((num_features+1, num_features+1))
        W_phi = torch.triu(W_phi, diagonal=1)
        W_phi[-2][-1] = 0.

        b_phi = torch.zeros(num_features+1)
        b_phi[-1] = -PI / 2.

        self.register_buffer('start_phi_L', phi_L)
        self.register_buffer('phi_L', phi_L)
        self.register_buffer('W_theta', W_theta)
        self.register_buffer('W_phi', W_phi)
        self.register_buffer('b_phi', b_phi)
        if cfg.reshape.sph.lrable[0]:
            self.scaling = nn.Parameter(scaling)
        else:
            self.register_buffer('scaling', scaling)
        if cfg.reshape.sph.lrable[1]:
            self.radius = nn.Parameter(radius)
        else:
            self.register_buffer('radius', radius)
    
        self.register_buffer('delta', delta)

        self.fc = nn.Linear(num_features, num_features)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.fc(x)
        x = self.spherize(x)
        return x

    def spherize(self, x):
        x = self.scaling * x
        x = self.angularize(x)
        x = torch.matmul(x, self.W_theta.T)

        v_sin = torch.sin(x)
        v_cos = torch.cos(x + self.b_phi)
        
        x = torch.matmul(torch.log(torch.abs(v_sin)+self.eps), self.W_phi) \
            + torch.log(torch.abs(v_cos)+self.eps)
        x = self.radius * torch.exp(x)
 
        x = self.sign(x, v_sin, v_cos)

        return x

    def angularize(self, x):
        if self.cfg.reshape.sph.angle_type == 'half':
            return (PI - 2 * self.phi_L) * torch.sigmoid(x) + self.phi_L
        else:
            return (PI / 2 - self.phi_L) * torch.sigmoid(x) + self.phi_L

    def sign(self, x, v_sin, v_cos):
        x = self.get_sign_only_cos(x, v_cos) * x
        return x

    def get_sign_only_cos(self, x, v_cos):
        v_sign_cos = torch.zeros(x.shape, dtype=torch.int16).to(x.device)
        v_sign_cos[v_cos < 0.] = 1
        v_sign_cos[:, -1] = 0

        return (-1.) ** (v_sign_cos)

    def lb_decay(self, epoch, num_epochs):
        # linear
        self.phi_L = self.start_phi_L * (1 - float(epoch)/num_epochs)


class SphMask(nn.Module):
    def __init__(self, cfg, num_features=None, **kwargs):
        super(SphMask, self).__init__()

        if num_features is None:
            raise Exception("'num_features' is None. You have to initialize 'num_features'.")
        
        self.cfg = cfg
        self.eps = cfg.reshape.sph.eps

        radius = torch.tensor(cfg.reshape.sph.radius, dtype=torch.float32)
        scaling = torch.tensor(cfg.reshape.sph.scaling, dtype=torch.float32)
        delta = torch.tensor(cfg.reshape.sph.delta, dtype=torch.float32)

        if cfg.reshape.sph.lowerbound:
            L = 0.01
            upper_bound = torch.tensor((PI / 2) * (1. - L), dtype=torch.float32)
            phi_L = torch.tensor(math.asin(delta ** (1. / num_features)), dtype=torch.float32)
            phi_L = phi_L if phi_L < upper_bound else upper_bound
        else:
            phi_L = torch.tensor(0.).float()

        W_theta = torch.diag(torch.ones(num_features))
        W_theta = torch.cat((W_theta, W_theta[-1].unsqueeze(0)))

        W_phi = torch.ones((num_features+1, num_features+1))
        W_phi = torch.triu(W_phi, diagonal=1)
        W_phi[-2][-1] = 0.

        b_phi = torch.zeros(num_features+1)
        b_phi[-1] = -PI / 2.

        self.register_buffer('start_phi_L', phi_L)
        self.register_buffer('phi_L', phi_L)
        self.register_buffer('W_theta', W_theta)
        self.register_buffer('W_phi', W_phi)
        self.register_buffer('b_phi', b_phi)
        if cfg.reshape.sph.lrable[0]:
            self.scaling = nn.Parameter(scaling)
        else:
            self.register_buffer('scaling', scaling)
        if cfg.reshape.sph.lrable[1]:
            self.radius = nn.Parameter(radius)
        else:
            self.register_buffer('radius', radius)
    
        self.register_buffer('delta', delta)

    def forward(self, x, feat_mask=None, pi_mask=None):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.spherize(x, feat_mask=feat_mask, pi_mask=pi_mask)
        return x

    def spherize(self, x, feat_mask=None, pi_mask=None):
        x = self.scaling * x
        x = self.angularize(x)
        if (feat_mask is not None) and (pi_mask is not None):
            x = feat_mask * x + pi_mask
        
        x = torch.matmul(x, self.W_theta.T)

        v_sin = torch.sin(x)
        v_cos = torch.cos(x + self.b_phi)
        
        x = torch.matmul(torch.log(torch.abs(v_sin)+self.eps), self.W_phi) \
            + torch.log(torch.abs(v_cos)+self.eps)
        x = self.radius * torch.exp(x)
 
        x = self.sign(x, v_sin, v_cos)

        return x

    def angularize(self, x):
        if self.cfg.reshape.sph.angle_type == 'half':
            return (PI - 2 * self.phi_L) * torch.sigmoid(x) + self.phi_L
        else:
            return (PI / 2 - self.phi_L) * torch.sigmoid(x) + self.phi_L

    def sign(self, x, v_sin, v_cos):
        x = self.get_sign_only_cos(x, v_cos) * x
        return x

    def get_sign_only_cos(self, x, v_cos):
        v_sign_cos = torch.zeros(x.shape, dtype=torch.int16).to(x.device)
        v_sign_cos[v_cos < 0.] = 1
        v_sign_cos[:, -1] = 0

        return (-1.) ** (v_sign_cos)

    def lb_decay(self, epoch, num_epochs):
        # linear
        self.phi_L = self.start_phi_L * (1 - float(epoch)/num_epochs)


class SphNoGrad(nn.Module):
    def __init__(self, cfg, num_features=None, **kwargs):
        super(SphNoGrad, self).__init__()

        if num_features is None:
            raise Exception("'num_features' is None. You have to initialize 'num_features'.")
        
        self.cfg = cfg
        self.eps = cfg.reshape.sph.eps

        radius = torch.tensor(cfg.reshape.sph.radius, dtype=torch.float32)
        scaling = torch.tensor(cfg.reshape.sph.scaling, dtype=torch.float32)
        delta = torch.tensor(cfg.reshape.sph.delta, dtype=torch.float32)

        if cfg.reshape.sph.lowerbound:
            L = 0.01
            upper_bound = torch.tensor((PI / 2) * (1. - L), dtype=torch.float32)
            phi_L = torch.tensor(math.asin(delta ** (1. / num_features)), dtype=torch.float32)
            phi_L = phi_L if phi_L < upper_bound else upper_bound
        else:
            phi_L = torch.tensor(0.).float()

        W_theta = torch.diag(torch.ones(num_features))
        W_theta = torch.cat((W_theta, W_theta[-1].unsqueeze(0)))

        W_phi = torch.ones((num_features+1, num_features+1))
        W_phi = torch.triu(W_phi, diagonal=1)
        W_phi[-2][-1] = 0.

        b_phi = torch.zeros(num_features+1)
        b_phi[-1] = -PI / 2.

        self.register_buffer('start_phi_L', phi_L)
        self.register_buffer('phi_L', phi_L)
        self.register_buffer('W_theta', W_theta)
        self.register_buffer('W_phi', W_phi)
        self.register_buffer('b_phi', b_phi)
        if cfg.reshape.sph.lrable[0]:
            self.scaling = nn.Parameter(scaling)
        else:
            self.register_buffer('scaling', scaling)
        if cfg.reshape.sph.lrable[1]:
            self.radius = nn.Parameter(radius)
        else:
            self.register_buffer('radius', radius)
    
        self.register_buffer('delta', delta)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.spherize(x)
        return x

    def spherize(self, x):
        x = self.scaling * x
        x = self.angularize(x)
        x = torch.matmul(x, self.W_theta.T)

        v_sin = torch.sin(x.detach().clone())
        v_cos = torch.cos(x + self.b_phi)
        
        x = torch.matmul(torch.log(torch.abs(v_sin)+self.eps), self.W_phi) \
            + torch.log(torch.abs(v_cos)+self.eps)
        x = self.radius * torch.exp(x)
 
        x = self.sign(x, v_sin, v_cos)

        return x

    def angularize(self, x):
        if self.cfg.reshape.sph.angle_type == 'half':
            return (PI - 2 * self.phi_L) * torch.sigmoid(x) + self.phi_L
        else:
            return (PI / 2 - self.phi_L) * torch.sigmoid(x) + self.phi_L

    def sign(self, x, v_sin, v_cos):
        x = self.get_sign_only_cos(x, v_cos) * x
        return x

    def get_sign_only_cos(self, x, v_cos):
        v_sign_cos = torch.zeros(x.shape, dtype=torch.int16).to(x.device)
        v_sign_cos[v_cos < 0.] = 1
        v_sign_cos[:, -1] = 0

        return (-1.) ** (v_sign_cos)

    def lb_decay(self, epoch, num_epochs):
        # linear
        self.phi_L = self.start_phi_L * (1 - float(epoch)/num_epochs)

