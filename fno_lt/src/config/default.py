from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from yacs.config import CfgNode as CN


_C = CN()

# ----- BASIC SETTINGS -----
_C.name = "default"
_C.output_dir = "./output"
_C.valid_step = 5
_C.save_step = -1
_C.show_step = 20
_C.pin_memory = True
_C.input_size = (28, 28)
_C.color_space = "Gray"
_C.cpu_mode = False
_C.eval_mode = False
_C.seed_num = 0
_C.pretrained = None
_C.save_only_result = False
# ------- ddp -------
_C.ddp = False
_C.port = 12355
_C.rank = 0
_C.world_size = -1
# ------- dp -------
_C.dp = False
# ------- fp16 -------
_C.mixed_precision = False

# ----- DATASET BUILDER -----
_C.dataset = CN()
_C.dataset.dataset = "CIFAR100"
_C.dataset.num_classes = 100
_C.dataset.root = "/data/anonymous"
_C.dataset.type = "balanced"
_C.dataset.data_type = "jpg"
_C.dataset.train_info = ""
_C.dataset.valid_info = ""
_C.dataset.random_seed = 0
_C.dataset.imbalancecifar = CN()
_C.dataset.imbalancecifar.imb_type = "exp"
_C.dataset.imbalancecifar.ratio = 0.01
_C.dataset.class_index = CN()
_C.dataset.class_index.many = [0, 3]
_C.dataset.class_index.med = [3, 7]
_C.dataset.class_index.few = [7, 10]

# ----- backbone BUILDER -----
_C.backbone = CN()
_C.backbone.type = "LeNet5"
_C.backbone.in_features = 784
_C.backbone.in_channels = 1
_C.backbone.no_bn = False
_C.backbone.no_actv = False
_C.backbone.no_dropout = False
_C.backbone.no_residual = False
_C.backbone.backbone_freeze = False
_C.backbone.backbone_pretrained = ""

# ----- pooling BUILDER -----
_C.pooling = CN()
_C.pooling.type = "Identity"

# ----- reshape BUILDER -----
_C.reshape = CN()
_C.reshape.type = "Identity"
_C.reshape.sph = CN()
_C.reshape.sph.eps = 1e-18
_C.reshape.sph.radius = 1.0
_C.reshape.sph.scaling = 5.0
_C.reshape.sph.delta = 1e-6
_C.reshape.sph.lrable = (True, False)
_C.reshape.sph.lowerbound = True
_C.reshape.sph.angle_type = 'sqrt'
_C.reshape.sph.lb_decay = False

# ----- classifier BUILDER -----
_C.classifier = CN()
_C.classifier.type = "FC"
_C.classifier.bias = True
_C.classifier.sparse_ratio = 0.5
_C.classifier.sparse_factor = -1

# ----- scaling BUILDER -----
_C.scaling = CN()
_C.scaling.type = "Identity"

# ----- loss BUILDER -----
_C.loss = CN()
_C.loss.loss_type = "CrossEntropyCustom"

_C.loss.LDAM = CN()
_C.loss.LDAM.drw_epoch = 160
_C.loss.LDAM.max_margin = 0.5

_C.loss.LAS = CN()
_C.loss.LAS.smooth_head = 0.3
_C.loss.LAS.smooth_tail = 0.0
_C.loss.LAS.shape = "concave"
_C.loss.LAS.power = None

# ----- train BUILDER -----
_C.train = CN()
_C.train.batch_size = 32
_C.train.num_epochs = 60
_C.train.shuffle = True
_C.train.num_workers = 8
_C.train.tensorboard = CN()
_C.train.tensorboard.enable = True

_C.train.trainer = CN()
_C.train.trainer.type = (
    "default"
)
_C.train.trainer.mixup_alpha = 0.2

# ----- train.sampler BUILDER -----
_C.train.sampler = CN()
_C.train.sampler.type = "default"

_C.train.sampler.dual_sampler = CN()
_C.train.sampler.dual_sampler.enable = False
_C.train.sampler.dual_sampler.type = "reversed"

_C.train.sampler.weighted_sampler = CN()
_C.train.sampler.weighted_sampler.type = "balance"

# ----- train.optimizer BUILDER -----
_C.train.optimizer = CN()
_C.train.optimizer.type = "SGD"
_C.train.optimizer.base_lr = 0.001
_C.train.optimizer.momentum = 0.9
_C.train.optimizer.weight_decay = 1e-4

# ----- train.lr_scheduler BUILDER -----
_C.train.lr_scheduler = CN()
_C.train.lr_scheduler.type = "multistep"
_C.train.lr_scheduler.lr_step = [40, 50]
_C.train.lr_scheduler.lr_factor = 0.1
_C.train.lr_scheduler.warm_epoch = 5
_C.train.lr_scheduler.cosine_decay_end = 0
_C.train.lr_scheduler.eta_min = 1e-4

# testing
_C.test = CN()
_C.test.batch_size = 32
_C.test.num_workers = 8
_C.test.model_file=""

_C.transforms = CN()
_C.transforms.train_transforms = ("random_resized_crop", "random_horizontal_flip")
_C.transforms.test_transforms = ("shorter_resize_for_crop", "center_crop")

_C.transforms.process_detail = CN()
_C.transforms.process_detail.resize = None
_C.transforms.process_detail.crop_size = None
_C.transforms.process_detail.random_crop = CN()
_C.transforms.process_detail.random_crop.padding = 4
_C.transforms.process_detail.random_resized_crop = CN()
_C.transforms.process_detail.random_resized_crop.scale = (0.08, 1.0)
_C.transforms.process_detail.random_resized_crop.ratio = (0.75, 1.333333333)
_C.transforms.process_detail.normalize = CN()
_C.transforms.process_detail.normalize.mean = [0.286,]
_C.transforms.process_detail.normalize.std = [0.353,]
_C.transforms.process_detail.random_rotation = CN()
_C.transforms.process_detail.random_rotation.degrees = 15
_C.transforms.process_detail.color_jitter = CN()
_C.transforms.process_detail.color_jitter.brightness = 0.4
_C.transforms.process_detail.color_jitter.contrast = 0.4
_C.transforms.process_detail.color_jitter.saturation = 0.4
_C.transforms.process_detail.color_jitter.hue = 0.1


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

