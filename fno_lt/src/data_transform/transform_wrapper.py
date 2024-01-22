import torchvision.transforms as transforms

from utils.registry import Registry

TRANSFORMS = Registry()


@TRANSFORMS.register("resize")
def resize(cfg, **kwargs):
    tmp = cfg.transforms.process_detail.resize
    size = tmp if tmp is not None else cfg.input_size
    return transforms.Resize(size)

@TRANSFORMS.register("center_crop")
def center_crop(cfg, **kwargs):
    tmp = cfg.transforms.process_detail.crop_size
    size = tmp if tmp is not None else cfg.input_size
    return transforms.CenterCrop(size)

@TRANSFORMS.register("random_crop")
def random_crop(cfg, **kwargs):
    tmp = cfg.transforms.process_detail.crop_size
    size = tmp if tmp is not None else cfg.input_size
    return transforms.RandomCrop(
        size, 
        padding=cfg.transforms.process_detail.random_crop.padding,
    )

@TRANSFORMS.register("random_resized_crop")
def random_resized_crop(cfg, **kwargs):
    tmp = cfg.transforms.process_detail.crop_size
    size = tmp if tmp is not None else cfg.input_size
    return transforms.RandomResizedCrop(
        size=size, 
        scale=cfg.transforms.process_detail.random_resized_crop.scale,
        ratio=cfg.transforms.process_detail.random_resized_crop.ratio,
    )

@TRANSFORMS.register("random_horizontal_flip")
def random_horizontal_flip(cfg, **kwargs):
    return transforms.RandomHorizontalFlip(p=0.5)

@TRANSFORMS.register("normalize")
def normalize(cfg, **kwargs):
    return transforms.Normalize(
        mean=cfg.transforms.process_detail.normalize.mean,
        std=cfg.transforms.process_detail.normalize.std
    )

@TRANSFORMS.register("random_rotation")
def random_rotation(cfg, **kwargs):
    degrees = cfg.transforms.process_detail.random_rotation.degrees
    return transforms.RandomRotation(degrees)

@TRANSFORMS.register("shorter_resize_for_crop")
def shorter_resize_for_crop(cfg, **kwargs):
    tmp = cfg.transforms.process_detail.resize
    size = tmp if tmp is not None else cfg.input_size
    return transforms.Resize(int(size[0] / 0.875))

@TRANSFORMS.register("color_jitter")
def color_jitter(cfg, **kwargs):
    return transforms.ColorJitter(
        brightness=cfg.transforms.process_detail.color_jitter.brightness,
        contrast=cfg.transforms.process_detail.color_jitter.contrast,
        saturation=cfg.transforms.process_detail.color_jitter.saturation,
        hue=cfg.transforms.process_detail.color_jitter.hue,
    )


def get_transform(cfg, mode='train', input_size=None):
    transform_ops = (
        cfg.transforms.train_transforms
        if mode == 'train'
        else cfg.transforms.test_transforms
    )

    transform_list = list()
    for op in transform_ops:
        if op == 'normalize': continue
        transform_list.append(TRANSFORMS[op](cfg, input_size=input_size))
    transform_list.append(transforms.ToTensor())

    if 'normalize' in transform_ops:
        transform_list.append(TRANSFORMS['normalize'](cfg))

    return transforms.Compose(transform_list)

