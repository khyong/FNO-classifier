# Fixed Non-negative Orthogonal Classifier: Inducing Zero-mean Neural Collapse with Feature Dimension Separation

This directory is the official implementation of the part of Imbalanced Learning in Fixed Non-negative Orthogonal Classifier: Inducing Zero-mean Neural Collapse with Feature Dimension Separation.

<img src="../figures/fno_imb.jpg" alt="FNO Imb">

## Experimental Details

### General Settings

We set the below settings for all experiments except for ImageNet and Places

- GPUs: Galaxy 2080TI classic x 1
- CPU cores: 16
- Memory: 128 GB
- NVIDIA Driver: 460.80
- CUDA version: 10.2

To train large models or train models on ImageNet and Places, we set the below setting.

- GPUs: A100 x 1
- CPU cores: 32
- Memory: 115 GB
- NVIDIA Driver: 450.142
- CUDA version: 11.0


## Requirements

To create an environment:

```setup
conda create -n myenv python=3.9.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train and test the model(s) in this task, run this command:

```
python main/train.py --cfg configs/imbalancecifar10/ce_imbalancecifar10_resnet32.yaml \
ddp False dp False mixed_precision False rank 0 output_dir ./output \
dataset.root /YOUR/DATA/DIRPATH \
dataset.type imbalanced \
dataset.imbalancecifar.ratio {100,50,10} \
backbone.in_channels 16 \
reshape.type FlattenNorm \
classifier.type OrthLinear \
classifier.bias False \
loss.loss_type LogLoss \
train.trainer.mixup_alpha 1.0 \
train.trainer.type mask
```


## Reference

[BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition](https://github.com/megvii-research/BBN)

[(MiSLAS) Improving Calibration for Long-Tailed Recognition](https://github.com/dvlab-research/MiSLAS)

## Contributing

T.B.A

