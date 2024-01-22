# Fixed Non-negative Orthogonal Classifier: Inducing Zero-mean Neural Collapse with Feature Dimension Separation

This directory is the official implementation of the part of Neural Collapse Experiments in Fixed Non-negative Orthogonal Classifier: Inducing Zero-mean Neural Collapse with Feature Dimension Separation.

<img src="figures/fno_nc.jpg" alt="FNO NC">

## Experimental Details

### General Settings

We set the below settings for all experiments

- GPUs: Galaxy 2080TI classic x 1
- CPU cores: 16
- Memory: 128 GB
- NVIDIA Driver: 460.80
- CUDA version: 10.2

To train large models, we set the below setting.

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
python main/nc_train_tr.py --cfg configs/mnist/ce_mnist_vgg.yaml \
ddp False dp False mixed_precision False rank 0 output_dir ./output \
dataset.root /YOUR/DATA/DIRPATH \
train.optimizer.base_lr 0.06786
```

## Reference

[Prevalence of neural collapse during the terminal phase of deep learning training](https://www.pnas.org/doi/10.1073/pnas.2015509117)

[BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition](https://github.com/megvii-research/BBN)

## Contributing

T.B.A

