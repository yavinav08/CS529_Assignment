## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.


## Getting started

To reproduce the results, simply run:

```.bash
python example.py
```

To generate a batch of images using a given model and sampler, run:

```.bash
# Generate 64 images and save them as out/*.png
python generate.py --outdir=out --seeds=0-63 --batch=64 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

Generating a large number of images can be time-consuming; the workload can be distributed across multiple GPUs by launching the above command using `torchrun`:

```.bash
# For CIFAR-10 at 32x32, use deterministic sampling with 18 steps (NFE = 35)
python generate.py --outdir=out --steps=18 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

## Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `fid.py`:

```.bash
# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

# Calculate FID
torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

## Preparing datasets

Custom datasets can be created from a folder containing images

**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

## Training new models

You can train new models using `train.py`. For example:

```.bash
# Train DDPM++ model for class-conditional CIFAR-10
torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
```