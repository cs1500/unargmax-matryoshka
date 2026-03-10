## Setup
### cvxpy
To use `unargm.py` in the scripts folder, create the following environment:
```
conda env create -f unmax.yml
conda activate unmax
```

### pytorch
Training of MRL resnets from https://github.com/RAIVNLab/MRL
was done using the following environment, on CUDA 12.2:
```
conda env create -f torch.yml
conda activate torch310_mrl
```
To fix the `CXXABI_1.3.15' not found` error, run the following in the conda
environment:
```
conda install -c conda-forge libstdcxx-ng libgcc-ng
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```
The command used to train a downscaled imagenet with 200 classes from
https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet/data
is done with the following:
```
python train_imagenet.py \
    --model.arch=resnet18 \
    --model.mrl=1 \
    --data.train_dataset=./MRL/data_run/train_500_0.50_90.ffcv \
    --data.val_dataset=./MRL/data_run/val_500_0.50_90.ffcv \
    --data.num_workers=12 \
    --data.in_memory=0 \
    --training.epochs=2 \
    --training.distributed=0 \
    --training.batch_size=8 \
    --validation.batch_size=8 \
    --logging.folder=trainlogs \
    --logging.log_level=1 \
    --dist.world_size=1 \
    --lr.lr=0.003
```