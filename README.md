## Introduction

Recently there has been a rapid development of new NAS algorithms. While earlier methods needed a vast amount of computational resources, new algorithms such as ENAS or DARTS focused on speeding up NAS algorithms. 
These automatically searched architectures showed competitive performance on image classification or objective detection tasks. However there is no research, how NAS algorithms perform on genomics data. The main contribution of this work is, to investigate how state-of-the art NAS algorithms, such as DARTS, P-DARTS, BONAS , Hyperband and random search can be used to find high-performance deep learning architectures in the field of genomics.


## Tested with

- Python 3.6
- Ubuntu 20.04
- torch 1.7.1+cu101

## Setup
```
conda create --name genomeNAS python=3.6

source activate genomeNAS

git clone https://github.com/ascheppach/GenomNet_MA.git

cd GenomNet_MA

export PYTHONPATH="$PYTHONPATH:~/supervisor"
```

## Data


## Run baseline Models

### DeepSEA
Run following Code to run a LSTM based basline model such as DanQ. Of course, you have to adapt the path of your data.
```
cd baseline_models
python train_LSTM_based.py --batch_size=32 --seq_size=150 --epochs=30 --data=/home/ascheppa/miniconda2/envs/GenomNet_MA/data/trainset.txt
```
### DanQ
```
cd baseline_models
python train_LSTM_based.py --data='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt' --seq_size=150
```

## Run NAS algorithms for Genomics
### genomicDARTS
```
cd genomicNAS_Algorithms

python train_genomicDARTS.py --num_steps=3000 --seq_len=100 --batch_size=38 --train_directory='/home/ascheppa/miniconda2/envs/GenomNet_MA/genomicData/train' --valid_directory='/home/ascheppa/miniconda2/envs/GenomNet_MA/genomicData/validation' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=2 --num_files=150 --one_clip=True --clip=0.25 --validation=False --report_validation=1000 --epochs=10

```
