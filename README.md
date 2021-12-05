## Introduction
The provided framework enables the user to benchmark Neural Architecture Search (NAS) algorithms on genomic sequences, using the DeepSEA data. There is little research on how NAS algorithms perform on genomic sequences, because most of the algorithms are applied on image classification, object detection or semantic segmentation. Due to this lack of research, we investigate how state-of-the art NAS algorithms, such as DARTS, Progressive Differentiable Architecture Search (P-DARTS) and Bayesian Optimized NeuralArchitecture Search (BONAS) can be used to find high-performance deep learning architectures in the field of genomics. Our search space combines convolutional and recurrent layer because popular genome models such as DanQ or NCNet also used hybrid models, which consist of convolutional layers together with recurrent layers. These hybrid models showed superior performance compared to pure convolution neural network architectures. We implement new NAS approaches with a new search space which includes CNN and Recurrent Neural Network (RNN) operations. We call these algorithms genomeDARTS, genomeP-DARTS, and genomeBONAS. Furthermore, we build novel DARTS algorithms such as CWP-DARTS, which enables continuous weight sharing across different P-DARTS stages by transferring the neural network weights and architecture weights between P-DARTS iterations. In another P-DARTS extension,  we discard not only bad performing operations but also bad performing edges.  Additionally, we implement an algorithm, which we call OSP-NAS. OSP-NAS starts with a super-network model which includes all randomly sampled operations and edges and then gradually discard randomly sampled operations based on the validation accuracy of the remaining super-network. 


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

export PYTHONPATH="$PYTHONPATH:~/GenomNet_MA"
```

## Data
To get the data for the predcion of noncoding variants please follow the instuctions of the DeepSea authors and download the data from their website.


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
Befor running a genomeNAS algorithm the user has to create a folder where the results get stored.

### genomeDARTS
```
cd genomicNAS_Algorithms

mkdir darts_search

python train_genomicDARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --epochs=50 --task='TF_bindings' --save='darts_1' --save_dir=darts_search
```

### Hyperband-NAS

```
cd genomicNAS_Algorithms

mkdir hb_search

python train_genomicHyperbandNAS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --epochs=20 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --budget=3 --num_samples=25 --iterations=3--task='TF_bindings'  --save='hb_1' --save_dir=hb_search
```

### genomeP-DARTS

```
cd genomicNAS_Algorithms

mkdir pdarts_search

python train_genomicPDARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --epochs=25 --task='TF_bindings' --save='pdarts_1' --save_dir=pdarts_search
```

### genomeOSP-NAS

```
cd genomicNAS_Algorithms

mkdir osp_search

python train_genomicOSP_NAS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --epochs=60 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --budget=5 --pretrain_epochs=10 --num_init_archs=108 --num_ops=7 --task='TF_bindings' --save='osp_1' --save_dir=osp_search
```

### Random search

```
cd genomicNAS_Algorithms

mkdir random_search

python train_genomicRandom.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --num_samples=20 --epochs=7 --task='TF_bindings' --save='random_1' --save_dir=random_search

```

### genomeCWP-DARTS

```
cd genomicNAS_Algorithms

mkdir cwp_search

python train_genomicCWP_DARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --task='TF_bindings' --save='cwp_1' --save_dir=cwp_search
```


### genomeDEP-DARTS

```
cd genomicNAS_Algorithms

mkdir dep_search

python train_genomicDEP_DARTS.py --num_steps=2000 --seq_size=1000 --batch_size=64 --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25 --validation=True --report_validation=1 --task='TF_bindings' --save='dep_1' --save_dir=dep_search
```

### genomeBONAS

```
cd genomicNAS_Algorithms

mkdir bonas_search

python train_genomicBONAS.py --num_steps=200 --seq_size=1000  --train_directory='/home/ascheppa/deepsea/train.mat' --valid_directory='/home/ascheppa/deepsea/valid.mat' --test_directory='/home/ascheppa/deepsea/test.mat' --report_freq=1000 --dropouth=0.05 --dropoutx=0.1 --rhn_lr=8 --one_clip=True --clip=0.25  --super_batch_size=64 --sub_batch_size=128 --generate_num=1000 --bo_sample_num=60 --epochs=60 --train_supernet_epochs=15 --iterations=2 --gcn_epochs=100 --sample_method='ea' --task='TF_bindings' --save='bonas_1' --save_dir=bonas_search
```
