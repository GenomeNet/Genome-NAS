Recently there has been a rapid development of new NAS algorithms. While earlier methods needed a vast amount of computational resources, new algorithms such as ENAS or DARTS focused on speeding up NAS algorithms. 
These automatically searched architectures showed competitive performance on image classification or objective detection tasks. However there is no research, how NAS algorithms perform on genomics data. The main contribution of this work is, to investigate how state-of-the art NAS algorithms, such as ENAS, DARTS, P-DARTS, BONAS or NASBOT/BANANAS can be used to find high-performance deep learning architectures in the field of genomics.

# What I have done so far
1. Implented genomics baseline models, based on genomics literature
2. Implemented genomicsDART

# In Progress
Implementing PDARTS for genomics

# Future Work
1. Implementing BONAS for genomics
2. Random Search
3. Hyperband


# Run baseline Models

## DeepVirFinder
```
cd baseline_models
python train_cnn_based.py --data='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt' --seq_size=150
```
## DanQ
```
cd baseline_models
python train_LSTM_based.py --data='/home/amadeu/anaconda3/envs/darts_env/cnn/data2/trainset.txt' --seq_size=150
```

# Run DARTS for Genomics
## Run search stage
```
cd genomicsDARTS
python train_search.py --unrolled
```

## Evaluate final Architecture

```
cd genomicsDARTS
python train_finalArchitecture.py --unrolled
```
