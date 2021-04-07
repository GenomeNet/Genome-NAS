Recently there has been a rapid development of new NAS algorithms. While earlier methods needed a vast amount of computational resources, new algorithms such as ENAS or DARTS focused on speeding up NAS algorithms. 
These automatically searched architectures showed competitive performance on image classification or objective detection tasks. However there is no research, how NAS algorithms perform on genomics data. The main contribution of this work is, to investigate how state-of-the art NAS algorithms, such as ENAS, DARTS, P-DARTS, BONAS or NASBOT/BANANAS can be used to find high-performance deep learning architectures in the field of genomics.

# What I have done so far
1. Implented genomics baseline models, based on genomics literature
2. Implemented genomicsDART
3. Implemented genomicsPDARTS


# In Progress
1. Implementing BONAS for genomics

# Future Work
2. Random Search
3. Hyperband


# Run baseline Models

## DeepVirFinder
Run following Code to run a LSTM based basline model such as DanQ. Of course, you have to adapt the path of your data.
```
cd baseline_models
python train_LSTM_based.py --batch_size=32 --seq_size=150 --epochs=30 --data=/home/ascheppa/miniconda2/envs/GenomNet_MA/data/trainset.txt
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
