# Exploring Meta-tag Supported Connectivity in Large-scale Financial Knowledge Graph for Explainable SMEs Supply Chain Prediction

## Overview
This repository is the implementation of the paper entitled as Exploring Meta-tag Supported Connectivity in Large-scale Financial Knowledge Graph for Explainable SMEs Supply Chain Prediction.

![](https://github.com/LiYouru0228/MSCL/blob/main/MSCL.png?raw=true)
This is a graphical illustration of meta-tag supported connectivity representation learning for explainable SMEs supply chain prediction. It is mainly composed of three modules: (a) Meta-tag Collaborative Filtering $\textbf{(MCF)}$; (b) DPPs-induced Hierarchical Paths Sampling $\textbf{(DHPS)}$; (c)Connections Representations Learning $\textbf{(CRL)}$.

## Required packages:
The code has been tested running under Python 3.9.7, and some main following packages installed and their version are:
- PyTorch == 1.10.1
- numpy == 1.21.2
- dppy == 0.3.2
- networkx == 2.8.2
- gensim == 4.1.2
- scikit-learn == 1.0.1

## Running the code
Firstly, you can run "load_data.py" to finish the data preprocessing and this command can save the preprocessed data into some pickel files. Therefore, you only need to run it the first time.

```
$ python ./src/load_data.py
```
Then, you can start to train the model and evaluate the performance by run:
```
$ python ./src/train.py
```

## Statements
This open demo implementation is used for academic research only.
