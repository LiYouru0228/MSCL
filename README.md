# MSCL: Exploring Meta-tag Supported Connectivity in Large-scale Financial Knowledge Graph for Explainable SMEs Supply Chain Prediction

## Overview
This repository is the implementation of the paper entitled as MSCL: Exploring Meta-tag Supported Connectivity in Large-scale Financial Knowledge Graph for Explainable SMEs Supply Chain Prediction.

![](https://github.com/LiYouru0228/MSCL/blob/main/MSCL.png?raw=true)
This is a graphical illustration of meta-tag supported connectivity representation learning for explainable SMEs supply chain prediction. It is mainly composed of three modules: $\textbf{(a) Meta-tag Collaborative Filtering (MCF)}$; (b) DPPs-induced Hierarchical Paths Sampling (DHPS); (c) Connections Representations Learning (CRL).

### Required packages:
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
$ python load_data.py
```
Then, you can start to train the model and evaluate the performance by run:
```
$ python train.py
```

## Statements
This open demo implementation is used for academic research only and not represent any real business situation in MYBank, Ant Group.
