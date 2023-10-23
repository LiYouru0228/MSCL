# Exploring Large-scale Financial Knowledge Graph for SMEs Supply Chain Mining

## Overview
This repository is the implementation of the paper entitled as Exploring Large-scale Financial Knowledge Graph for SMEs Supply Chain Mining. ([TKDE'23](https://ieeexplore.ieee.org/abstract/document/10256685))
> Youru Li, Zhenfeng Zhu, Linxun Chen, Bin Yang, Yaxi Wu, Xiaobo Guo, Bing Han, Yao Zhao: Exploring Large-scale Financial Knowledge Graph for SMEs Supply Chain Mining. IEEE Transactions on Knowledge and Data Engineering (2023).

![](https://github.com/LiYouru0228/MSCL/blob/main/MSCL.png?raw=true)

This is a graphical illustration of meta-tag supported connectivity representation learning for SMEs supply chain mining. It is mainly composed of three modules: (a) Meta-tag Collaborative Filtering $\textbf{(MCF)}$; (b) DPPs-induced Hierarchical Paths Sampling $\textbf{(DHPS)}$; (c)Connectivity Representation Learning $\textbf{(CRL)}$.

## Required packages:
The code has been tested by running a demo pipline under Python 3.9.7, and some main following packages installed and their version are:
- PyTorch == 1.10.1
- numpy == 1.21.2
- dppy == 0.3.2
- networkx == 2.8.2
- gensim == 4.1.2
- scikit-learn == 1.0.1

## Running the code
Firstly, you can run "load_data.py" to finish the data preprocessing and this command can save the preprocessed data into some pickel files. Noted, you only need to run it the first time.

```
$ python ./src/load_data.py
```
Then, you can start to train the model and evaluate the performance by run:
```
$ python ./src/train.py
```

## Citation 
If you want to use our codes in your research, please cite:
```
@article{li2023exploring,
  title={Exploring Large-scale Financial Knowledge Graph for SMEs Supply Chain Mining},
  author={Li, Youru and Zhu, Zhenfeng and Chen, Linxun and Yang, Bin and Wu, Yaxi and Guo, Xiaobo and Han, Bing and Zhao, Yao},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```

## Statements
This open demo implementation is used for academic research only.
