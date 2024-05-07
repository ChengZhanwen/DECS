# Deep Embedding Clustering Driven by Sample Stability (DECS)

## Abstract
Deep clustering methods improve the performance of clustering tasks by jointly optimizing deep representation learning and clustering. While numerous deep clustering algorithms have been proposed, most of them rely on artificially constructed pseudo targets for performing clustering. This construction process requires some prior knowledge, and it is challenging to determine a suitable pseudo target for clustering. To address this issue, we propose a deep embedding clustering algorithm driven by sample stability (DECS), which eliminates the requirement of pseudo targets. Specifically, we start by constructing the initial feature space with an autoencoder and then learn the cluster-oriented embedding feature constrained by sample stability. The sample stability aims to explore the deterministic relationship between samples and all cluster centroids, pulling samples to their respective clusters and keeping them away from other clusters with high determinacy. We analyzed the convergence of the loss using Lipschitz continuity in theory, which verifies the validity of the model. The experimental results on five datasets illustrate that the proposed method achieves superior performance compared to state-of-the-art clustering approaches.


Tensorflow implementation for IJCAI-2024 paper:
* Zhanwen Cheng, Feijiang Li, Jieting Wang, Yuhua Qian. Deep Embedding Clustering Driven by Sample Stability. 
The 33rd International Joint Conference on Artificial Intelligence (IJCAI), 2024.

State-of-the-art clustering performance on:
- MNIST (acc=0.990, nmi=0.973) 
- MNIST-TEST (acc=0.990, nmi=0.971) 
- USPS (acc=992, nmi=976)
- Fashion-MNIST (acc=0.642, nmi=0.716)
- YTF (acc=0.827, nmi=0.911)


## Usage

### 1. Prepare environment

Install [Anaconda](https://www.anaconda.com/download/) with Python 3.6 version (_Optional_).   
Create a new env (_Optional_):   
```
conda create -n decs python=3.7
source activate decs  # Linux 
#  or 
conda activate decs  # Windows
```
Install required packages:
```
pip install scipy scikit-learn h5py tensorflow-gpu==1.10  
```

### 2. Run experiments.    

Quick test (_Optional_):
```bash
python run_exp.py --trials 1 --pretrain-epochs 1 --maxiter 150
```
Reproduce the results in Table 1 in the paper (this may take a long time):
```bash
python run_exp.py
```

