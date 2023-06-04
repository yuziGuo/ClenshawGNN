# Artifact of Clenshaw Graph Neural Networks

**[Under Construction]**

This repository includes the implementation for ClenshawGCN(KDD'23). 
By equipping GCN convolution layers with a simple residual connection, 
our model simulates any polynomial based on the second kind of Chebyshev polynomials! 

A previous version can be viewed [here](https://arxiv.org/abs/2210.16508).  A new version is coming soon!



## Install

```bash
conda env create -f environment.yml  # create a conda environment for ClenshawGNN
conda activate checkclenshaw         # enter conda environment
```

## Reproducing Results for Geom-GCN and Citation datasets

```bash
bash run_all.sh
```


## Reproducing Results for LINKX datasets

```bash
bash run_all_linkx.sh
```




