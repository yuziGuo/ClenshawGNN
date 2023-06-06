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

Run the scripts altogether: 
```bash
bash scripts/run_all.sh
```
or run one command in the script at a time.


## Reproducing Results for LINKX datasets

ClenshawGCN achieves state-of-the-art results on genius and twitch-gamer.
The results are as below; they will be released with the KDD'23 paper.
<img src="./scripts/clenshawlinkx.jpg" alt="Linkx" width="500" height="300">

```bash
bash scripts/run_all_linkx.sh
```




