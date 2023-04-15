# Artifact of ClenshawGNN

## Install

```bash
conda env create -f environment.yml  # create a conda environment for ClenshawGNN
conda activate checkclenshaw         # enter conda environment
```

## Reproduce Results for Table 2

```bash
bash run_all.sh
```


## Reproduce Results for Table 3

```bash
bash run_all_linkx.sh
```

### Log files for rebuttal stage

```md
rebuttalLogs/
├── results_Amazon@ReviewerRqQP_TableR1
│   ├── computer.log
│   └── photo.log
├── results_gat@ReviewerRqQP_TableR1
│   ├── cham.log
│   ├── pub.log
│   ├── README.md
│   └── sq.log
├── results_gcniiBasedJK@Reviewer3158_TableR1_M1
│   ├── gcniijk_cham.log
│   ├── gcniijk_cora.log
│   ├── gcniijk_pub.log
│   └── gcniijk_sq.log
├── results_gcniiEnsemblesJK@Reviewer3158_TableR1_M2
│   ├── cham.log
│   ├── cora.log
│   ├── pub.log
│   ├── run.sh
│   └── sq.log
├── results_K=10Ablation@ReviewerekCQ_TableR1
│   ├── all_K=10.log
│   ├── README.md
│   └── run.sh
└── results_KAblation@ReviewerekCQ_TableR2
    ├── allK_pub.log
    └── allK_sq.log
```



