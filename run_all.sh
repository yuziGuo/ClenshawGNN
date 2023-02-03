# # [film]
python train_clenshaw.py  --dataset geom-film --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.0 --dropout2 0.0 --lamda 2.0 --lr1 0.04 --lr2 0.05 --lr3 0.03 --momentum 0.95 --n-layers 20 --wd1 1e-8 --wd2 1e-3 --wd3 1e-5 

# # [squirrel]
python train_clenshaw.py  --dataset geom-squirrel --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.5 --dropout2 0.1 --lamda 1.0 --lr1 0.01 --lr2 0.04 --lr3 0.05 --momentum 0.95 --n-layers 8 --wd1 1e-4 --wd2 1e-5 --wd3 1e-8 

# [chameleon]
python train_clenshaw.py  --dataset geom-chameleon --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.6 --dropout2 0.0 --lamda 1.0 --lr1 0.01 --lr2 0.02 --lr3 0.01 --momentum 0.95 --n-layers 8 --wd1 1e-5 --wd2 1e-3 --wd3 1e-7 

# [corafull]
python train_clenshaw.py  --dataset corafull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.7 --dropout2 0.0 --lamda 1.5 --lr1 0.01 --lr2 0.04 --lr3 0.03 --momentum 0.8 --n-layers 8 --wd1 1e-8 --wd2 1e-3 --wd3 1e-4 

# [pubmedfull]
python train_clenshaw.py  --dataset pubmedfull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.1 --dropout2 0.1 --lamda 1.0 --lr1 0.02 --lr2 0.03 --lr3 0.05 --momentum 0.85 --n-layers 16 --wd1 1e-4 --wd2 1e-3 --wd3 1e-7 

# [citeseerfull]
python train_clenshaw.py  --dataset citeseerfull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.7 --dropout2 0.2 --lamda 1.5 --lr1 0.05 --lr2 0.005 --lr3 0.005 --momentum 0.8 --n-layers 28 --wd1 1e-6 --wd2 1e-7 --wd3 1e-5 

# [cornell]
python train_clenshaw.py  --dataset geom-cornell --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.2 --dropout2 0.5 --lamda 2.0 --lr1 0.01 --lr2 0.05 --lr3 0.02 --momentum 0.8 --n-layers 16 --wd1 1e-3 --wd2 1e-4 --wd3 1e-3 

# [texas]
python train_clenshaw.py  --dataset geom-texas --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.2 --dropout2 0.6 --lamda 1.0 --lr1 0.001 --lr2 0.05 --lr3 0.05 --momentum 0.85 --n-layers 8 --wd1 1e-3 --wd2 1e-3 --wd3 1e-3