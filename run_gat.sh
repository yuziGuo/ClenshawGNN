python train_gat.py  --dataset corafull --early-stop  --udgraph --self-loop   --loss nll  --n-cv 20 --log-detail --log-detailedCh --lr1 0.005  --wd1 5e-4  


python train_gat.py  --dataset pubmedfull --early-stop  --udgraph --self-loop   --loss nll  --n-cv 20 --log-detail --log-detailedCh --lr1 0.01  --wd1 1e-3 --out-heads 8  


python train_gat.py  --dataset geom-film --early-stop  --udgraph --self-loop   --loss nll  --n-cv 20 --log-detail --log-detailedCh --lr1 0.01  --wd1 1e-3 --out-heads 8  
