
# python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 2 --kw clengnn2layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw2.log 2> optunalogKlayers/sq-clenshaw2.err 
# sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 4 --kw clengnn4layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw4.log 2> optunalogKlayers/sq-clenshaw4.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 6 --kw clengnn6layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw6.log 2> optunalogKlayers/sq-clenshaw6.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 8 --kw clengnn8layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw8.log 2> optunalogKlayers/sq-clenshaw8.err 

# python tune_with_optuna_clenshaw.py --dataset pubmedfull --n-layers 2 --kw clengnn2layers --gpu 0  --id-log 0410017803 1> optunalogKlayers/pub-clenshaw2.log 2> optunalogKlayers/pub-clenshaw2.err 
# sleep 3
# python tune_with_optuna_clenshaw.py --dataset pubmedfull --n-layers 4 --kw clengnn4layers --gpu 0  --id-log 0410017803 1> optunalogKlayers/pub-clenshaw4.log 2> optunalogKlayers/pub-clenshaw4.err 
# sleep 3
# python tune_with_optuna_clenshaw.py --dataset pubmedfull --n-layers 6 --kw clengnn6layers --gpu 0  --id-log 0410017803 1> optunalogKlayers/pub-clenshaw6.log 2> optunalogKlayers/pub-clenshaw6.err 
# sleep 3
# python tune_with_optuna_clenshaw.py --dataset pubmedfull --n-layers 8 --kw clengnn8layers --gpu 0  --id-log 0410017803 1> optunalogKlayers/pub-clenshaw8.log 2> optunalogKlayers/pub-clenshaw8.err 