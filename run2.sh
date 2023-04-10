
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 8 --kw clengnn8layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw8.log 2> optunalogKlayers/sq-clenshaw8.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 12 --kw clengnn12layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw12.log 2> optunalogKlayers/sq-clenshaw12.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 16 --kw clengnn16layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw16.log 2> optunalogKlayers/sq-clenshaw16.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 20 --kw clengnn20layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw20.log 2> optunalogKlayers/sq-clenshaw20.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 24 --kw clengnn24layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw24.log 2> optunalogKlayers/sq-clenshaw24.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 28 --kw clengnn28layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw28.log 2> optunalogKlayers/sq-clenshaw28.err 
sleep 3
python tune_with_optuna_clenshaw.py --dataset geom-squirrel --n-layers 32 --kw clengnn32layers --gpu 1  --id-log 0410017803 1> optunalogKlayers/sq-clenshaw32.log 2> optunalogKlayers/sq-clenshaw32.err 
sleep 3