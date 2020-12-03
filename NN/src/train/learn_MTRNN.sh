for cf_num in 80 90 100
do
python train_MTRNN_cs.py $cf_num 8 &
python train_MTRNN_cs.py $cf_num 10 &
wait
python train_MTRNN_cs.py $cf_num 12 &
python train_MTRNN_cs.py $cf_num 15 &
wait
done
# python train_MTRNN_cs.py 1 1 1 1 &
# python train_MTRNN_cs.py 3 1 1 1 &
# wait
# python train_MTRNN_cs.py 2 0.5 0.5 1 &
# python train_M    TRNN_cs.py 2 0.2 0.8 1 &
# wait
# python train_MTRNN_cs.py 2 0.2 0.8 1 &
# python train_MTRNN_cs.py 2 1 1 2 &
