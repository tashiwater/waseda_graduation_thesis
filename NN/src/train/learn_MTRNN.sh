py=train_MTRNN_size
pyname="${py}.py"
# python $pyname 90 10 20201209_003102_3000 &
# python $pyname 90 12 20201209_003611_2600 &
# wait
# python $pyname 100 10 20201209_013113_2800 &
for cf_num in 100 110
do
for open_rate in 0.1 0.3 0.5 0.7
do
python $pyname $cf_num 6 $open_rate &
python $pyname $cf_num 8 $open_rate &
python $pyname $cf_num 10 $open_rate &
# python $pyname $cf_num 12 &
wait
done
done
# for cs_num in 8 10 12 15
# do
# python $pyname $cf_num $cs_num
#wait
# done
# python train_MTRNN_cs.py 1 1 1 1 &
# python train_MTRNN_cs.py 3 1 1 1 &
# wait
# python train_MTRNN_cs.py 2 0.5 0.5 1 &
# python train_M    TRNN_cs.py 2 0.2 0.8 1 &
# wait
# python train_MTRNN_cs.py 2 0.2 0.8 1 &
# python train_MTRNN_cs.py 2 1 1 2 &
