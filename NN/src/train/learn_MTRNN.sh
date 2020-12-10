py=train_MTRNN
pyname="${py}.py"
# python $pyname 90 10 20201209_003102_3000 &
# python $pyname 90 12 20201209_003611_2600 &
# wait
# python $pyname 100 10 20201209_013113_2800 &
for cf_num in 70 80 90 100
do
python $pyname $cf_num 6 &
python $pyname $cf_num 8 &
wait
python $pyname $cf_num 10 &
python $pyname $cf_num 12 &
wait
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
