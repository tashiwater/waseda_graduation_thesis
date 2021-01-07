py=train_MTRNN
pyname="${py}.py"
pyen=python
for cf_num in 80 90 100
do
$pyen $pyname $cf_num 6 &
$pyen $pyname $cf_num 8 &
wait
$pyen $pyname $cf_num 10 &
$pyen $pyname $cf_num 12 &
wait
done
# open_val=0.1
# for cf_num in 80 90 100
# do
# python $pyname $cf_num 6 $open_val &
# python $pyname $cf_num 8 $open_val &
# wait
# python $pyname $cf_num 10 $open_val &
# python $pyname $cf_num 12 $open_val &
# wait
# done
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
