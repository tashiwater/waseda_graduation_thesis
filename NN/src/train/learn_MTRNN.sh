for cf_num in 70 80 90 100
do
python train_MTRNN.py $cf_num 8 &
python train_MTRNN.py $cf_num 10 &
python train_MTRNN.py $cf_num 12 &
python train_MTRNN.py $cf_num 15 &
wait
done


