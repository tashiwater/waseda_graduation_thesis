python /home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/NN/src/train/train_CAE.py
wait
python /home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/preprocess/src/connect_datas_onlyimg.py &
python /home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/preprocess/src/connect_datas_all.py &
wait
python /home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/NN/src/train/train_MTRNN.py onlyimg 22 &
python /home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/NN/src/train/train_MTRNN.py all 45

