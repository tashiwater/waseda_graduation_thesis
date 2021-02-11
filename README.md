# waseda_graduation_thesis
## Requirements
### python3
- pip install torch torchvision pandas matplotlib sklearn openpyxl torchsummary pyyaml xlrd

### python2
- pip install future
- pip install pathlib

## Run
You need to rewrite parameters and paths in *.py before you execute them.   
```
cd waseda_graduation_thesis
```
### CAE
1. make your data directories(all, result, test, train) in NN/data/CAE/{date}/{name}
1. pyenv global 2.7.12
1. python preprocess/src/img_preprocess.py
1. pyenv global 3.8.5 
1. python NN/src/train/train_CAE.py
1. python NN/src/test/test_CAE.py
1. python NN/src/test/CAE_dump_hidden.py

### MTRNN
1. make your data directories(result, test, train) in NN/data/MTRNN/{date}/{name}
1. cp {your tactile_raw} preprocess/data/connect_input/tactile_raw
1. python preprocess/src/yaml2csv.py
1. python preprocess/src/check_position_range.py
1. python preprocess/src/cal_imgnum.py
1. python preprocess/src/connect_datas_{your model}.py
1. python NN/src/train/train_MTRNN_{your model}.py
1. python NN/src/test/test_MTRNN_{your model}.py

### Cs0maker
1. make your data directories (test, train) in NN/data/cs0_maker/{date}
1. make datas in train, test like NN/data/cs0_maker/0106
1. python NN/src/train/train_cs0_maker.py





