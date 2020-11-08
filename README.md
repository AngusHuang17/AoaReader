# AoaReader

This is a simple implementation of paper [*Attention-over-Attention Neural Networks for Reading Comprehension*](https://arxiv.org/abs/1607.04423). We only use CNN news database to train and test the model. Also, the implementation doesn't consist of *N-best Re-ranking Strategy* in the paper.


## About the folder structure

```
AoaReader
├─
│  .gitignore
│  pre_train.py
│  README.md
│  test.ipynb
│  test.py
│  train.py
│  
├─.vscode
│      settings.json
│      
├─data
│      Readme.md
│      
├─model
│  │  model.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          model.cpython-36.pyc
│          __init__.cpython-36.pyc
│          
├─temp
│      dictionary.pickle
│      Readme.md
│      test_vec.pickle
│      train_vec.pickle
│      valid_vec.pickle
│      
└─utils
    │  dataloader.py
    │  dict.py
    │  __init__.py
    │  
    └─__pycache__
            dataloader.cpython-36.pyc
            dataloader.cpython-38.pyc
            dict.cpython-36.pyc
            dict.cpython-38.pyc
            __init__.cpython-36.pyc
            __init__.cpython-38.pyc
```

- AoaReader
  - data : consists of the dataset.
    - cnn : download from [here](https://cs.nyu.edu/~kcho/DMQA/)
  - model : consists of the model AoaReader.
    - model.py
  - utils : consists of some tool class the deal with data.
    - dict.py
    - dataloader.py
  - temp : save some temporary files, like the trained model, the prepared data.
  - pre_train.py : deal with the origin data
  - train.py : train the model
  - test.py : predict on the test data with the trained model


## How to run

1. Download the dataset from [here](https://cs.nyu.edu/~kcho/DMQA/).
2. Run `rm ./temp/*.pickle` to delete the temp files generate by us. Also, you can choose to use them instead of deleting. (It will save several minutes to deal with the data). Attention, if you choose to use our temp files, you can skip step 1 but you should change some lines in the `pre_train.py` to stop accessing dataset files.
3. Run `python pre_train.py`.
4. Run `python train.py`.
5. Run `python test.py`.
