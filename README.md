# Stock Trend Prediction by Hierarchical LSTM Network

## Get Started

1. Download stock training data [from Kaggle](https://www.kaggle.com/aaron7sun/stocknews)

2. Divide the file into `test.csv`, `train.csv` files and put them outside this folder

3. [Download `GloVe`](https://nlp.stanford.edu/projects/glove/) and set corresponding path in the flag of this program.

4. In your terminal, type

```
python main.py --FLAGNAME FLAGVALUE
```

All the flags can be found at the beginning of `main.py`:

```
flags.DEFINE_string("model", 'hlstm', "which deep learning model to use [hlstm]") # only hlstm currently
flags.DEFINE_integer("batch_size", 32, "training batch size [32]")
flags.DEFINE_integer("iterations", 400, "number of iterations totally [400]")
flags.DEFINE_float("learning_lr", 0.001, "init learning rate [0.01]")
flags.DEFINE_boolean("reload_word_emb", False, "reload wordembedding from GloVe or use saved one [False]")
flags.DEFINE_string("word_emb_path", "../glove.6B/glove.6B.50d.txt", "GloVe source file location")
flags.DEFINE_integer("word_emb_dim", 50, "word embedding vectors dimension [50]")
flags.DEFINE_integer("emb_dim", 50, "embedding dimension for LSTM layers [50]")
flags.DEFINE_string("train_data_path", "../train.csv", "training data set path [../train.csv]")
flags.DEFINE_string("test_data_path", "../test.csv", "testing data set path [../test.csv]")
flags.DEFINE_string("checkpoint_dir", "../checkpoints", "checkpoint directory [../checkpoints]")
flags.DEFINE_integer("class_cnt", 2, "number of classes in the dataset [2]")
flags.DEFINE_boolean("debug", False, "debug mode [False]")
flags.DEFINE_boolean("show", True, "show learning progress [True]"
```

## Environment

* Python3
* TensorFlow
* pip installable packages: `numpy`, `pandas`, `progress`

## Author

[Mingyu MA](http://derek.ma)

<!--## Acknowledgement

Some part of the code for LSTM+DQN model is taken from [carpedm20's work](https://github.com/carpedm20/text-based-game-rl-tensorflow). LSTM+DQN model code also references the presentation of author and his original code.-->

