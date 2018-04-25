'''
main program to control the flow. data I/O
'''

import os, time
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import pprint
import codecs
import array
import collections
import io
import numpy as np
try:
    import cPickle as pickle # for Python 2
except ImportError:
    import pickle # for Python 3
import tensorflow as tf
from dataset import *
from hlstm import TextLSTM
from lstmdqn import LSTMDQN

pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("model", 'hlstm', "which deep learning model to use [hlstm]") # only hlstm currently
flags.DEFINE_integer("batch_size", 32, "training batch size [32]")
flags.DEFINE_integer("iterations", 400, "number of iterations totally [400]")
flags.DEFINE_float("learning_lr", 0.001, "init learning rate [0.01]")
flags.DEFINE_boolean("reload_word_emb", False, "reload wordembedding from GloVe or use saved one [False]")
flags.DEFINE_string("word_emb_path", "../glove.6B/glove.6B.50d.txt", "GloVe source file location")
flags.DEFINE_integer("word_emb_dim", 50, "word embedding vectors dimension [50]")
flags.DEFINE_integer("emb_dim", 50, "embedding dimension for LSTM layers [50]")
flags.DEFINE_string("train_data_path", "data/train.csv", "training data set path [data/train.csv]")
flags.DEFINE_string("test_data_path", "data/test.csv", "testing data set path [data/test.csv]")
flags.DEFINE_string("checkpoint_dir", "../checkpoints", "checkpoint directory [../checkpoints]")
flags.DEFINE_integer("class_cnt", 2, "number of classes in the dataset [2]")
flags.DEFINE_boolean("debug", False, "debug mode [False]")
flags.DEFINE_boolean("show", True, "show learning progress [True]")
FLAGS = flags.FLAGS

def load_stanford(filename):
    dct = {}
    vectors = array.array('d')

    with open(filename, 'r') as savefile:
        i = 0
        for line in savefile:
            tokens = line.split(' ')

            word = tokens[0]
            entries = tokens[1:]

            dct[word] = i
            vectors.extend(float(x) for x in entries)
            i += 1
            # Infer word vectors dimensions.

        no_components = len(entries)
        no_vectors = len(dct)
        print("Corpus stats: ", no_components, no_vectors)

        # Make these into numpy arrays
        word_vecs = np.array(vectors).reshape(no_vectors, no_components)
        inverse_dictionary = {v: k for k, v in dct.items()}
        return (word_vecs, dct, inverse_dictionary)

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    # LOAD WORD VECTORS and VOC
    if FLAGS.reload_word_emb == True:
        wordVectors, voc, voc_inv = load_stanford(FLAGS.word_emb_path)
        f = open('../wordVectors.save', 'wb')
        pickle.dump(wordVectors, f)
        f.close()
        f = open('../voc.save', 'wb')
        pickle.dump(voc, f)
        f.close()
        f = open('../voc_inv.save', 'wb')
        pickle.dump(voc_inv, f)
        f.close()
    else:
        f = open('../wordVectors.save', 'rb')
        wordVectors = pickle.load(f, encoding='latin1')
        f = open('../voc.save', 'rb')
        voc = pickle.load(f, encoding='latin1')
        f = open('../voc_inv.save', 'rb')
        voc_inv = pickle.load(f, encoding='latin1')

    print('-> wordVectors dim: ', np.shape(wordVectors))
    print('-> voc size: ', len(voc))

    # LOAD DATA
    train_data = Dataset(FLAGS.train_data_path, voc, FLAGS.batch_size, FLAGS.word_emb_dim)
    test_data = Dataset(FLAGS.test_data_path, voc, FLAGS.batch_size, FLAGS.word_emb_dim, prev_wordinsent_cnt = train_data.wordinsent_cnt)
    wordinsent_cnt = max(train_data.wordinsent_cnt, test_data.wordinsent_cnt)

    tf.reset_default_graph()

    if FLAGS.model == 'hlstm':
        with tf.Session() as sess:
            model = TextLSTM(FLAGS, sess, 25, wordinsent_cnt, FLAGS.class_cnt, len(voc), FLAGS.emb_dim, FLAGS.emb_dim, wordVectors)
            model.run(train_data, test_data)
    elif FLAGS.model == 'lstmdqn':
        with tf.Session() as sess:
            model = LSTMDQN(FLAGS, sess, train_data, wordVectors, checkpoint_dir=FLAGS.checkpoint_dir,
                    seq_length=wordinsent_cnt,
                    embed_dim=FLAGS.emb_dim,
                    layer_depth=3,
                    batch_size=FLAGS.batch_size,
                    start_epsilon=1,
                    class_cnt=FLAGS.class_cnt, wordinsent_cnt=wordinsent_cnt)
            model.train()


if __name__ == "__main__":
    tf.app.run()