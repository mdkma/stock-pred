"""LSTM (Long Short-Term Memory) NN for tweet sentiment analysis."""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn import rnn_cell, dynamic_rnn
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

class TextLSTM(object):
    def __init__(self, sentindoc_cnt, 
                 wordinsent_cnt, 
                 class_cnt,
                 vocab_size, 
                 embedding_size, 
                 hidden_size,
                 embeddings,
                 layer_count=1, **kw):
        assert layer_count >= 1, "An LSTM cannot have less than one layer."

        self.input_x = tf.placeholder(tf.int32,
                                      [None, sentindoc_cnt, wordinsent_cnt],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, class_cnt],
                                      name="input_y")
        self.input_reg = tf.placeholder(tf.float32,
                                      [None, sentindoc_cnt, wordinsent_cnt, embedding_size],
                                      name="input_reg")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        # Layer 1: Word embeddings
        # embfild = open(emb_file, 'rb')
        # embeddings = pickle.load(embfild, encoding='bytes')
        self.embeddings = tf.Variable(embeddings)
#         self.embeddings = tf.Variable(
#             tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1),
#             name="embeddings")
        embedded_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        # Funnel the words into the LSTM.
        # Current size: (batch_size, n_words, emb_dim)
        # Want:         [(batch_size, n_hidden) * n_words]
        #
        # Since otherwise there's no way to feed information into the LSTM cell.
        # Yes, it's a bit confusing, because we want a batch of multiple
        # sequences, with each step being of 'embedding_size'.
#         embedded_words = tf.transpose(embedded_words, [2, 0, 1, 3])
        embedded_words = tf.reshape(embedded_words, [-1, wordinsent_cnt, embedding_size])
#         embedded_reg = tf.reshape(self.input_reg, [-1, wordinsent_cnt, embedding_size])
        
        # Note: 'tf.split' outputs a **Python** list.
#         embedded_words = tf.split(0, wordinsent_cnt, embedded_words)

        # Layer 2: LSTM cell
        lstm_use_peepholes = True
        # 'state_is_tuple = True' should NOT be used despite the warnings
        # (which appear as of TF 0.9), since it doesn't work on the version of
        # TF installed on Euler (0.8).
        with tf.variable_scope('lstm_cell1'):
            print("Using simple 1-layer LSTM with hidden layer size {0}."
                  .format(hidden_size))
            lstm_cells = rnn_cell.LSTMCell(num_units=hidden_size,
    #                                            input_size=embedding_size,
                                           forget_bias=1.0,
                                           use_peepholes=lstm_use_peepholes)

            lstm_cells_dropout = rnn_cell.DropoutWrapper(lstm_cells,
                                                        input_keep_prob=self.dropout_keep_prob,
                                                        output_keep_prob=self.dropout_keep_prob)
        # Q: Can't batches end up containing both positive and negative labels?
        #    Can the LSTM batch training deal with this?
        #
        # A: Yes. Each batch feeds each sentence into the LSTM, incurs the loss,
        #    and backpropagates the error separately. Each example in a bath
        #    is independent. Note that as opposed to language models, for
        #    instance, where we incur a loss for all outputs, in this case we
        #    only care about the final output of the RNN, since it doesn't make
        #    sense to classify incomplete tweets.

            outputs1, _states1 = dynamic_rnn(lstm_cells_dropout,
                                   inputs=embedded_words,
                                   dtype=tf.float32)
        
#         outputs_reg = tf.add(outputs1, embedded_reg, 'output_with_reg')
        
        # pooling
        poollayer = tf.reduce_mean(outputs1, axis=1)

        output_restore = tf.reshape(poollayer, [-1, sentindoc_cnt, embedding_size])

        with tf.variable_scope('lstm_cell2'):
        # Layer 2: LSTM cell
            lstm_cells2 = rnn_cell.LSTMCell(num_units=hidden_size,
    #                                            input_size=embedding_size,
                                           forget_bias=1.0,
                                           use_peepholes=lstm_use_peepholes)
            lstm_cells2_dropout = rnn_cell.DropoutWrapper(lstm_cells2,
                                                        input_keep_prob=self.dropout_keep_prob,
                                                        output_keep_prob=self.dropout_keep_prob)

            outputs2, _states2 = dynamic_rnn(lstm_cells2_dropout,
                                   inputs=output_restore,
                                   dtype=tf.float32)
        
        outputs = tf.reduce_mean(outputs2, axis=1)

        # Layer 3: Final Softmax
        out_weight = tf.Variable(tf.random_normal([hidden_size, class_cnt]))
        out_bias = tf.Variable(tf.random_normal([class_cnt]))

        with tf.name_scope("output"):
            lstm_final_output = outputs
            self.scores = tf.nn.xw_plus_b(lstm_final_output, out_weight,
                                          out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores,
                                                                  labels = self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.predictlabel = tf.argmax(self.predictions, 1)
            self.truelabel = tf.argmax(self.input_y, 1)
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),
                                         tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),
                                           name="accuracy")

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    
    def run(self, train_data, test_data):
            