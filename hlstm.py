"""LSTM (Long Short-Term Memory) NN for tweet sentiment analysis."""
import datetime, os
import tensorflow as tf
import numpy as np
import pprint
from tensorflow.python.ops.nn import rnn_cell, dynamic_rnn
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

pp = pprint.PrettyPrinter()

class TextLSTM(object):
    def __init__(self, config, sess, sentindoc_cnt, 
                 wordinsent_cnt, 
                 class_cnt,
                 vocab_size, 
                 embedding_size, 
                 hidden_size,
                 embeddings,
                 layer_count=1, **kw):
        assert layer_count >= 1, "An LSTM cannot have less than one layer."
        self.sess = sess
        self.iterations = config.iterations
        self.debug = config.debug
        self.learning_lr = config.learning_lr
        self.current_lr = self.learning_lr
        self.class_cnt = config.class_cnt
        self.show = config.show
        self.log_dir = '../log/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        self.checkpoint_dir = config.checkpoint_dir

        self.input_x = tf.placeholder(tf.int32,
                                      [None, None],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, class_cnt],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float64,
                                                name="dropout_keep_prob")

        # Layer 1: Word embeddings
        self.embeddings = tf.Variable(embeddings)
        embedded_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)
        embedded_words = tf.reshape(embedded_words, [-1, wordinsent_cnt, embedding_size])
        

        # Layer 2: LSTM cell
        lstm_use_peepholes = True
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
            outputs1, _states1 = dynamic_rnn(lstm_cells_dropout,
                                   inputs=embedded_words,
                                   dtype=tf.float64)
        
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
                                   dtype=tf.float64)
        
        outputs = tf.reduce_mean(outputs2, axis=1)

        # Layer 3: Final Softmax
        out_weight = tf.Variable(tf.random_normal([hidden_size, class_cnt]))
        out_bias = tf.Variable(tf.random_normal([class_cnt]))

        with tf.name_scope("output"):
            lstm_final_output = outputs
            out_weight = tf.cast(out_weight, tf.float64)
            out_bias = tf.cast(out_bias, tf.float64)
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
        self.optim = tf.train.AdamOptimizer(self.current_lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, inputs):
        if self.debug:
            train_size = 20
        else:
            train_size = inputs.epoch
        batchList = np.random.randint(inputs.epoch, size = train_size)
        n = 0
        loss_total = 0
        accuracy_total = 0

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('train', max=train_size)

        for i in batchList:
            nextBatchData = np.array(inputs.docs[i]).astype(np.int32).transpose()
            # Make the labels one-hot
            labels_temp = inputs.label[i]
            nextBatchLabels = np.array(np.eye(self.class_cnt)[labels_temp], dtype=np.float64)
            _, _loss, _accuracy, _prediction = self.sess.run(
                [self.optim, self.loss, self.accuracy, self.predictlabel],
                feed_dict={
                    self.input_x: nextBatchData,
                    self.input_y: nextBatchLabels,
                    self.dropout_keep_prob: 1.0
                })
            loss_total += _loss
            accuracy_total += _accuracy
            n += 1
            if self.show: bar.next()
        if self.show: bar.finish()
        return loss_total/train_size, accuracy_total/train_size

    def test(self, inputs):
        if self.debug:
            test_size = 20
        else:
            test_size = inputs.epoch
        batchList = np.random.randint(inputs.epoch, size = test_size)
        n = 0
        loss_total = 0
        accuracy_total = 0

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('test ', max=test_size)

        for i in batchList:
            nextBatchData = np.array(inputs.docs[i]).astype(np.int32).transpose()
            # Make the labels one-hot
            labels_temp = inputs.label[i]
            nextBatchLabels = np.array(np.eye(self.class_cnt)[labels_temp], dtype=np.float64)
            _loss, _accuracy, _prediction = self.sess.run(
                [self.loss, self.accuracy, self.predictlabel],
                feed_dict={
                    self.input_x: nextBatchData,
                    self.input_y: nextBatchLabels,
                    self.dropout_keep_prob: 1.0
                })
            loss_total += _loss
            accuracy_total += _accuracy
            n += 1
            if self.show: bar.next()
        if self.show: bar.finish()
        return loss_total/test_size, accuracy_total/test_size
    
    def run(self, train_data, test_data):
        print("-------> Training start. Total epoch: %s" % train_data.epoch)
        bestAccuracy = 0
        bestIteration = 0

        record_file = open(self.log_dir,'w')
        record_file.write('epoch,train_loss,train_accuracy,test_loss,test_accuracy,bestAccuracy,bestIterations\n')
        record_file.close()

        for idx in range(self.iterations):
            print ('-> iterations: %s' % idx)
            train_loss, train_accuracy = self.train(train_data)
            test_loss, test_accuracy = self.test(test_data)
            state = {
                '         epoch': idx,
                '    train_loss': train_loss,
                'train_accuracy': train_accuracy,
                '     test_loss': test_loss,
                ' test_accuracy': test_accuracy,
                '  bestAccuracy': bestAccuracy,
                ' bestIteration': bestIteration
            }
            pp.pprint(state)
            # self.test(test_data)
            with open(self.log_dir, "a") as record_file:
                record_file.write('%s,%s,%s,%s,%s,%s,%s\n' % (idx, train_loss, train_accuracy, test_loss, test_accuracy, bestAccuracy, bestIteration))

            if test_accuracy > bestAccuracy and idx < 6:
                bestAccuracy = test_accuracy

            if test_accuracy > bestAccuracy and idx > 5:
                bestAccuracy = test_accuracy
                bestIteration = idx
                self.saver.save(self.sess,
                                os.path.join(self.checkpoint_dir, "Network.model"))
                print('--> model saved: better accuracy!')
            