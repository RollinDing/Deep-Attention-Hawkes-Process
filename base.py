import sys
import arrow
import numpy as np
import tensorflow as tf

import utils

"""
Base Deep Attention Hawkes Model

Dependencies:
- Python==3.7.1
- tensorflow==1.12.0

References:
- weights initialization: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78.
"""

class BaseDeepAttentionHawkes():
    """
    Deep Attention Hawkes Base Model
    """

    def __init__(self, model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers, data_dim=1):
        """
        Args:
        - model_name:   the name of the model
        - max_seq_len:  maximun length of input sequences
        - n_heads:      the number of heads in multi-head attention
        - key_size:     the size of key vectors
        - score_layers: the size of each layer in score function
        - lam_layers:   the size of each layer in lambda function
        """
        # model configuration
        self.data_dim     = data_dim
        self.max_seq_len  = max_seq_len
        self.key_size     = key_size          # length of key vectors
        self.n_heads      = n_heads           # number of attention heads
        self.score_layers = score_layers
        self.lam_layers   = lam_layers
        # weights for key projection
        self.key_weights = []
        for h in range(n_heads):
            weight = tf.get_variable(
                name='%s_keyweights_head%d' % (model_name, h), 
                # shape=(self.data_dim, self.key_size), 
                dtype=tf.float32, 
                initializer=tf.constant(
                    np.random.randn(self.data_dim, self.key_size) * np.sqrt(2 / (self.data_dim + self.key_size)), 
                    dtype=tf.float32))
            self.key_weights.append(weight)
        # weights and biases of the multi-layer nn for scoring
        self.score_layers  = [self.data_dim] + self.score_layers + [1]
        self.score_weights = []
        self.score_biases  = []
        for h in range(n_heads):
            weights = []
            biases  = []
            for l in range(1, len(self.score_layers)):
                weight = tf.get_variable(
                    name='%s_scoreweights_head%d_layer%d' % (model_name, h, l), 
                    dtype=tf.float32, 
                    initializer=tf.constant(
                        np.random.randn(self.score_layers[l-1], self.score_layers[l]) * np.sqrt(2 / (self.score_layers[l-1] + self.score_layers[l])), 
                        dtype=tf.float32))
                bias   = tf.get_variable(
                    name='%s_scorebiases_head%d_layer%d' % (model_name, h, l), 
                    dtype=tf.float32, 
                    initializer=tf.constant(
                        np.random.randn(self.score_layers[l]) * np.sqrt(2 / self.score_layers[l]), 
                        dtype=tf.float32))
                weights.append(weight)
                biases.append(bias)
            self.score_weights.append(weights)
            self.score_biases.append(biases)
        # weights and biases of the multi-layer nn for scoring
        self.lam_layers  = [self.n_heads * self.key_size] + self.lam_layers + [1]
        self.lam_weights = []
        self.lam_biases  = []
        for l in range(1, len(self.lam_layers)):
            weight = tf.get_variable(
                name='%s_lamweights_layer%d' % (model_name, l), 
                dtype=tf.float32, 
                initializer=tf.constant(
                    np.random.randn(self.lam_layers[l-1], self.lam_layers[l]) * np.sqrt(2 / (self.lam_layers[l-1] + self.lam_layers[l])), 
                    dtype=tf.float32))
            bias   = tf.get_variable(
                name='%s_lambiases_layer%d' % (model_name, l), 
                dtype=tf.float32, 
                initializer=tf.constant(
                    np.random.randn(self.lam_layers[l]) * np.sqrt(2 / self.lam_layers[l]), 
                    dtype=tf.float32))
            self.lam_weights.append(weight)
            self.lam_biases.append(bias)

        # input sequences
        self.X      = tf.placeholder(tf.float32, (None, self.max_seq_len, self.data_dim))
        # log likelihood
        self.loglik = self._log_likelihood()
    
    def _distance_func(self, xi, xj):
        """
        return the distance between xi and xj, where the time distance is calculated by ti - tj,
        the mark distance is calculated by mi - mj, and the space distance is determined by spatial
        covariance alpha_{t,si,sj}

        xi, xj: [batch_size, data_dim]
        """
        # dt = xi[:, 0] - xj[:, 0]              # [batch_size]
        # d  = tf.expand_dims(dt, axis=-1)       # [batch_size, data_dim]
        d = xi - xj # [batch_size, data_dim]
        return d

    def _score_func(self, h, xi, xj):
        """
        return the score between xi and xj at the k-th attention head. The score function is 
        parameterized by a multi-layer neural network.

        xi, xj: [batch_size, data_dim]
        """
        last_layer = self._distance_func(xi, xj) # [batch_size, data_dim]
        for w, b in zip(self.score_weights[h][:-1], self.score_biases[h][:-1]):
            last_layer = tf.math.softplus(tf.matmul(last_layer, w) + b)
        last_layer = tf.math.sigmoid(tf.matmul(last_layer, self.score_weights[h][-1]) + self.score_biases[h][-1])
        return last_layer                        # [batch_size, 1]
    
    def _multihead_attention(self, x, xs, mask=None):
        """
        return the single attention given a set of key vectors and corresponding scores. 
        The attention only keeps scores where their mask == 1 if mask is available.

        xs:   [batch_size, n_points, data_dim]
        x:    [batch_size, data_dim]
        mask: [batch_size, n_points, 1]
        """
        # retrieve data dimension
        batch_size = tf.shape(xs)[0]
        n_points   = tf.shape(xs)[1]
        attentions = []
        # data preparation
        mask = mask if mask is not None else tf.ones([batch_size, n_points, 1], dtype=tf.float32) 
        xs   = tf.reshape(xs, shape=[-1, self.data_dim])     # [batch_size * n_points, data_dim]
        x    = tf.tile(tf.expand_dims(x, axis=1), 
            multiples=[1, n_points, 1])                      # [batch_size, n_points, data_dim]
        x    = tf.reshape(x, shape=[-1, self.data_dim])      # [batch_size * n_points, data_dim]
        # construct attention
        for h in range(self.n_heads):
            # scores calculation between xs and x
            scores = self._score_func(h, x, xs)              # [batch_size * n_points, 1]
            scores = tf.reshape(scores, 
                shape=[batch_size, n_points, 1]) * mask      # [batch_size, n_points, 1]
            # key vectors of xs
            keys   = tf.matmul(xs, self.key_weights[h])      # [batch_size * n_points, key_size]
            keys   = tf.reshape(keys, 
                shape=[batch_size, n_points, self.key_size]) # [batch_size, n_points, key_size]
            # single attention for x
            attention = tf.reduce_sum(scores * keys, axis=1) # [batch_size, key_size]
            softmax   = tf.reduce_sum(scores, axis=1)        # [batch_size, 1]
            attention = attention / (softmax + 1e-8)         # [batch_size, key_size]
            attentions.append(attention)
        # multi-head attention for x
        multihead_attention = tf.concat(attentions, axis=1)  # [batch_size, key_size * n_heads]
        return multihead_attention
    
    def _mu(self, x):
        """
        return base intensity at specific location x in the space. 

        x: [batch_size, data_dim]
        """
        batch_size = tf.shape(x)[0]
        return tf.ones((batch_size, 1), dtype=tf.float32) # [batch_size, 1]

    def _history_val(self, attention):
        """
        return the history embedding value given multi-head attention. The history is parameterized
        by a multi-layer neural network.
        
        attention: [batch_size, 1]
        """
        last_layer = attention # [batch_size, key_size * n_heads]
        for l in range(len(self.lam_layers) - 1):
            last_layer = tf.math.softplus(tf.matmul(last_layer, self.lam_weights[l]) + self.lam_biases[l])
        return last_layer      # [batch_size, 1]

    def _lam(self, x, attention, mask=None):
        """
        return the lambda value given current x and history value. It will keep 
        attention values where its mask == 1 if the mask is available.

        x:       [batch_size, data_dim]
        history: [batch_size, 1]
        mask:    [batch_size, 1]
        """
        if mask is not None:
            _lam = self._mu(x) + self._history_val(attention) * mask # [batch_size, 1]
        else:
            _lam = self._mu(x) + self._history_val(attention)        # [batch_size, 1]
        return _lam                                                  # [batch_size, 1]

    def _integral_lam(self, tlim, n_tgrid):
        """
        calculate the integral of lambda via numerical approximation.
        TODO: To make it compatible with high-dimensional data (it only considers one-dimension for now).
        """
        # configuration
        batch_size = tf.shape(self.X)[0]
        T          = np.linspace(tlim[0], tlim[1], n_tgrid)            # np: [n_tgrid]
        lams       = []
        for t in T:
            # data preparation:
            # get historical sequences before time t.
            x         = tf.ones((batch_size, self.data_dim)) * t       # [batch_size, data_dim] (TODO)
            # calculate mask for scores in attention.
            mask1     = tf.cast(self.X[:, :, 0] < t, dtype=tf.float32) # [batch_size, max_seq_len]
            mask2     = tf.cast(self.X[:, :, 0] > 0, dtype=tf.float32) # [batch_size, max_seq_len]
            maskA     = tf.expand_dims(mask1 * mask2, axis=-1)         # [batch_size, max_seq_len, 1]
            attention = self._multihead_attention(x, self.X, maskA)    # [batch_size, key_size * n_heads]
            # calculate mask for attentions in lambda.
            maskL     = tf.cast(
                tf.reduce_sum(maskA, axis=1) > 0, dtype=tf.float32)    # [batch_size, 1]
            # calculate lambda given the history
            lam       = self._lam(x, attention, maskL)                 # [batch_size, 1]
            lams.append(lam)
        lams         = tf.concat(lams, axis=1)                         # [batch_size, n_tgrid]
        integral_lam = tf.reduce_sum(lams, axis=1) * \
            ((tlim[1] - tlim[0]) / n_tgrid)                            # [batch_size] (TODO)
        return lams, integral_lam

    def _log_likelihood(self):
        """
        calculate the log-likelihood of input data. 
        """
        # configuration and data preparation
        batch_size = tf.shape(self.X)[0]
        # data mask for tail truncation
        mask       = tf.cast(self.X[:, :, 0] > 0., dtype=tf.float32)    # [batch_size, max_seq_len]

        # term 1: the sum of log likelihood of each data point
        log_lams = []
        for i in range(self.max_seq_len):
            xs = self.X[:, :(i+1), :]                                   # [batch_size, i, data_dim]
            x  = self.X[:, i, :]                                        # [batch_size, 1, data_dim]
            attention = self._multihead_attention(x, xs)                # [batch_size, key_size * n_heads]
            log_lam   = tf.log(self._lam(x, attention))                 # [batch_size, 1]
            log_lams.append(log_lam)
        log_lams = tf.squeeze(tf.concat(log_lams, axis=1))              # [batch_size, max_seq_len]
        # truncate the invalid log_lam which corresponds to the tail of the sequence and 
        # sum up the valid log_lam
        log_lams = log_lams * mask
        log_lams = tf.reduce_sum(log_lams, axis=1)                      # [batch_size]

        # term 2: the survival probablity of the sequence
        _, integral_lam = self._integral_lam(tlim=[0., 1.], n_tgrid=50) # [batch_size]
        return log_lams - integral_lam

    def train(self, sess, data, epoches, batch_size, lr, decay_rate=0.99, decay_steps=100, test_seqs=None, is_new=True):
        """
        train model

        is_new:    is the model initialized as new model
        test_seqs: test sequences for real-time lambda value visualization
        """
        # Adam optimizer
        self.cost      = - tf.reduce_mean(self.loglik)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
        global_step    = tf.Variable(0, trainable=False)
        learning_rate  = tf.train.exponential_decay(
            learning_rate=lr, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)
        # initialize variables
        if is_new:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("[%s] parameters are initialized." % arrow.now(), file=sys.stderr)

        # # plotter initialization
        if test_seqs is not None: 
            tlim    = [0., 1.]
            n_tgrid = 100
            x       = np.linspace(tlim[0], tlim[1], n_tgrid)
            plotter = utils.LambdaPlotter(x)

        # data configurations
        n_data    = data.shape[0]            # number of data samples
        n_batches = int(n_data / batch_size) # number of batches
        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_train_ids = np.arange(n_data)
            np.random.shuffle(shuffled_train_ids)

            # training over batches
            avg_train_cost = []
            for b in range(n_batches):
                idx             = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_train_ids[idx]
                # training and testing batch data
                batch_train     = data[batch_train_ids, :, :]
                # optimization procedure
                sess.run(self.optimizer, feed_dict={self.X: batch_train})
                # cost for train batch and test batch
                train_cost      = sess.run(self.cost, feed_dict={self.X: batch_train})
                # record cost for each batch
                avg_train_cost.append(train_cost)

            # # update real-time plot
            if test_seqs is not None:
                lam     = self.evaluate_lam(sess, 
                    data=test_seqs, 
                    tlim=tlim, 
                    n_tgird=n_tgrid)       #[seq_size, n_tgrid, n_sgrid, n_sgrid]
                
                plotter.update(lam[0])
                    
            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d, learning_rate=%f)' % \
                (arrow.now(), epoch, n_batches, batch_size, sess.run(learning_rate)), file=sys.stderr)
            print('[%s] Train cost:\t%f' % \
                (arrow.now(), avg_train_cost), file=sys.stderr)

    def save_model(self, sess, file_path):
        """
        save model parameters as files

        e.g. file_path = "/tmp/model.ckpt"
        """
        # add ops to save and restore all the variables.
        saver     = tf.train.Saver()
        save_path = saver.save(sess, file_path)
        print('[%s] Model has been saved at %s' % (arrow.now(), save_path), file=sys.stderr)

    def evaluate_lam(self, sess, data, tlim, n_tgrid):
        """
        evaluate lambda
        """
        lams, _ = self._integral_lam(tlim, n_tgrid)
        lams    = sess.run(lams, feed_dict={self.X: data})
        return lams