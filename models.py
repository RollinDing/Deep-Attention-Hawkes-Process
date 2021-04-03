import sys
import arrow
import numpy as np
import tensorflow as tf
import utils
from base import BaseDeepAttentionHawkes

"""
Extended Attention Hawkes Models based on the Base Model

Dependencies:
- Python==3.7.1
- tensorflow==1.12.0
"""

# FOR SIMULATION EXPERIMENTS USING SYNTHETIC DATA

class SelfCorrectingAttentionTemporalHawkes(BaseDeepAttentionHawkes):
    """
    Self-Correcting Attention Hawkes

    The model considers consistently incremental base intensity as its mu function and attention
    structure is used to capture the `jump` behavior in the process.
    """

    def __init__(self, model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers):
        """
        Args:
        - model_name:   the name of the model
        - max_seq_len:  maximun length of input sequences
        - n_heads:      the number of heads in multi-head attention
        - key_size:     the size of key vectors
        - score_layers: the size of each layer in score function
        - lam_layers:   the size of each layer in lambda function
        """
        super().__init__(model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers)

    def _mu(self, x):
        """
        return base intensity at specific location x in the space. 

        x: [batch_size, data_dim]
        """
        # return tf.cast(tf.exp(10. * x), dtype=tf.float32)
        batch_size = tf.shape(x)[0]
        _mu        = tf.ones((batch_size, 1), dtype=tf.float32) * 10. # [batch_size, 1]
        return tf.cast(tf.exp(_mu * x), dtype=tf.float32)

    def _lam(self, x, attention, mask=None):
        """
        return the lambda value given multi-head attention. The lambda function is parameterized
        by a multi-layer neural network.
        
        x:         [batch_size, data_dim]
        attention: [batch_size, 1]
        """
        if mask is not None:
            _lam = self._mu(x) / (tf.exp(super()._history_val(attention) * mask)) # [batch_size, 1]
        else:
            _lam = self._mu(x) / tf.exp(super()._history_val(attention))          # [batch_size, 1]
        return _lam
    
class SelfExcitingAttentionTemporalHawkes(BaseDeepAttentionHawkes):
    """
    Self-Exciting Attention Hawkes

    The model considers constant base intensity given a location in the space as mu function.
    """

    def __init__(self, model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers):
        """
        Args:
        - model_name:   the name of the model
        - max_seq_len:  maximun length of input sequences
        - n_heads:      the number of heads in multi-head attention
        - key_size:     the size of key vectors
        - score_layers: the size of each layer in score function
        - lam_layers:   the size of each layer in lambda function
        """
        super().__init__(model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers)

    def _mu(self, x):
        """
        return base intensity at specific location x in the space. 

        x: [batch_size, data_dim] or [data_dim]
        """
        batch_size = tf.shape(x)[0]
        return tf.ones((batch_size, 1), dtype=tf.float32) * 10 # [batch_size, 1]

class AttentionSpatialHawkes(BaseDeepAttentionHawkes):
    """
    Self-Exciting Attention Hawkes

    The model considers constant base intensity given a location in the space as mu function.
    """

    def __init__(self, model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers):
        """
        Args:
        - model_name:   the name of the model
        - max_seq_len:  maximun length of input sequences
        - n_heads:      the number of heads in multi-head attention
        - key_size:     the size of key vectors
        - score_layers: the size of each layer in score function
        - lam_layers:   the size of each layer in lambda function
        """
        super().__init__(model_name, max_seq_len, n_heads, key_size, score_layers, lam_layers, data_dim=3)

    def _mu(self, x):
        """
        return base intensity at specific location x in the space. 

        x: [batch_size, data_dim] or [data_dim]
        """
        batch_size = tf.shape(x)[0]
        return tf.ones((batch_size, 1), dtype=tf.float32) * 10 # [batch_size, 1]

    def _integral_lam(self, tlim, slim, n_tgrid, n_sgrid):
        """
        calculate the integral of lambda via numerical approximation.
        """
        # configuration
        batch_size = tf.shape(self.X)[0]
        T          = np.linspace(tlim[0], tlim[1], n_tgrid)            # np: [n_tgrid]
        X          = np.linspace(slim[0], slim[1], n_sgrid)            # np: [n_sgrid]
        Y          = np.linspace(slim[0], slim[1], n_sgrid)            # np: [n_sgrid]
        S          = np.array([ [x, y] for x in X for y in Y ])        # np: [n_sgrid * n_sgrid, 2]
        lams       = []
        for t in T:
            for s in S:
                # data preparation:
                # get historical sequences before time t.
                x = tf.constant((t, s[0], s[1]), dtype=tf.float32)                # [data_dim]
                x = tf.tile(tf.expand_dims(x, axis=0), multiples=[batch_size, 1]) # [batch_size, data_dim]
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
        
        lams         = tf.concat(lams, axis=1)                             # [batch_size, n_tgrid * n_sgrid * n_sgrid]
        integral_lam = tf.reduce_sum(lams, axis=1) * \
            (((tlim[1] - tlim[0]) / n_tgrid) * \
            (((slim[1] - slim[0]) / n_sgrid) ** 2))                        # [batch_size]
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
        _, integral_lam = self._integral_lam(tlim=[0., 1.], slim=[-1., 1.], n_tgrid=1, n_sgrid=1) # [batch_size]
        return log_lams - integral_lam
    
    def evaluate_lam(self, sess, data, tlim, slim, n_tgrid, n_sgrid):
        """
        evaluate lambda
        """
        n       = len(data)
        lams, _ = self._integral_lam(tlim, slim, n_tgrid, n_sgrid)
        lams    = sess.run(lams, feed_dict={self.X: data})
        # TODO: Reshape lams to [n_tgrid, n_sgrid, n_sgrid]
        lams    = lams.reshape([n, n_tgrid, n_sgrid, n_sgrid])
        return lams

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
            slim    = [-1., 1.]
            n_tgrid = 100
            n_sgrid = 10
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
                    slim=slim, 
                    n_tgrid=n_tgrid, 
                    n_sgrid=n_sgrid)       #[seq_size, n_tgrid, n_sgrid, n_sgrid]
                
                plotter.update(lam[0, :, :, :])
                    
            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d, learning_rate=%f)' % \
                (arrow.now(), epoch, n_batches, batch_size, sess.run(learning_rate)), file=sys.stderr)
            print('[%s] Train cost:\t%f' % \
                (arrow.now(), avg_train_cost), file=sys.stderr)