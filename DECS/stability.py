import tensorflow as tf

# 启用即时执行模式
tf.compat.v1.enable_eager_execution()
device = tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0')

class stability(tf.keras.losses.Loss):
    def __init__(self, k=10):
        super(stability, self).__init__()
        self.k = k

    def call(self, y_pred, t):
        '''
        compute loss
        :param y_pred: predicted co-probability
        :param t: threshold
        :return: loss
        '''
        # get fq
        fq_pij = self.get_fq(y_pred, t)
        # get sq
        sq = self.get_sq(fq_pij)
        return 1.0 - (tf.reduce_mean(sq) + tf.math.reduce_variance(sq))

    def get_fq(self, p_ij, t):
        '''
        compute determinacy
        :param p_ij: co-probability
        :param t: threshold
        :return: determinacy
        '''
        fq_pij = tf.where(p_ij < t, tf.abs((p_ij - t) / t), tf.abs((p_ij - t) / (1 - t)))
        return fq_pij

    def get_sq(self, fq_pij):
        '''
        compute stability
        :param fq_pij: determinacy
        :return: stability
        '''
        mean = tf.reduce_mean(fq_pij, axis=1)
        var = tf.math.reduce_variance(fq_pij, axis=1)
        sq = mean - 0.4 * var

        # objective
        return sq
