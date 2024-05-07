from time import time
import numpy as np
import platform
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from stability import stability
import metrics
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import LambdaCallback
import datetime

tf.compat.v1.enable_eager_execution()

def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=0.5, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FcDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0):

        super(FcDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.autoencoder, self.encoder = autoencoder(self.dims)

        # prepare FcDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256,
                 save_dir='results/temp', verbose=1, aug_pretrain=False):
        print('Begin pretraining: ', '-' * 60)
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        visualization_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.visualize_clusters(self.predict(x), y, 'p'))
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        # cb = [csv_logger, visualization_callback]   # 加了回调函数来可视化预训练过程
        if y is not None and verbose > 0:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs / 10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(index=int(len(self.model.layers) / 2)).output)
                    features = feature_model.predict(self.x)    # shape:(9298, 1024)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20)
                    y_pred = km.fit_predict(features)
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
                          % (acc, nmi, ari))
            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        if not aug_pretrain:
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        else:
            print('-=*'*20)
            print('Using augmentation for ae')
            print('-=*'*20)
            def gen(x, batch_size):
                if len(x.shape) > 2:  # image
                    gen0 = self.datagen.flow(x, shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        yield (batch_x, batch_x)
                else:
                    width = int(np.sqrt(x.shape[-1]))
                    if width * width == x.shape[-1]:  # gray
                        im_shape = [-1, width, width, 1]
                    else:  # RGB
                        width = int(np.sqrt(x.shape[-1] / 3.0))
                        im_shape = [-1, width, width, 3]
                    gen0 = self.datagen.flow(np.reshape(x, im_shape), shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
                        yield (batch_x, batch_x)
            self.autoencoder.fit_generator(gen(x, batch_size), steps_per_epoch=int(x.shape[0]/batch_size),
                                           epochs=epochs, callbacks=cb, verbose=verbose,
                                           workers=1, use_multiprocessing=True if platform.system() != "Windows" else False)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def load_weights(self, weights):  # load weights of FcDEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q

    def predict_labels(self, x):  # predict cluster labels using the output of clustering layer
        return np.argmax(self.predict(x), 1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def random_transform(self, x):
        if len(x.shape) > 2:  # image
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

        # if input a flattened vector, reshape to image before transform
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen = self.datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
        return np.reshape(gen.next(), x.shape)

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def OTSU(self, tensor):
        # Compute the histogram of the tensor
        hist = tf.histogram_fixed_width(tensor, [0.0, 1.0], nbins=1000)
        # Compute the probabilities of each intensity level
        total_pixels = tf.reduce_prod(tf.shape(tensor))
        probs = tf.cast(hist, tf.float32) / tf.cast(total_pixels, tf.float32)
        # Compute the cumulative sums of the probabilities and the intensity levels
        cum_probs = tf.cumsum(probs)
        cum_intensities = tf.cumsum(tf.range(1000, dtype=tf.float32) * probs)
        # Compute the global mean intensity level
        global_mean = cum_intensities[-1]
        # Compute the between-class variance for each possible threshold value
        threshold_values = tf.range(1000, dtype=tf.float32)
        between_class_variances = ((global_mean * cum_probs - cum_intensities) ** 2 /
                                   (cum_probs * (1.0 - cum_probs) + 1e-6))
        # Find the threshold value that maximizes the between-class variance
        threshold = tf.argmax(between_class_variances)
        # Normalize the threshold value to the range [0, 1]
        threshold = tf.cast(threshold, tf.float32) / 1000.0
        return threshold

    def train_on_batch(self, x, model, optimizer, loss_fn, t):
        with tf.GradientTape() as tape:
            q = model(x, training=True)
            loss = loss_fn(q, t)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-5,
            update_interval=30, save_dir='./results/temp', aug_cluster=False,
            optimizer=tf.keras.optimizers.Adam(), loss_fn=stability()):
        print('Begin clustering:', '-' * 60)
        print('Update interval', update_interval)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)
        print(f'batch size:{batch_size}')

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)  # , init='k-means++'
        features = self.encoder.predict(x)
        y_pred = kmeans.fit_predict(features)
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        acc_best = 0.0
        nmi_best = 0.0
        ari_best = 0.0
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])

        t=0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.predict(x)
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                avg_loss = loss / update_interval
                loss = 0.0

                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=tf.reduce_sum(avg_loss).numpy())
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f; loss=%.5f' % (ite, acc, nmi, ari, avg_loss))
                    if acc > acc_best:
                        acc_best = acc
                        nmi_best = nmi
                        ari_best = ari

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/model_' + str(ite) + '.h5')

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            x_batch = self.random_transform(x[idx]) if aug_cluster else x[idx]

            q = self.model(x_batch)
            t = self.OTSU(q)

            loss += self.train_on_batch(x=x_batch, model=self.model, optimizer=optimizer, loss_fn=loss_fn, t=t)
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)
        print('Best: Best_acc=%.5f, Best_nmi=%.5f, Best_ari=%.5f' % (acc_best, nmi_best, ari_best))

        return y_pred

