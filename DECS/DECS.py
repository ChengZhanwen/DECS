"""
Tensorflow implementation for ConvDEC and ConvDEC-DA algorithms:
    - Xifeng Guo, En Zhu, Xinwang Liu, and Jianping Yin. Deep Embedded Clustering with Data Augmentation. ACML 2018.

Author:
    Xifeng Guo. 2018.6.30
"""

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, InputLayer, MaxPooling2D, UpSampling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from FcDEC import FcDEC, ClusteringLayer
import tensorflow as tf

def CAE(input_shape=(150, 150, 3), filters=[32, 64, 128, 256, 10]):
    model = Sequential()

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(InputLayer(input_shape))
    model.add(Conv2D(filters[0], 3, padding='same', activation='relu', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters[1], 3, padding='same', activation='relu', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters[2], 3, padding='same', activation='relu', name='conv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters[3], 3, padding=pad3, activation='relu', name='conv4'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(units=filters[4], name='embedding'))

    if input_shape[0] % 8 == 0:
        padding = 'same'
    else:
        padding = 'valid'

    model.add(Dense(units=int(input_shape[0]/8) * int(input_shape[0]/8) * filters[3], activation='relu'))
    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[3])))
    model.add(Conv2DTranspose(filters[2], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding='same', activation='relu', name='deconv2'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv1'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(input_shape[2], 3, padding='same', name='deconv0'))

    encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
    # return model, encoder

    from tensorflow.keras.layers import Input
    decoder_input = Input(shape=(filters[4],))
    dec_layer = decoder_input
    dec_layer = model.layers[-9](dec_layer)
    for i in range(len(model.layers) - 8, len(model.layers)):
        dec_layer = model.layers[i](dec_layer)

    decoder = Model(inputs=decoder_input, outputs=dec_layer)

    return model, encoder, decoder

class DECS(FcDEC):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10):

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.datagenx = ImageDataGenerator()
        self.autoencoder, self.encoder, self.decoder = CAE(input_shape, filters)


        # Define ConvIDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=clustering_layer)
