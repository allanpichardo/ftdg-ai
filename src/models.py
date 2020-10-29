from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.composed import get_frequency_aware_conv2d, get_melspectrogram_layer
import tensorflow as tf
import os


# Audio is channels first. Images are channels last

def get_audio_layer(SR=22050, DT=8.0):
    input_shape = (1, int(SR * DT))
    melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
                                       n_mels=96, sample_rate=SR, input_data_format='channels_first',
                                       output_data_format='channels_last')
    return melgram


def get_spectrogram_input_layer():
    input = tf.keras.layers.Input(shape=(686, 96, 1), name='spectro_input')
    return input


def get_wavform_input_layer(sr=22050, duration=8.0):
    input = tf.keras.layers.Input(shape=(1, int(sr * duration)))
    return input


def get_1d_autoencoder():
    enc = get_1d_encoder()
    dec = get_1d_decoder(enc.output.shape[1])
    return tf.keras.Sequential([enc, dec])


def get_2d_model(sr=22050, duration=8.0, n_classes=40):
    i = get_audio_layer(sr, duration)
    encoder = get_2d_encoder()
    model = tf.keras.Sequential([
        i,
        encoder,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),  # L2 normalize embeddings,
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
    return model


def get_1d_model(sr=22050, duration=8.0, n_classes=40):
    encoder = get_1d_encoder()
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(96, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    ])
    return model


def get_1d_decoder(input_shape=(96,)):
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32 * 3, activation='tanh')(i)
    x = tf.keras.layers.Reshape(target_shape=(32, 3, 1))(x)
    x = TimeDistributed(layers.Conv1D(96, kernel_size=(3), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Conv1D(96, kernel_size=(3), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Conv1D(96, kernel_size=(3), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Conv1D(72, kernel_size=(5), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Conv1D(48, kernel_size=(7), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Conv1D(24, kernel_size=(9), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.Cropping2D((169, 0))(x)

    x = TimeDistributed(layers.Conv1D(1, kernel_size=3, activation='tanh', padding='same'))(x)

    model = Model(inputs=i, outputs=x, name='1d_decoder')
    return model



def get_2d_encoder():
    i = get_spectrogram_input_layer()
    x = tf.keras.layers.BatchNormalization()(i)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = Model(inputs=i, outputs=x, name='2d_encoder')
    return model


def get_1d_encoder():
    i = get_spectrogram_input_layer()
    x = TimeDistributed(layers.Conv1D(24, kernel_size=(9), activation='tanh', padding='same'))(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = TimeDistributed(layers.Conv1D(48, kernel_size=(7), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = TimeDistributed(layers.Conv1D(72, kernel_size=(5), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = TimeDistributed(layers.Conv1D(96, kernel_size=(3), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = TimeDistributed(layers.Conv1D(96, kernel_size=(3), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = TimeDistributed(layers.Conv1D(96, kernel_size=(3), activation='tanh', padding='same'))(x)
    x = tf.keras.layers.Flatten()(x)

    model = Model(inputs=i, outputs=x, name='1d_encoder')
    return model


# def Conv1D(N_CLASSES=10, SR=22050, DT=1.0):
#     input_shape = (1, int(SR*DT))
#     melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
#                                        n_mels=96, sample_rate=SR, input_data_format='channels_first',
#                                        output_data_format='channels_last')
#
#     x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(melgram)
#     x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
#     x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
#     x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
#     x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
#     x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
#     x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
#     x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4')(x)
#     x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
#     x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
#     x = layers.Dropout(rate=0.1, name='dropout')(x)
#     x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
#     o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
#
#     model = Model(inputs=i, outputs=o, name='1d_convolution')
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# def Conv2D(N_CLASSES=10, SR=22050, DT=1.0):
#     input_shape = (1, int(SR * DT))
#     melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
#                                        n_mels=96, sample_rate=SR, input_data_format='channels_first',
#                                        output_data_format='channels_last')
#     x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(melgram)
#     x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
#     x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
#     x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
#     x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
#     x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
#     x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
#     x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
#     x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
#     x = layers.Flatten(name='flatten')(x)
#     x = layers.Dropout(rate=0.2, name='dropout')(x)
#     x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
#     o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
#
#     model = Model(inputs=i, outputs=o, name='2d_convolution')
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# def LSTM(N_CLASSES=10, SR=22050, DT=1.0):
#     i = layers.Input(shape=(1, int(SR*DT)), name='input')
#     x = Melspectrogram(n_dft=512, n_hop=160,
#                        padding='same', sr=SR, n_mels=128,
#                        fmin=0.0, fmax=SR/2, power_melgram=2.0,
#                        return_decibel_melgram=True, trainable_fb=False,
#                        trainable_kernel=False,
#                        name='melbands')(i)
#     x = Normalization2D(str_axis='batch', name='batch_norm')(x)
#     x = layers.Permute((2,1,3), name='permute')(x)
#     x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
#     s = TimeDistributed(layers.Dense(64, activation='tanh'),
#                         name='td_dense_tanh')(x)
#     x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
#                              name='bidirectional_lstm')(s)
#     x = layers.concatenate([s, x], axis=2, name='skip_connection')
#     x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
#     x = layers.MaxPooling1D(name='max_pool_1d')(x)
#     x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
#     x = layers.Flatten(name='flatten')(x)
#     x = layers.Dropout(rate=0.2, name='dropout')(x)
#     x = layers.Dense(32, activation='relu',
#                          activity_regularizer=l2(0.001),
#                          name='dense_3_relu')(x)
#     o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
#
#     model = Model(inputs=i, outputs=o, name='long_short_term_memory')
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model


if __name__ == '__main__':
    # enc = get_1d_encoder()
    # dec = get_1d_decoder(enc.output.shape[1])
    # model = tf.keras.Sequential([enc, dec])
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.layers[0].summary()
    # model.layers[1].summary()
    # model.summary()
    model = get_2d_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    for layer in model.layers:
        layer.summary()
