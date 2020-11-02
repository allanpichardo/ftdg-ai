from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.signal import LogmelToMFCC
from kapre.composed import get_frequency_aware_conv2d, get_melspectrogram_layer
import tensorflow as tf
import os


# Audio is channels first. Images are channels last

def get_audio_layer(SR=22050, DT=8.0):
    input_shape = (1, int(SR * DT))
    melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=2048, hop_length=512, mel_f_min=60.0,
                                       mel_f_max=8000.0, return_decibel=True,
                                       n_mels=128, sample_rate=SR, input_data_format='channels_first',
                                       output_data_format='channels_last')
    return melgram


def get_spectrogram_input_layer():
    input = tf.keras.layers.Input(shape=(341, 128, 1), name='spectro_input')
    return input


def get_mfcc_input_layer():
    input = tf.keras.layers.Input(shape=(341, 13, 1), name='spectro_input')
    return input


def get_wavform_input_layer(sr=22050, duration=8.0):
    input = tf.keras.layers.Input(shape=(1, int(sr * duration)))
    return input


def get_classifier_model(sr=22050, duration=8.0, n_classes=40):
    i = get_audio_layer(sr, duration)
    encoder = get_2d_encoder()
    model = tf.keras.Sequential([
        i,
        LogmelToMFCC(n_mfccs=13),
        encoder,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(96, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),  # L2 normalize embeddings,
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
    return model


def get_2d_encoder():
    input = get_spectrogram_input_layer()
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='tanh')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(1):
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D()(x)

    for i in range(1):
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D()(x)

    for i in range(1):
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D()(x)

    for i in range(1):
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    for i in range(1):
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = Model(inputs=input, outputs=x, name='2d_encoder')
    return model


def get_efficientnet_triplet(sr=22050, duration=8.0, embedding_size=256):
    i = get_audio_layer(sr, duration)
    en = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                           input_shape=(341, 128, 3),
                                                           pooling='avg',
                                                           weights='imagenet')
    model = tf.keras.Sequential([
        i,
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Conv2D(3, (3, 3), padding='same'),
        en,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(embedding_size, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),  # L2 normalize embeddings,
    ])
    return model


def get_vgg_triplet(sr=22050, duration=8.0, embedding_size=128):
    i = get_audio_layer(sr, duration)
    encoder = get_2d_encoder()
    model = tf.keras.Sequential([
        i,
        tf.keras.layers.BatchNormalization(),
        encoder,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(embedding_size, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),  # L2 normalize embeddings,
    ])
    return model


def get_embedding_classifier(embedding_size=128, n_classes=40):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(embedding_size,)),
        tf.keras.layers.Dense(embedding_size * 3 // 4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model


def get_resnet_triplet(sr=22050, duration=8.0, embedding_size=128):
    stride = 1
    channel_axis = 3

    logmel = get_audio_layer(sr, duration)
    inp = get_spectrogram_input_layer()
    x = inp
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=stride, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = res_layer(x, 32, dropout=0.2)
    x = res_layer(x, 32, dropout=0.3)
    x = res_layer(x, 32, dropout=0.4, pooling=True)
    x = res_layer(x, 64, dropout=0.2)
    x = res_layer(x, 64, dropout=0.2, pooling=True)
    x = res_layer(x, 256, dropout=0.4)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(4096, activation=None)(x)
    x = tf.keras.layers.Dropout(0.23)(x)
    x = tf.keras.layers.Dense(embedding_size)(x)
    x = tf.keras.layers.Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x)  # L2 normalize embeddings

    resnet_model = Model(inp, x, name="Resnet")
    return tf.keras.Sequential([logmel, resnet_model])


def res_layer(x, filters, pooling=False, dropout=0.0, stride=1, channel_axis=3):
    temp = x
    temp = tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)
    temp = tf.keras.layers.BatchNormalization(axis=channel_axis)(temp)
    temp = tf.keras.layers.Activation("relu")(temp)
    temp = tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)

    x = tf.add(temp, tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(x))
    if pooling:
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    if dropout != 0.0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


if __name__ == '__main__':
    model = get_vgg_triplet()
    # model.layers[2].summary()
    model.summary()
