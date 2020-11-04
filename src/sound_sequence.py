import tensorflow as tf
from glob import glob
import os
import numpy as np
from scipy.io import wavfile
from tensorflow.keras.utils import to_categorical
from kapre.composed import get_melspectrogram_layer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import sys


class SoundSequence(tf.keras.utils.Sequence):

    def __init__(self, music_path, sr=22050, duration=8.0, subset='training', use_categorical=True,
                 batch_size=32, is_autoencoder=False, shuffle=True, use_raw_audio=True):
        """
        Create a data generator that reads wav files from a directory
        :param music_path:
        :param sr: sampling rate
        :param duration: duration of sound clips
        :param subset: one of 'training' or 'validation'
        :param batch_size:
        :param is_autoencoder: if True, y = x
        :param shuffle:
        """
        self.sr = sr
        self.duration = duration
        self.batch_size = batch_size
        self.is_autoencoder = is_autoencoder
        self.shuffle = shuffle
        self.classes = [os.path.basename(x) for x in glob(os.path.join(music_path, subset, '*'))]
        self.n_classes = len(self.classes)
        self.use_raw_audio = use_raw_audio
        self.use_categorical = use_categorical

        input_shape = (1, int(self.sr * self.duration))
        melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
                                           n_mels=96, sample_rate=self.sr, input_data_format='channels_first',
                                           output_data_format='channels_last')
        self.mel = melgram

        le = LabelEncoder()
        le.fit(self.classes)

        self.wav_paths = glob(os.path.join(music_path, subset, '*', '*'))
        self.labels = []
        for path in self.wav_paths:
            real_label = os.path.basename(os.path.dirname(path))
            self.labels.append(real_label)

        self.labels = le.transform(self.labels)
        counts = Counter(self.labels)
        min = sys.maxsize
        for key, count in counts.items():
            if count < min:
                min = count

        self.weights = {}
        for i in range(self.n_classes):
            self.weights[i] = min / counts[i] if counts[i] > 0 else 0.0


        self.on_epoch_end()

    def __normalize_wav(self, wav):
        min = -32768
        max = 32767
        return (wav - min) / (max - min)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        # X = np.empty((self.batch_size, 1, int(self.sr * self.duration)), dtype=np.float32)
        # Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)
        X = []
        Y = []

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            wav = wav[:rate * int(self.duration)]
            X.append(wav.reshape(1, -1))
            Y.append(to_categorical(label, num_classes=self.n_classes))
            # X[i,] = wav.reshape(1, -1)
            # Y[i,] = to_categorical(label, num_classes=self.n_classes)

        X = np.array(X)
        # X = self.__normalize_wav(X)
        Y = np.array(labels) if not self.use_categorical else np.array(Y)

        if self.use_raw_audio:
            return X, Y if not self.is_autoencoder else X
        else:
            spec = self.mel.predict_on_batch(X)
            return spec, Y if not self.is_autoencoder else spec

    def __len__(self):
        return int(np.ceil(len(self.wav_paths) / float(self.batch_size)))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    # seq = SoundSequence('/Users/allanpichardo/PycharmProjects/ftdg-ai/music', is_autoencoder=True, batch_size=1, use_raw_audio=False)
    # batch = seq.__getitem__(0)
    # sample = batch[0]
    # from src.models import get_minmax_normalize_layer
    # normed = get_minmax_normalize_layer((341, 128, 1))(sample)
    # print(normed)
    val = SoundSequence('/Users/allanpichardo/PycharmProjects/ftdg-ai/music', use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=True,
                        batch_size=1, subset='validation')

    for wav, labels in val:
        [print(str(x) + "\n") for x in labels]