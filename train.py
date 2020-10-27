import tensorflow as tf
import tensorflow_addons as tfa
from src.sound_sequence import SoundSequence
from src.models import get_1d_model, get_1d_autoencoder
from src.callbacks import TensorBoardImage

if __name__ == '__main__':
    seq = SoundSequence('/Users/allanpichardo/PycharmProjects/ftdg-ai/music', shuffle=True, is_autoencoder=True, use_raw_audio=False)
    model = get_1d_autoencoder()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'],
    )
    model.fit(seq, epochs=10, callbacks=[
        tf.keras.callbacks.TensorBoard(write_images=True),
        TensorBoardImage()
    ])