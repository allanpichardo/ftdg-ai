import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
from src.sound_sequence import SoundSequence
from src.models import get_1d_model, get_1d_autoencoder
from src.callbacks import TensorBoardImage

if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    train = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'),
                        shuffle=True, is_autoencoder=True, use_raw_audio=False,
                        batch_size=8, subset='training')

    val = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'),
                        shuffle=True, is_autoencoder=True, use_raw_audio=False,
                        batch_size=8, subset='validation')

    model = get_1d_autoencoder()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'],
    )
    model.fit(train, validation_data=val, epochs=10, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
        TensorBoardImage()
    ])