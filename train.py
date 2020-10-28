import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
from src.sound_sequence import SoundSequence
from src.models import get_2d_model
from src.callbacks import TensorBoardImage

if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    batch_size = 128

    train = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=False,
                        batch_size=batch_size, subset='training')

    val = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=False,
                        batch_size=batch_size, subset='validation')

    model = get_2d_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0000001),
        loss=tfa.losses.TripletSemiHardLoss()
    )
    model.fit(train, validation_data=val, epochs=10, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
        # TensorBoardImage()
    ])