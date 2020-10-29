import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
from src.sound_sequence import SoundSequence
from src.models import get_2d_model
import argparse

if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    checkpoint = os.path.join(os.path.dirname(__file__), 'checkpoints', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint, exist_ok=True)
    checkpoint = os.path.join(checkpoint, 'checkpoint.h5')

    parser = argparse.ArgumentParser(description='Train the music model.')
    parser.add_argument('--batch_size', type=int,
                        help='the batch size', default=32)
    parser.add_argument('--margin', type=float, help='The triplet loss margin', default=1.0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--epochs', type=int, help='Training epoch amount', default=10)
    args = parser.parse_args()

    batch_size = args.batch_size
    margin = args.margin
    lr = args.lr
    epochs = args.epochs

    train = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=True,
                        batch_size=batch_size, subset='training')

    val = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=True,
                        batch_size=batch_size, subset='validation')

    model = get_2d_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tfa.losses.TripletSemiHardLoss(margin),
        metrics=['accuracy']
    )
    model.fit(train, validation_data=val, epochs=epochs, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, save_best_only=True, verbose=1)
    ], class_weight=train.weights)