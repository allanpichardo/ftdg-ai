import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
from src.sound_sequence import SoundSequence
from src.models import get_efficientnet_triplet, get_vgg_triplet
import argparse

if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    checkpoint = os.path.join(os.path.dirname(__file__), 'checkpoints', 'triplet', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint, exist_ok=True)
    checkpoint = os.path.join(checkpoint, 'checkpoint.h5')

    parser = argparse.ArgumentParser(description='Train the music model.')
    parser.add_argument('--batch_size', type=int,
                        help='the batch size', default=32)
    parser.add_argument('--margin', type=float, help='The triplet loss margin', default=1.0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--epochs', type=int, help='Training epoch amount', default=10)
    parser.add_argument('--checkpoint', type=str, help='Load weights from checkpoint file', default=checkpoint)
    parser.add_argument('--embedding_dim', type=int, help='Size of output embedding', default=256)
    parser.add_argument('--freeze', type=bool, help='Freeze weights?', default=False)
    parser.add_argument('--architecture', type=str, help='CNN architecture: vgg or efficientnet', default='efficientnet')
    args = parser.parse_args()

    batch_size = args.batch_size
    margin = args.margin
    lr = args.lr
    epochs = args.epochs
    embedding_dim = args.embedding_dim
    freeze = args.freeze
    architecture = args.architecture

    train = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=True,
                        batch_size=batch_size, subset='training')

    model = get_efficientnet_triplet(embedding_size=embedding_dim) if architecture == 'efficientnet' else get_vgg_triplet(embedding_size=embedding_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tfa.losses.TripletSemiHardLoss(margin=margin),
    )

    if freeze and architecture == 'efficientnet':
        for layer in model.get_layer('efficientnetb0').layers:
            layer.trainable = False

    if os.path.exists(checkpoint):
        print("Loading weights from checkpoint {}".format(checkpoint))
        model.load_weights(checkpoint)

    model.fit(train, epochs=epochs, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=5, mode=min)
    ], class_weight=train.weights)