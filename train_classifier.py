import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
from src.sound_sequence import SoundSequence
from src.models import get_embedding_classifier, get_efficientnet_triplet
import argparse

if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    checkpoint = os.path.join(os.path.dirname(__file__), 'checkpoints', 'classifier',
                              datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint, exist_ok=True)
    checkpoint = os.path.join(checkpoint, 'checkpoint.h5')

    parser = argparse.ArgumentParser(description='Train the music model.')
    parser.add_argument('--batch_size', type=int,
                        help='the batch size', default=32)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--epochs', type=int, help='Training epoch amount', default=10)
    parser.add_argument('--checkpoint', type=str, help='Load weights from checkpoint file', default=checkpoint)
    parser.add_argument('--triplet_model', help='Path to a savedmodel of triplet model', required=True)
    parser.add_argument('--tag', type=str, help='A version to tag the final output', default='v1')
    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    checkpoint = args.checkpoint
    tag = args.tag
    triplet_path = args.triplet_model

    train = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=True,
                          shuffle=True, is_autoencoder=False, use_raw_audio=True,
                          batch_size=batch_size, subset='training')
    val = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=True,
                        shuffle=True, is_autoencoder=False, use_raw_audio=True,
                        batch_size=batch_size, subset='validation')

    print("Loading triplet model {}".format(triplet_path))
    triplet = tf.keras.models.load_model(triplet_path)

    for layer in triplet.layers:
        layer.trainable = False

    model = get_embedding_classifier(triplet, embedding_size=256, n_classes=train.n_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.fit(train, epochs=epochs, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='max')
    ], class_weight=train.weights, validation_data=val)

    save_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'classifier')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'classifier_{}'.format(tag))

    print("Saving model to {}".format(save_path))
    model.save(save_path, overwrite=True)
