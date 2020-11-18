import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
from src.sound_sequence import SoundSequence
from src.models import get_efficientnet_triplet, get_vgg_triplet, get_inception_resnet_triplet
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
    parser.add_argument('--subset', type=str, help='Which subset of music to train on?', default='training')
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--epochs', type=int, help='Training epoch amount', default=10)
    parser.add_argument('--checkpoint', type=str, help='Load weights from checkpoint file', default=checkpoint)
    parser.add_argument('--embedding_dim', type=int, help='Size of output embedding', default=256)
    parser.add_argument('--freeze', type=bool, help='Freeze weights?', default=False)
    parser.add_argument('--architecture', type=str, help='CNN architecture: vgg or efficientnet or inception', default='efficientnet')
    parser.add_argument('--tag', type=str, help='A version to tag the final output', default='v1')
    parser.add_argument('--embedding_frequency', type=int, help='How often to generate embeddings for projector', default=10)
    parser.add_argument('--use_weights', type=bool, help='Use class weights during training', default=False)
    args = parser.parse_args()

    batch_size = args.batch_size
    margin = args.margin
    lr = args.lr
    epochs = args.epochs
    embedding_dim = args.embedding_dim
    freeze = args.freeze
    architecture = args.architecture
    tag = args.tag
    checkpoint = args.checkpoint
    embed_freq = args.embedding_frequency
    subset = args.subset
    use_weights = args.use_weights

    train = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                        shuffle=True, is_autoencoder=False, use_raw_audio=True,
                        batch_size=batch_size, subset=subset)

    model = get_efficientnet_triplet(embedding_size=embedding_dim)
    if architecture == 'efficientnet':
        model = get_efficientnet_triplet(embedding_size=embedding_dim)
    elif architecture == 'inception':
        model = get_inception_resnet_triplet(embedding_size=embedding_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=1e-3),
        loss=tfa.losses.TripletSemiHardLoss(margin=margin),
    )

    if freeze and architecture == 'efficientnet':
        for layer in model.get_layer('efficientnetb0').layers:
            layer.trainable = False

    if os.path.exists(checkpoint):
        print("Loading weights from checkpoint {}".format(checkpoint))
        model.load_weights(checkpoint)

    model.fit(train, epochs=epochs, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, embeddings_freq=embed_freq),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=5, mode='min')
    ], class_weight=train.weights if use_weights else None)

    save_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'triplet')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, '{}_{}'.format(architecture, tag))

    print("Saving model to {}".format(save_path))
    model.save(save_path, overwrite=True)
