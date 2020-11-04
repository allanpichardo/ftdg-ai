import argparse
import tensorflow as tf
from src.sound_sequence import SoundSequence
import os
import numpy as np
import io
import tensorflow_datasets as tfds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the music model.')
    parser.add_argument('--batch_size', type=int,
                        help='the batch size', default=500)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--epochs', type=int, help='Training epoch amount', default=10)
    parser.add_argument('--triplet_model', type=str, help='Path to a savedmodel of triplet model',
                        default='saved_models/triplet/efficientnet_v1.1')
    parser.add_argument('--tag', type=str, help='A version to tag the final output', default='v1')
    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    tag = args.tag
    triplet_path = args.triplet_model

    print("Loading model {}".format(triplet_path))
    model = tf.keras.models.load_model(triplet_path, custom_objects={'tf': tf}, compile=False)

    test_dataset = SoundSequence(os.path.join(os.path.dirname(__file__), 'music'), use_categorical=False,
                                 shuffle=True, is_autoencoder=False, use_raw_audio=True,
                                 batch_size=batch_size, subset='validation')

    # Evaluate the network
    results = model.predict(test_dataset)

    # Save test embeddings for visualization in projector
    visualization_path = os.path.join(os.path.dirname(__file__), 'visualizations')
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    np.savetxt(os.path.join(visualization_path, "vecs.tsv"), results, delimiter='\t')

    out_m = io.open(os.path.join(visualization_path, 'meta.tsv'), 'w', encoding='utf-8')
    for wav, labels in test_dataset:
        [out_m.write(str(x) + "\n") for x in labels]
    out_m.close()

    print("Embeddings written to {}".format(visualization_path))
