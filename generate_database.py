import argparse
from dotenv import load_dotenv
import os
from urllib.parse import urljoin
from src.sound_sequence import SoundSequence
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import math


def get_url_from_filename(filename):
    basename = os.path.basename(filename)
    parts = basename.split('-')
    url = "{}/{}/{}?cid={}".format('https://p.scdn.co', 'mp3-preview', parts[0], os.environ['SPOTIPY_CLIENT_ID'])
    return url


def l2_normalize(v):
    norm = np.sqrt(np.sum(np.square(v)))
    return v / norm


def maginitude(v):
    mag = 0
    for n in v:
        mag = mag + (n * n)
    return math.sqrt(mag)


def insert_data(row_data, cursor):
    sql = """INSERT INTO public.music (url, embedding, x, y, z, origin, magnitude) VALUES %s"""
    execute_values(cursor, sql, row_data, template='''(%s, cube(%s::float8[]), %s, %s, %s, %s, %s)''')


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description='Populate the database.')
    parser.add_argument('--music_path', type=str, help='Base path of music files',
                        default=os.path.join(os.path.dirname(__file__), 'music'))
    parser.add_argument('--triplet_model', type=str, help='Path to a savedmodel of triplet model',
                        default='saved_models/triplet/efficientnet_v3.0')
    parser.add_argument('--normalize', type=bool, help='Normalize embeddings', default=False)
    args = parser.parse_args()

    music_path = args.music_path
    model_path = args.triplet_model
    normalized = args.normalize

    sequences = SoundSequence(music_path=music_path, use_categorical=False, shuffle=False, subset='validation', batch_size=4)
    X, paths, labels = sequences.get_all()

    labels = sequences.encoder.inverse_transform(labels)

    urls = [get_url_from_filename(x) for x in paths]

    print("Loading triplet model from {}".format(model_path))
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'tf': tf})

    print("Computing embeddings...")
    Y = model.predict(sequences)

    print("Computing TSNE...")
    # tsne = PCA(n_components=3).fit_transform(Y)
    # tsne = TSNE(n_components=3, verbose=1, n_jobs=-1, perplexity=5, n_iter=250).fit_transform(Y)
    tsne = np.random.rand(len(Y), 3)

    row_data = []
    for i in range(len(tsne)):

        normed = l2_normalize(Y[i]) if normalized else Y[i]
        vector = []
        for j in range(len(normed)):
            vector.append(normed.item(j))

        length = maginitude(vector)
        coords = tsne[i]
        url = urls[i]
        origin = labels[i]
        data = (url, vector, coords.item(0), coords.item(1), coords.item(2), origin, length)
        row_data.append(data)

    print("Connecting to database at {}:{}".format(os.environ['DB_HOST'], os.environ['DB_PORT']))
    conn = psycopg2.connect(host=os.environ['DB_HOST'], port=os.environ['DB_PORT'],
                            user=os.environ['DB_USERNAME'], password=os.environ['DB_PASSWORD'],
                            dbname='ftdg')
    cursor = conn.cursor()
    print("Inserting all {} rows".format(len(row_data)))
    insert_data(row_data, cursor)
    print("Committing...")
    conn.commit()
    cursor.close()
    conn.close()

    print("Insert complete.")
