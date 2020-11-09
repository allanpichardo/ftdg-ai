import argparse
from dotenv import load_dotenv
import os
from urllib.parse import urljoin
from src.sound_sequence import SoundSequence
import tensorflow as tf
from sklearn.manifold import TSNE
import psycopg2
from psycopg2.extras import execute_values


def get_url_from_filename(filename):
    basename = os.path.basename(filename)
    parts = basename.split('-')
    url = "{}/{}/{}?cid={}".format('https://p.scdn.co', 'mp3-preview', parts[0], os.environ['SPOTIPY_CLIENT_ID'])
    return url


def insert_data(row_data, cursor):
    sql = "INSERT INTO public.music(id, embedding, x, y, z, origin) VALUES %s, cube(%s::float8[]), %s, %s, %s, %s)"
    execute_values(cursor, sql, row_data)


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description='Populate the database.')
    parser.add_argument('--music_path', type=str, help='Base path of music files',
                        default=os.path.join(os.path.dirname(__file__), 'music'))
    parser.add_argument('--triplet_model', type=str, help='Path to a savedmodel of triplet model',
                        default='saved_models/triplet/efficientnet_v1.1')
    args = parser.parse_args()

    music_path = args.music_path
    model_path = args.triplet_model

    sequences = SoundSequence(music_path=music_path, use_categorical=False, shuffle=False, subset='validation')
    X, paths, labels = sequences.get_all()

    labels = sequences.encoder.inverse_transform(labels)

    urls = [get_url_from_filename(x) for x in paths]

    print("Loading triplet model from {}".format(model_path))
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'tf': tf})

    print("Computing embeddings...")
    Y = model.predict(X)

    print("Computing TSNE...")
    tsne = TSNE(n_components=3, verbose=1, n_jobs=-1).fit_transform(Y)

    row_data = []
    for i in range(len(tsne)):
        vector = Y[i]
        coords = tsne[i]
        url = urls[i]
        origin = labels[i]
        data = (url, vector, coords[0], coords[1], coords[2], origin)
        row_data.append(data)

    print("Connecting to database at {}:{}".format(os.environ['DB_HOST'], os.environ['DB_PORT']))
    conn = psycopg2.connect(host=os.environ['DB_HOST'], port=os.environ['DB_PORT'],
                            user=os.environ['DB_USERNAME'], password=os.environ['DB_PASSWORD'],
                            dbname='ftdg')
    cursor = conn.cursor()
    print("Inserting all {} rows".format(len(row_data)))
    insert_data(row_data, cursor)

    print("Insert complete.")
