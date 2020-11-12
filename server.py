import pathlib
import librosa
from flask import Flask
from flask import request
from dotenv import load_dotenv
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.parse
import io
import urllib.request
import os
import tensorflow as tf
import psycopg2
from flask import jsonify

load_dotenv()
app = Flask(__name__)
auth_manager = SpotifyClientCredentials()
spotify = spotipy.Spotify(auth_manager=auth_manager)

model_path = os.path.join('saved_models', 'triplet', 'efficientnet_v1.1')
if not os.path.exists(model_path):
    print("Couldn't find model in {}".format(model_path))
    exit(1)

print("Loading tensorflow model")
model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf}, compile=False)

print("Connecting to database at {}:{}".format(os.environ['DB_HOST'], os.environ['DB_PORT']))
conn = psycopg2.connect(host=os.environ['DB_HOST'], port=os.environ['DB_PORT'],
                        user=os.environ['DB_USERNAME'], password=os.environ['DB_PASSWORD'],
                        dbname='ftdg')

america = {"Brazil", "Detroit", "Chicago", "Memphis", "New York City", "Baltimore", "Atlanta", "New Orleans",
           "Puerto Rico", "Colombia", "Dominican Republic", "Barbados", "Jamaica", "Trinidad and Tobago", "Honduras",
           "Haiti"}

africa = {"Senegal", "Ghana", "Angola", "Benin", "Nigeria"}


def l2_normalize(v):
    norm = np.sqrt(np.sum(np.square(v)))
    return v / norm


def read_mp3_data(url, duration=8.0):
    z = io.BytesIO(urllib.request.urlopen(url).read())
    pathlib.Path('temp.mp3').write_bytes(z.getbuffer())
    data, sr = librosa.load('temp.mp3')
    partition = len(data) // 3
    wav = data[partition * 2:-1]
    wav = wav[:sr * int(duration)]
    return wav.reshape(1, -1)


def get_track_preview(q):
    res = spotify.search(q)
    if res['tracks']:
        if res['tracks']['items'][0]:
            return res['tracks']['items'][0]['preview_url']
    return None


def get_first_neighbor(embedding, origins):
    cursor = conn.cursor()
    cursor.execute(
        "select embedding, x, y, x, origin, url from public.music where origin in %s order by embedding <-> cube(%s::float8[]) asc limit 1",
        (tuple(origins), embedding))
    results = cursor.fetchone()
    next_origin = results[4]
    next_embedding = list(eval(results[0]))
    origins.remove(next_origin)
    return results, next_embedding, origins


def get_embeddings_from_pcm(pcm):
    pcm = np.array([pcm])
    embeddings = model.predict([pcm])
    normed = l2_normalize(embeddings[0])
    vector = []
    for j in range(len(normed)):
        vector.append(normed.item(j))
    return vector


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/search')
def search():
    try:
        query = request.args.get('q')
        url = get_track_preview(query)
        pcm = read_mp3_data(url)
        embeddings = get_embeddings_from_pcm(pcm)
        am = america.copy()
        treks = {
            "success": True,
            "query": query,
            "query_url": url,
            "constellation": []
        }
        while len(am) > 0:
            results, next_embedding, new_am = get_first_neighbor(embeddings, am)
            treks['constellation'].append({
                "x": results[1],
                "y": results[2],
                "z": results[3],
                "origin": results[4],
                "url": results[5]
            })
            # embeddings = next_embedding
            am = new_am
        results, next_embedding, new_am = get_first_neighbor(embeddings, africa.copy())
        treks['constellation'].append({
            "x": results[1],
            "y": results[2],
            "z": results[3],
            "origin": results[4],
            "url": results[5]
        })
        return jsonify(treks)
    except AttributeError:
        return jsonify({
            "success": False
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ['PORT'])
