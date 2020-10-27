import pathlib

import librosa
from dotenv import load_dotenv
import os
import json
import soundfile as sf
import urllib.parse
import io
import urllib.request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

auth_manager = None
spotify = None
output_directory = None


def get_previews_from_playlist(uri):
    previews = []
    playlist = spotify.playlist(uri, market='ES')
    tracks = playlist['tracks']
    while tracks['next']:
        tracks = spotify.next(tracks)
        for item in playlist['tracks']['items']:
            p = item['track']['preview_url']
            if p is not None:
                previews.append(p)
    return previews


def download_previews(previews):
    return


def get_all_playlists(json):
    for kingdom, playlists in json.items():
        kingdom_previews = []
        print("Kingdom: {}".format(kingdom))
        for playlist in playlists:
            kingdom_previews = kingdom_previews + get_previews_from_playlist(playlist)
            print(".", sep="")

        if not os.path.exists('music'):
            os.mkdir('music')

        if not os.path.exists(os.path.join('music', 'training')):
            os.mkdir(os.path.join('music', 'training'))

        if not os.path.exists(os.path.join('music', 'validation')):
            os.mkdir(os.path.join('music', 'validation'))

        if not os.path.exists(os.path.join('music', 'training', kingdom)):
            os.mkdir(os.path.join('music', 'training', kingdom))

        if not os.path.exists(os.path.join('music', 'validation', kingdom)):
            os.mkdir(os.path.join('music', 'validation', kingdom))

        for preview in kingdom_previews:
            filename = os.path.splitext(os.path.basename(urllib.parse.urlparse(preview).path))[0]
            print("Processing {}:".format(filename))
            if os.path.exists(os.path.join('music', 'training', kingdom, '{}-a.wav'.format(filename))):
                print("Duplicate clip. Skipping...")
                continue

            z = io.BytesIO(urllib.request.urlopen(preview).read())
            pathlib.Path('temp.mp3').write_bytes(z.getbuffer())
            data, sr = librosa.load('temp.mp3')
            partition = len(data) // 3
            a = data[0:partition]
            b = data[partition:partition*2]
            c = data[partition*2:-1]
            print("Writing files...")
            sf.write(os.path.join('music', 'training', kingdom, '{}-a.wav'.format(filename)), a, sr, 'PCM_16')
            sf.write(os.path.join('music', 'training', kingdom, '{}-b.wav'.format(filename)), b, sr, 'PCM_16')
            sf.write(os.path.join('music', 'validation', kingdom, '{}-c.wav'.format(filename)), c, sr, 'PCM_16')


def main():
    global auth_manager
    global spotify

    load_dotenv()
    scope = "user-library-read"
    # auth_manager = SpotifyClientCredentials()
    auth_manager = SpotifyOAuth(scope=scope)
    spotify = spotipy.Spotify(auth_manager=auth_manager)

    # with open('playlists.json', 'r') as file:
    #     playlists = json.load(file)
    #     get_all_playlists(playlists)

    previews = get_previews_from_playlist('spotify:playlist:38bzETkEYJNkcR8vhqMlMz')
    print(previews)


if __name__ == '__main__':
    main()
