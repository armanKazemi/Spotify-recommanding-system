# get arguments from command line
import sys
import os
import json
from pickle import load

import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


def get_features():
    playlist_tracks = sp.playlist("0FHTNm52sH5iebCJHcxbhK?si=goF1uuaOSzKfKS51dJpylg&utm_source=copy-link")["tracks"]
    track_ids = [x["track"]["id"] for x in playlist_tracks["items"]]
    return sp.audio_features(track_ids)


def predict(tracks):
    kmeans = load(open("./kmeans.pkl", "rb"))
    scaler = load(open("./scaler.pkl", "rb"))
    best_cols = ['instrumentalness', 'speechiness', 'acousticness']
    df = pd.DataFrame(tracks)
    x = scaler.transform(df[best_cols])
    clusters = kmeans.predict(x)
    df["cluster"] = clusters
    return df


def get_title_and_artists(playlist):
    res = []
    for track in sp.tracks(playlist)["tracks"]:
        res.append({
            "artists": ", ".join([x["name"] for x in track["artists"]]),
            "name": track["name"],
            "link": track["external_urls"]["spotify"]
        })
    return res


def create_playlists(matched):
    print(matched)
    hot_tracks = json.load(open("hot_tracks.json", "r"))
    c = 1
    total_df = pd.DataFrame(columns=['id'])
    for k, v in matched.items():
        for i in range(v):
            playlist = hot_tracks[str(k - 1)][i:v * 5:v]
            infos = get_title_and_artists(playlist)
            df = pd.DataFrame(infos, columns=['id'])
            total_df.append(df)
            df.to_csv("playlist_%d_%d.csv" % (c, k))
            c += 1
    total_df.to_csv('total_playlist.csv', )


def main(args) -> None:
    tracks = get_features()
    df = predict(tracks)
    counts = dict([(x, 0) for x in range(1, 8)])
    for _, row in df.iterrows():
        counts[row["cluster"]] += 1

    new_counts = dict()
    for k, v in counts.items():
        if float(v) / float(len(tracks)) >= 1.0 / 7.0:
            new_counts[k] = v

    result = dict()

    while sum([v for _, v in result.items()]) < 5:
        remain_count = sum([v for _, v in new_counts.items()])
        remain_list = sorted(list(new_counts.items()), key=lambda x: x[1])

        c = round((float(remain_list[-1][1]) / float(remain_count)) * (5 - sum([v for _, v in result.items()])))
        result[remain_list[-1][0]] = c

        del (new_counts[remain_list[-1][0]])

    create_playlists(result)


if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv
    main(args)
