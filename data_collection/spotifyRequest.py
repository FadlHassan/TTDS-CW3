import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from musixmatch import Musixmatch

def connectSpotify():
    #### CONNECTING TO SPOTIFY API ####
    cid ='6eab62e055d94fad926550e22f78bd4a' # Client id (NEED TO CHANGE)
    secret ='18b3117c2ce240e6b2e13b53adba62ea'# Secret key(NEED TO CHANGE)

    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    return sp

def connectMusixMatch():
    apikey = '79bbd0b21e73f1e50187712f0d5cdc64'
    musixmatch = Musixmatch(apikey)
    return musixmatch

def searchForTrack(sp,track):
    try:
        track_info = sp.search(track)
        return track_info
    except:
        print("Can't find track on Spotify")
        return None

def getImageOfTrack(track_info):
    try:
        return track_info['tracks']['items'][0]['album']['images'][0]['url']
    except:
        return None
    

def getGenreOfTrack(musixmatch, artistName, trackName):
    try:
        mx_track = musixmatch.matcher_track_get(artistName, trackName)
        return mx_track['message']['body']['track']['primary_genres']['music_genre_list'][0]['music_genre']['music_genre_name']
    except Exception as e:
        print(e)
        return None


def main():
    print("Connecting to Spotify")
    sp = connectSpotify()
    print("Connecting to Musixmatch")
    musixmatch = connectMusixMatch()
    artistName = 'Taylor Swift'
    trackName = 'Blank Space'
    q = "artist: {} track: {}".format(artistName,trackName)
    print(q)
    track_info = searchForTrack(sp, q)
    if track_info:
        image = getImageOfTrack(track_info)
        if image is not None:
            print("Image link: ", image)
        genre = getGenreOfTrack(musixmatch, artistName, trackName)
        if genre is not None:
            print("Genre: ", genre)



if __name__ == '__main__':
    main()


