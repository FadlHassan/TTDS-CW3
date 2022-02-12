import lyricsgenius
import pandas as pd
import csv
import sys

def connectToGenius():
    client_access_token = 'v1RaGoN7_VBkkydj6ZpiK9wlaF-CbW-DIf_wgENhQETztCWFLpRyaHfs8THDSMfC'
    LyricsGenius = lyricsgenius.Genius(client_access_token)
    LyricsGenius.timeout = 5
    LyricsGenius.sleep_time = 0.2
    LyricsGenius.retries = 3

    return LyricsGenius

def getArtists():
    col_list = ['artist_mb', 'scrobbles_lastfm']
    df = pd.read_csv('artists.csv', usecols=col_list)
    df = df.sort_values(by=['scrobbles_lastfm'], ascending = False)
    artists_list = df.values.tolist()
    artists = [artist[0] for artist in artists_list[:50000]]
    return artists

def getLyrics(LyricsGenius, artists, artist_cat,All_Songs_Table):
    Collected_Lyrics = 0 
    count = 0
    cat = artist_cat[int(sys.argv[1])]
    for i in range(cat[0],cat[1]):
        Artist_Name = artists[i]
        try:
            artist = LyricsGenius.search_artist(Artist_Name, max_songs=100)
            for song in artist.songs:
                lyrics = song.lyrics   
                lyrics = lyrics.replace(song.title + ' Lyrics', '')

                All_Songs_Table = All_Songs_Table.append({'artist': song.artist, 'title':song.title,'lyrics':lyrics}, ignore_index=True) 
                headerList = ['Artist', 'Song', 'Lyrics']
                
                lines_written = len(All_Songs_Table.index)
                All_Songs_Table.iloc[lines_written-1:].to_csv('Songs_Table.csv', mode='a', header=False)
        except:
            continue

# Table to save all songs
All_Songs_Table = pd.DataFrame(columns=['artist','title','lyrics'])
LyricsGenius = connectToGenius()
artists = getArtists()
artist_cat = [(0, 225499), (225499, 450998), (450998, 676497), (676497, 901996), (901996, 1127495), (1127495, 1352994)]
getLyrics(LyricsGenius, artists, artist_cat,All_Songs_Table)


