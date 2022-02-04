import lyricsgenius
import pandas as pd
import csv

# Table to save all songs
All_Songs_Table = pd.DataFrame(columns=['artist','title','lyrics'])

def connectToGenius():
    client_access_token = 'v1RaGoN7_VBkkydj6ZpiK9wlaF-CbW-DIf_wgENhQETztCWFLpRyaHfs8THDSMfC'
    LyricsGenius = lyricsgenius.Genius(client_access_token)
    LyricsGenius.timeout = 5
    LyricsGenius.sleep_time = 0.2
    LyricsGenius.retries = 3

    return LyricsGenius

def getArtists():
    col_list = ['artist_mb']
    df = pd.read_csv('artists.csv', usecols=col_list)
    artists = df.values.tolist() 
    artists = [artist[0] for artist in artists]
    artists = list(dict.fromkeys(artists))
    return artists

def getLyrics(LyricsGenius, artists):
    Collected_Lyrics = 0 
    for Artist_Name in artists:
        try:
            artist = LyricsGenius.search_artist(Artist_Name, max_songs=100000)
            for song in artist.songs:
                artist = song.artist
                title = song.title
                lyrics = song.lyrics   
                lyrics.replace(title+' Lyrics', '')   

                All_Songs_Table = All_Songs_Table.append({'artist': artist, 'title':title,'lyrics':lyrics}, ignore_index=True) 
                headerList = ['Artist', 'Song', 'Lyrics']
                
                lines_written = len(All_Songs_Table.index)
                All_Songs_Table.iloc[lines_written-1:].to_csv('Songs_Table.csv', mode='a', header=False)
        except:
            continue

LyricsGenius = connectToGenius()
artists = getArtists()
getLyrics(LyricsGenius, artists)


