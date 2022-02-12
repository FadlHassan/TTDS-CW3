import sqlite3
from sqlite3 import Error

import lyricsgenius
import pandas as pd
import csv
import sys


def main():

    database = ""

    if(sys.argv[1] == "0"):
        database = r"lyricsDB_0.db"
    else if(sys.argv[1] == "1"):
        database = r"lyricsDB_1.db"
    else if(sys.argv[1] == "2"):
        database = r"lyricsDB_2.db"
    else if(sys.argv[1] == "3"):
        database = r"lyricsDB_3.db"
    else if(sys.argv[1] == "4"):
        database = r"lyricsDB_4.db"
    else if(sys.argv[1] == "5"):
        database = r"lyricsDB_5.db"
    else:
        print("Wrong argument")
        System.exit(0)

    # Add data to the tables
    
    # create a database connection
    conn = create_connection(database)

    with conn:


        # Table to save all songs
        # All_Songs_Table = pd.DataFrame(columns=['artist','title','lyrics'])
        # LyricsGenius = connectToGenius()
        # artists = getArtists()
        # artist_cat = [(0, 225499), (225499, 450998), (450998, 676497), (676497, 901996), (901996, 1127495), (1127495, 1352994)]
        # getLyrics(LyricsGenius, artists, artist_cat, conn)
        

        # # Create new tracks
        track_1 = ('Passion Fruit', "Passion fruit lyrics", "Drake")
        track_2 = ('Toosie Slide', "Toosie slide lyrics", "Drake")
        track_3 = ('Free Smoke', "Free smoke lyrics", "Drake")
        # track_4 = ('Astroworld', "Astroworld lyrics", "Travis Scott")
        # track_5 = ('London', "London lyrics", "Travis Scott")

        # # METHODS TO CREATE Artist & Track
        create_track(conn, track_1)
        create_track(conn, track_2)
        create_track(conn, track_3)
        # create_track(conn, track_4)
        # create_track(conn, track_5)

        # # METHODS TO GET all the Artists & Tracks

        # print("--- All Tracks")
        # select_all_tracks(conn)

        # # METHOD to get all Tracks from a particular Artist
        # print("--- Tracks from The Weeknd")
        # select_tracks_by_artist(conn, "The Weeknd")

        # # METHOD to get all Tracks from a particular Artist
        # print("--- Tracks from Post Malone")
        # select_tracks_by_artist(conn, "Post Malone")



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

def getLyrics(LyricsGenius, artists, artist_cat, conn):
    Collected_Lyrics = 0 
    count = 0
    cat = artist_cat[int(sys.argv[1])]
    for i in range(cat[0],cat[1]):
        Artist_Name = artists[i]

        try:
            artist = LyricsGenius.search_artist(Artist_Name, max_songs=10)
            for song in artist.songs:
                lyrics = song.lyrics   
                lyrics = lyrics.replace(song.title + ' Lyrics', '')


                # add artist & lyrics to DB using the "song" instance
                track = (song.title, lyrics, song.artist)
                create_track(conn, track)

                # All_Songs_Table = All_Songs_Table.append({'artist': song.artist, 'title':song.title,'lyrics':lyrics}, ignore_index=True) 
                # headerList = ['Artist', 'Song', 'Lyrics']
                
                # lines_written = len(All_Songs_Table.index)
                # All_Songs_Table.iloc[lines_written-1:].to_csv('Songs_Table.csv', mode='a', header=False)
        except:
            continue




def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_track(conn, track):
    """
    Create a new track
    :param conn:
    :param track:
    :return: track id
    """

    sql = ''' INSERT INTO tracks(title,lyrics,artist)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, track)
    conn.commit()
    return cur.lastrowid


def select_all_tracks(conn):
    """
    Query all rows in the tracks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tracks")

    rows = cur.fetchall()

    for row in rows:
        print(row)

def select_tracks_by_artist(conn, artistName):
    """
    Query tracks by artist
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tracks WHERE artist =?", (artistName,))

    rows = cur.fetchall()

    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
