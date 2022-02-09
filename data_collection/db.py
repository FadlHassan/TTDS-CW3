import sqlite3
from sqlite3 import Error


def main():
    database = r"pythonsqlite.db"

    # Add data to the tables
    
    # create a database connection
    conn = create_connection(database)
    with conn:
        
        # # create a new artist
        # artist_1 = ("The Weeknd");
        # artist_id = create_artist(conn, artist_1)

        # # create new tracks
        # track_1 = ('Save Your Tears', '2015-01-01', 'pop', "Save your tears", 'picture_save_your_tears', artist_id)
        # track_2 = ('Blinding Lights', '2015-01-01', 'pop', "I've been trynna call", 'picture_blinding_lights', artist_id)
        # track_3 = ('Hills', '2015-01-01', 'pop', "I only call you when its half past 5 the only time I ever call you mine.", 'picture_hills', artist_id)

        # # METHODS TO CREATE Artist & Track
        # create_track(conn, track_1)
        # create_track(conn, track_2)


        # # create a new artist
        # artist_2 = ("Post Malone");
        # artist_id = create_artist(conn, artist_2)

        # # create new tracks
        # track_1 = ('Circle', '2015-01-01', 'pop', "We couldn't turn around 'Til we were upside down I'll be the bad guy now But no, I ain't too proud", 'picture_circles', artist_id)
        # track_2 = ('Flex', '2015-01-01', 'pop', "Lighting stog after stog, choke on the smoke. They tell me to quit, don't listen what. I'm told Help me forget that this world is so cold", 'picture_flex', artist_id)

        # METHODS TO CREATE Artist & Track
        # create_track(conn, track_1)
        # create_track(conn, track_2)

        # METHODS TO GET all the Artists & Tracks
        print("--- All Artists")
        select_all_artists(conn)

        print("--- All Tracks")
        select_all_tracks(conn)

        # METHOD to get all Tracks from a particular Artist
        print("--- Tracks from The Weeknd")
        select_tracks_by_artist(conn, "The Weeknd")

        # METHOD to get all Tracks from a particular Artist
        print("--- Tracks from Post Malone")
        select_tracks_by_artist(conn, "Post Malone")





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

def create_artist(conn, artist):
    """
    Create a new artist into the artists table
    :param conn:
    :param artist:
    :return: artist id
    """
    sql = ''' INSERT INTO artists(name)
              VALUES(?) '''
    cur = conn.cursor()
    cur.execute(sql, (artist,))
    conn.commit()
    return cur.lastrowid

def create_track(conn, track):
    """
    Create a new track
    :param conn:
    :param track:
    :return: track id
    """

    sql = ''' INSERT INTO tracks(title,date,genre,lyrics,picture,artistId)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, track)
    conn.commit()
    return cur.lastrowid


def select_all_artists(conn):
    """
    Query all rows in the artists table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM artists")

    rows = cur.fetchall()

    for row in rows:
        print(row)


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
    cur.execute("SELECT * FROM artists A JOIN tracks T ON A.id = T.artistId WHERE A.name =?", (artistName,))

    rows = cur.fetchall()

    for row in rows:
        print(row)


if __name__ == '__main__':
    main()