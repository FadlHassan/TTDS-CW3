import sqlite3
from sqlite3 import Error

# import lyricsgenius
import pandas as pd
import csv
import sys
import os
import math
import re
import numpy as np
from scipy.sparse import dok_matrix
import random
from collections import Counter
from stemming.porter2 import stem
# from gensim.corpora.dictionary import Dictionary
# from gensim.models import LdaModel
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC, LinearSVC



def main():

    database = "lyricsDB.db"

    # if(sys.argv[1] == "0"):
    #     database = r"lyricsDB_0.db"
    # elif(sys.argv[1] == "1"):
    #     database = r"lyricsDB_1.db"
    # elif(sys.argv[1] == "2"):
    #     database = r"lyricsDB_2.db"
    # elif(sys.argv[1] == "3"):
    #     database = r"lyricsDB_3.db"
    # elif(sys.argv[1] == "4"):
    #     database = r"lyricsDB_4.db"
    # elif(sys.argv[1] == "5"):
    #     database = r"lyricsDB_5.db"
    # elif(sys.argv[1] == "6"):
    #     database = r"lyricsDB_6.db"
    # elif(sys.argv[1] == "7"):
    #     database = r"lyricsDB_7.db"
    # elif(sys.argv[1] == "8"):
    #     database = r"lyricsDB_8.db"
    # elif(sys.argv[1] == "9"):
    #     database = r"lyricsDB_9.db"
    # else:
    #     print("Wrong argument")
    #     System.exit(0)

    # Add data to the tables
    
    # create a database connection
    conn = create_connection(database)

    with conn:


        # Table to save all songs
        # All_Songs_Table = pd.DataFrame(columns=['artist','title','lyrics'])
        # LyricsGenius = connectToGenius()
        # artists = getArtists()
        # artist_cat = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000), 
        #                 (20000, 25000), (25000, 30000), (30000, 35000), 
        #                 (35000, 40000), (40000, 45000), (45000, 50000)]
        # getLyrics(LyricsGenius, artists, artist_cat, conn)
        

        # # # Create new tracks
        # track_1 = ('Passion Fruit', "Passion fruit lyrics", "Drake")
        # track_2 = ('Toosie Slide', "Toosie slide lyrics", "Drake")
        # track_3 = ('Free Smoke', "Free smoke lyrics", "Drake")
        # # track_4 = ('Astroworld', "Astroworld lyrics", "Travis Scott")
        # # track_5 = ('London', "London lyrics", "Travis Scott")

        # # # METHODS TO CREATE Artist & Track
        # create_track(conn, track_1)
        # create_track(conn, track_2)
        # create_track(conn, track_3)
        # create_track(conn, track_4)
        # create_track(conn, track_5)

        # METHODS TO GET all the Artists & Tracks

        print("--- All Tracks")
        # rows = select_all_tracks(conn)
        rows = select_all_tracks(conn)
        tracks = {}

        for row in rows:
            tracks[row[1]] = row[2]
            # tracks.append(track)

        print("----- Reading the Terms -----")
        terms = readDataTextClassification(tracks)
        print("----- Creating the Inverted Index -----")
        inverted, totalNumberOfDocuments = invertedIndex(terms, tracks)
        processingRankQueries(totalNumberOfDocuments,inverted)



        # # METHOD to get all Tracks from a particular Artist
        # print("--- Tracks from The Weeknd")
        # select_tracks_by_artist(conn, "The Weeknd")

        # # METHOD to get all Tracks from a particular Artist
        # print("--- Tracks from Post Malone")
        # select_tracks_by_artist(conn, "Post Malone")



def searchLyrics(searchedKeyword):

    database = "lyricsDB_0.db"
    
    # create a database connection
    conn = create_connection(database)

    with conn:

        # METHODS TO GET all the Artists & Tracks

        print("--- All Tracks")
        # rows = select_all_tracks(conn)
        rows = select_top_tracks(conn, 5000)
        tracks = {}

        for row in rows:
            tracks[row[1]] = row[2]

        print("----- Reading the Terms -----")
        terms = readDataTextClassification(tracks)
        print("----- Creating the Inverted Index -----")
        inverted, totalNumberOfDocuments = invertedIndex(terms, tracks)

        processingRankQueries([searchedKeyword], totalNumberOfDocuments,invertedIndex)


# Method read and process the data
def readDataTextClassification(tracks):
    
    print('----- Processing the data----- ')

    terms = set()
    
    # N = len(lines)

    # Strips the newline character
    for song in tracks:
        lyrics = tracks[song] # Case Folding
        for term in lyrics:
            terms.add(term)
    
    # print('Len data: ', len(data))
    # print('Len unique terms: ', len(terms))
    return list(terms)

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
            artist = LyricsGenius.search_artist(Artist_Name, max_songs=100)
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

    # for row in rows:
    #     print(row)
    return rows


def select_top_tracks(conn, num):
    """
    Query all rows in the tracks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tracks LIMIT " + str(num))

    rows = cur.fetchall()

    # for row in rows:
    #     print(row)
    return rows

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

def preProcessing(document):
    tokens = re.sub(r"[^\w]+", " ", document).lower() #tokenisation
    tokenisation = tokens.split(" ")
    terms = [word for word in tokenisation] #stop word removal
    return terms


def queryPreProcessing(query): #preprocessing the boolean queries
    chars = '[!"$%&\()*+,-./:;<=>?@[\\]^_{|}~]+|\r|\t'
    query = re.sub(chars, " ", query).lower().split(" ")
    query_processed = [word for word in query]
    # stopWordRemoval = [word for word in query if word not in connectingStopWords and word !=""]
    # stemming = [stemmer.stem(word) for word in stopWordRemoval]
    return query_processed

def queryProcessing(query,totalNumberOfDocuments,invertedIndex):
    def wordIndexing(word,dictionary): #indexing for when the query has only one word
        queryList = []
        for key in dictionary[word]:
            queryList.append(key)
        return queryList
    
    preprocessedQuery = queryPreProcessing(query)
    andFlag = False
    orFlag = False
    notFlag = False
    l1 = []
    l2 = []
    for query in range(len(preprocessedQuery)):
        if preprocessedQuery[query] == "and" and "not" not in preprocessedQuery: #A and B
            andFlag = True
            l1 = preprocessedQuery[:query]
            l2 = preprocessedQuery[query+1:]
        elif preprocessedQuery[query] == "or" and "not" not in preprocessedQuery: #A or B
            orFlag = True
            l1 = preprocessedQuery[:query]
            l2 = preprocessedQuery[query+1:]
        elif preprocessedQuery[query] == "not" and query == 0: 
            notFlag = True
            if "and" in preprocessedQuery:#A and not B
                andIndex = preprocessedQuery.index("and")
                andFlag = True
                l1 = preprocessedQuery[andIndex+1:]
                l2 = preprocessedQuery[1:andIndex]
            elif "or" in preprocessedQuery:#A and or B
                orIndex = preprocessedQuery.index("or")
                orFlag = True
                l1 = preprocessedQuery[orIndex+1:]
                l2 = preprocessedQuery[1:orIndex]
            else:
                l1 = preprocessedQuery[1:]
        elif preprocessedQuery[query] == "not" and query != 0:
            notFlag = True
            if preprocessedQuery[query - 1] == "and": #not A and B
                andFlag = True
                l1 = preprocessedQuery[:query - 1]
                l2 = preprocessedQuery[query+1:]
            elif preprocessedQuery[query - 1] == "or": #not A or B
                orFlag = True
                l1 = preprocessedQuery[:query - 1]
                l2 = preprocessedQuery[query+1:]
            else: #not A
                l1 = preprocessedQuery[1:]
    totalList = [x for x in range(1,len(invertedIndex))]
    totalSet = set(totalList)
    if andFlag == False and orFlag == False and notFlag == False:
        if len(preprocessedQuery) == 1:
            queryList = wordIndexing(preprocessedQuery[0],invertedIndex)
        elif len(preprocessedQuery) == 2: 
            queryList = phraseAndProximitySearch(preprocessedQuery[0],preprocessedQuery[1],invertedIndex,1)
        elif len(preprocessedQuery) == 3: 
            distance = int(preprocessedQuery[0].strip("#"))
            queryList = phraseAndProximitySearch(preprocessedQuery[1],preprocessedQuery[2],invertedIndex,distance)
    else:
        if andFlag == False and orFlag == False and notFlag == True:
            if len(l1) == 1:
                tempQueryList = wordIndexing(l1[0],invertedIndex)
                tempQuerySet = set(tempQueryList)
                queryList = list(tempQuerySet.difference(totalSet))
            elif len(l1) == 2:
                tempQueryList = phraseAndProximitySearch(l1[0],l1[1],invertedIndex,1)
                tempQuerySet = set(tempQueryList)
                queryList = list(tempQuerySet.difference(totalSet))
        else:
            if len(l1) == 1:
                tempQueryList = wordIndexing(l1[0],invertedIndex)
            else:
                tempQueryList = phraseAndProximitySearch(l1[0],l1[1],invertedIndex,1)
            if len(l2) == 1:
                tempQueryList2 = wordIndexing(l2[0],invertedIndex)
            else:
                tempQueryList2 = phraseAndProximitySearch(l2[0],l2[1],invertedIndex,1)
            queryList = booleanSearch(tempQueryList,tempQueryList2,andFlag,orFlag,notFlag,totalList)
    return sorted(queryList)

def invertedIndex(terms, documents):
    total_num = len(documents) 
    invertedIndex = {} #dictionary of positional inverted index
    for title in documents:
        lyrics = documents[title]
        terms = preProcessing(lyrics)
        count = 0
        #store inverted index
        for term in terms:
            if term not in invertedIndex:
                invertedIndex[term] = {}
            if title not in invertedIndex[term]:
                invertedIndex[term][title] = []
                invertedIndex[term][title].append(str(count))
            else:
                invertedIndex[term][title].append(str(count))
            count += 1

    f1 = "index.txt"
    for term in invertedIndex:
        with open(f1, "a+") as file:
            file.write(term + ": " + str(len(invertedIndex[term])) + "\n")
            for doc in invertedIndex[term]:
                file.write("\t" + doc + ": " + ",".join(invertedIndex[term][doc]) + "\n")
    return invertedIndex,total_num

def processingRankQueries(rankQueries, totalNumberOfDocuments,invertedIndex):
    file1 = "results.ranked.txt"
    # rankQueries = ["Don't carry the world upon your shoulders"]
    queryTerms = []
    for query in rankQueries:
        queryTerms.append(preProcessing(query))
    documentDictionary = {}
    for query in range(len(queryTerms)):
        documentDictionary[query] = {}
        for q in queryTerms[query][1:]:
            documentNumber = query + 1
            documentList = queryProcessing(q,totalNumberOfDocuments,invertedIndex) #processing the queries
            df = len(invertedIndex[q])
            for document in documentList:
                if document not in documentDictionary[query]:
                    documentDictionary[query][document] = 0
                tf = len(invertedIndex[q][str(document)])
                score = (1 + math.log10(tf)) * math.log10(totalNumberOfDocuments/df) #calculating TFIDF score
                documentDictionary[query][document] += score
    file1 = "results.ranked.txt"
    for query in range(len(queryTerms)):
        for rank in sorted(documentDictionary[query].items(), key = lambda x:(x[1],x[0]) , reverse = True)[:150]:
            with open(file1, "a+") as file:
                file. write(str(query+1) + "," + str(rank[0]) + "," + "{:.4f}".format(rank[1]) + "\n") #writing the top 150 results to file




if __name__ == '__main__':
    main()
