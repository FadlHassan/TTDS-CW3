
def connectSpotify():
    #### CONNECTING TO SPOTIFY API ####
    cid ='6eab62e055d94fad926550e22f78bd4a' # Client id (NEED TO CHANGE)
    secret ='18b3117c2ce240e6b2e13b53adba62ea'# Secret key(NEED TO CHANGE)

    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    return sp

def searchForTrack(sp,track):
	track_info = sp.search(track)
	print(track_info)

def getImageOfTrack():
	pass

def getGenreOfTrack():
	pass

def main():
	sp = connectSpotify()
	searchForTrack(sp, 'Free Smoke')

if __name__ == '__main__':
	main()


