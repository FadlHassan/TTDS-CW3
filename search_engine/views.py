from django.shortcuts import render
from .models import Song
import urllib.parse
import main_code

def display_home(request):
    return render(request, "home.html")

def display_search(request, encodedLyric):
    lyrics = urllib.parse.unquote(encodedLyric)
    print(lyrics)
    # results = main_code.searchLyrics(lyrics, 10)

    # print(results)
    results = [{"photoid":"1", "name":"goerges song", "artist":"ryan kilgour", "genre":"country", "match":"low"}]

    return render(request, "results.html", {'results':results})

def display_song(request, title):
    # results = somefunc(lyrics)
    result = main_code.Query(f"SELECT * FROM tracks WHERE title=='{title}'")
    print(result)

    song = {"title" : "goerges song", "weekdiff":"+3%", "searches":3000}
    return render(request, "song-info.html", {"song":song})