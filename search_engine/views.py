from django.shortcuts import render
from .models import Song

def display_home(request):
    return render(request, "home.html")

def display_search(request, lyric):
    songs = Song.objects.all()
    # results = somefunc(lyrics)
    print(lyric)
    results = [{"photoid":"1", "name":"goerges song", "artist":"ryan kilgour", "genre":"country", "match":"low"}]

    return render(request, "results.html", {'results':results})

def display_song(request, songid):
    songs = Song.objects.all()
    # results = somefunc(lyrics)
    song = {"title" : "goerges song", "weekdiff":"+3%", "searches":3000}
    return render(request, "song-info.html", {"song":song})