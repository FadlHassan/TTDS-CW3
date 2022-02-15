from django.shortcuts import render
from .models import Song

def home(request):
    return render(request, "home.html")

def display_search(request, lyric):
    songs = Song.objects.all()
    # results = somefunc(lyrics)
    print(lyric)
    results = [{"photoid":"1", "name":"goerges song", "artist":"ryan kilgour", "genre":"country", "match":"low"}]

    return render(request, "results.html", {'results':results})