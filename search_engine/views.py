from django.shortcuts import render
from .models import Song

def home(request):
    return render(request, "home.html")

def display_song(request):
    songs = Song.objects.all()
    # print(songs)
    return render(request, "home.html", {'songs':songs})