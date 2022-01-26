from django.db import models

class Song(models.Model):
    Title = models.CharField(max_length=100)
    Artist = models.CharField(max_length=100)
    Lyrics = models.TextField()

    def __str__(self):
        return self.Title
#   ...

class Artist(models.Model):
    Name = models.CharField(max_length=100)