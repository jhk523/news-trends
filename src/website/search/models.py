from django.db import models


# Create your models here.
class NewsArticle(models.Model):
    date = models.DateTimeField(blank=True, null=True)
    publisher = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255, blank=True, null=True)
    link = models.CharField(max_length=255, blank=True, null=True, unique=True)
    description = models.TextField(blank=True, null=True)
