from django.urls import path

from . import views
from search.views import Search, Result

app_name = 'status'
urlpatterns = [
    path('', Search.as_view(), name='index'),
    path('result/', Result.as_view(), name='result')
]
