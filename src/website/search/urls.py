from django.urls import path

from . import views
from search.views import Search

app_name = 'status'
urlpatterns = [
    path('', Search.as_view(), name='index'),
]
