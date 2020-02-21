from django.urls import path
from django.conf.urls import url

from . import views
from search.views import Search, Result

app_name = 'status'
urlpatterns = [
    url(r'^$', Search.as_view(), name='index'),
    url(r'^result/$', Result.as_view(), name='result')
    # path('', Search.as_view(), name='index'),
    # path('result/', Result.as_view(), name='result')
]
