from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView
from search.models import NewsArticle
from newstrends.utils import search_keyword_sentiment


class Search(TemplateView):
    template_name = 'search/index.html'

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)

        news_articles = NewsArticle.objects.first()

        context['news_articles'] = news_articles

        return context


class Result(TemplateView):
    template_name = 'search/result.html'

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)

        df = search_keyword_sentiment('코로나')
        df = df.iloc[1:3, :]
        df_list = df.to_dict('records')

        context['df'] = df_list[0]

        return context



