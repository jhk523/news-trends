from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView
from django.views.generic.edit import FormView

from search.models import NewsArticle
from search.forms import SearchValue
from newstrends.utils import search_keyword_sentiment

SEARCH_WORD = ""


class Search(TemplateView):
    template_name = 'search/index.html'

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)

        news_articles = NewsArticle.objects.first()

        context['news_articles'] = news_articles

        return context


class Result(FormView):
    template_name = 'search/result.html'
    form_class = SearchValue
    success_url = '/result/'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def form_valid(self, form):
        global SEARCH_WORD

        SEARCH_WORD = form.cleaned_data['search_value']
        return super(Result, self).form_valid(form)

    def get_context_data(self, *args, **kwargs):
        context = super(Result, self).get_context_data(*args, **kwargs)

        df = search_keyword_sentiment(SEARCH_WORD)
        df = df.iloc[1:3, :]
        df_list = df.to_dict('records')

        context['df'] = df_list[0]

        return context
