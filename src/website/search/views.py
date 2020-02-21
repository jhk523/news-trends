from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView
from django.views.generic.edit import FormView

from search.models import NewsArticle
from search.forms import SearchValue
from newstrends.utils import search_keyword_sentiment, find_popular_keywords

SEARCH_WORD = ""


class Search(TemplateView):
    template_name = 'search/index.html'

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)

        df = find_popular_keywords()
        dates_list = df.date.unique()

        date1_df = df[df['date'] == dates_list[0]]
        date2_df = df[df['date'] == dates_list[1]]

        date1_list = date1_df.to_dict('records')
        date2_list = date2_df.to_dict('records')

        print(date1_df)
        print(date2_df)

        context['date1'] = dates_list[0]
        context['date2'] = dates_list[1]
        context['date1_list'] = date1_list
        context['date2_list'] = date2_list

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
        df_list = df.to_dict('records')
        publishers_list = df.publisher.unique()

        context['df'] = df_list
        context['publishers'] = publishers_list

        return context

# class Result(TemplateView):
#     template_name = 'search/result.html'
#
#     def get_context_data(self, *args, **kwargs):
#         context = super().get_context_data(*args, **kwargs)
#
#         df = search_keyword_sentiment('추미애')
#         df_list = df.to_dict('records')
#         publishers_list = df.publisher.unique()
#
#         context['df'] = df_list
#         context['publishers'] = publishers_list
#
#         print(df.iloc[0, :])
#
#         return context
