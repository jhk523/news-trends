from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView
from django.views.generic.edit import FormView

from search.models import NewsArticle
from search.forms import SearchValue

SEARCH_WORD = ""


class Search(TemplateView):
    template_name = 'search/index.html'

    def get_context_data(self, *args, **kwargs):
        from newstrends.utils import find_popular_keywords

        context = super().get_context_data(*args, **kwargs)

        df = find_popular_keywords()
        dates_list = df.date.unique()

        date1_df = df[df['date'] == dates_list[0]]
        date2_df = df[df['date'] == dates_list[1]]

        date1_list = date1_df.to_dict('records')
        date2_list = date2_df.to_dict('records')

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

        search_word_len = len(SEARCH_WORD.split(' '))

        if search_word_len > 2:
            from newstrends.utils import compute_sentence_polarity
            polarity = compute_sentence_polarity(SEARCH_WORD)

            context['keyword'] = SEARCH_WORD
            context['type'] = 'polarity'
            context['left'] = polarity['진보']
            context['right'] = polarity['보수']

            return context
        else:
            from newstrends.utils import search_keyword_sentiment
            df = search_keyword_sentiment(SEARCH_WORD)
            df.sort_values(by='date', ascending=False, inplace=True)
            df['pos_score'] = df['pos_score'].apply(lambda x: '{:.2f}'.format(x))
            df['neg_score'] = df['neg_score'].apply(lambda x: '{:.2f}'.format(x))
            df['neu_score'] = df['neu_score'].apply(lambda x: '{:.2f}'.format(x))

            df_list = df.to_dict('records')
            publishers_list = df.publisher.unique()

            context['keyword'] = SEARCH_WORD
            context['type'] = 'sentiment'
            context['df'] = df_list
            context['publishers'] = publishers_list

            return context
