from django import forms
from django.db import models


class SearchValue(forms.Form):
    search_value = forms.CharField(label='search_value', max_length=100)

