import requests
import dateparser
import re
import pandas as pd
from bs4 import BeautifulSoup as bs

from newstrends.data.db import mysql

URLS = {'조선일보': 'http://www.chosun.com/site/data/rss/rss.xml',
        '동아일보': 'https://rss.donga.com/total.xml',
        '노컷뉴스': 'http://rss.nocutnews.co.kr/nocutnews.xml',
        '경향신문': 'http://www.khan.co.kr/rss/rssdata/total_news.xml'}


def get_html(url):
    _html = ""
    resp = requests.get(url)
    resp.encoding = 'UTF-8'
    if resp.status_code == 200:
        _html = resp.text
    return _html


def change_datetime(date_info):
    result = dateparser.parse(date_info)
    return result.strftime('%Y-%m-%d %H:%M:%S')


def find_tag(item, tag):
    try:
        result = item.find(tag).text
    except AttributeError:
        return None

    if 'date' in tag:
        return change_datetime(result)
    elif tag is 'description':
        result = re.sub('<table.*?>.*?</table>', "", result, 0, re.I | re.S)
    return result.strip()


def _create_news_table():
    query = "create table if not exists news(" \
            "`date` DATETIME not null, " \
            "title VARCHAR(255), " \
            "author VARCHAR(255), " \
            "link VARCHAR(255), " \
            "description TEXT)"
    mysql.ENGINE.execute(query)


def _create_news_dataframe(init=False, df=None, item=None):
    if init:
        columns = ['date', 'title', 'author', 'link', 'description']
        return pd.DataFrame(columns=columns)
    else:
        news_date = find_tag(item, 'dc:date')
        news_title = find_tag(item, 'title')
        news_author = find_tag(item, 'author')
        news_link = find_tag(item, 'link')
        news_description = find_tag(item, 'description')

        temp_df = pd.DataFrame({"date": [news_date],
                                "title": [news_title],
                                "author": [news_author],
                                "link": [news_link],
                                "description": [news_description]})

        return df.append(temp_df, ignore_index=True)


def update_news(initialize, verbose):
    if initialize:
        _create_news_table()

    news_df = _create_news_dataframe(init=True)

    markup = get_html(URLS['조선일보'])
    soup = bs(markup, 'lxml-xml')

    news_item = soup.find('item')

    news_df = _create_news_dataframe(df=news_df, item=news_item)
    news_df.to_sql('news', mysql.ENGINE, if_exists='append', index=False)
