import requests
import dateparser
import re
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
    return result.strftime('%Y-%m-%d %H:%M')


def find_tag(item, tag):
    try:
        result = item.find(tag).text
    except AttributeError:
        return None

    if tag is 'dc:date':
        return change_datetime(result)
    elif tag is 'description':
        result = re.sub('<table.*?>.*?</table>', "", result, 0, re.I | re.S)
    return result.strip()


def _create_news_table():
    query = "create table if not exists news(" \
            "`date` date not null, " \
            "title VARCHAR(255), " \
            "link VARCHAR(255), " \
            "author VARCHAR(20), " \
            "description TEXT)"
    mysql.ENGINE.execute(query)


def update_news(initialize, verbose):
    if initialize:
        _create_news_table()

    # markup = get_html(URLS['조선일보'])
    # soup = bs(markup, 'lxml-xml')
    #
    # news_item = soup.find('item')
    # news_title = find_tag(news_item, 'title')
    # news_description = find_tag(news_item, 'description')
    # news_author = find_tag(news_item, 'author')
    # news_link = find_tag(news_item, 'link')
    # news_date = find_tag(news_item, 'dc:date')
    # news_date = change_datetime(news_date)
    #
    # print(news_item)
    # print()
    # print()
    # print(news_title)
    # print(news_link)
    # print(news_description)
    # print(news_date)
    # print(news_author)
