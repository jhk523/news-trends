import requests
import dateparser
import re
import datetime
from pytz import timezone
from bs4 import BeautifulSoup as bs


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

    result = re.sub('<table.*?>.*?</table>', "", result, 0, re.I | re.S)
    return result.strip()


URLS = ['http://www.chosun.com/site/data/rss/rss.xml',  # 조선일보
        'http://rss.nocutnews.co.kr/nocutnews.xml',  # 노컷뉴스
        'http://www.khan.co.kr/rss/rssdata/total_news.xml'  # 경향신문
        ]

markup = get_html(URLS[0])
soup = bs(markup, 'html.parser')

news = soup.find('title').text

news_item = soup.find('item')
news_title = find_tag(news_item, 'title')
news_description = find_tag(news_item, 'description')
news_author = find_tag(news_item, 'author')
news_date = find_tag(news_item, 'dc:date')
news_date = change_datetime(news_date)

news_link = news_item.find('link')

print(news_item)
print()
print()
print(news)
print(news_title)
print(news_description)
print(news_author)
print(news_link)
print(news_date)
