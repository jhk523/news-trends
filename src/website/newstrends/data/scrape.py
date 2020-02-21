import datetime
import re

import dateparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import mysql

URLS = {'조선일보': ['http://www.chosun.com/site/data/rss/rss.xml'],
        '동아일보': ['https://rss.donga.com/total.xml'],
        '중앙일보': ['https://rss.joins.com/joins_news_list.xml'],
        '노컷뉴스': ['http://rss.nocutnews.co.kr/nocutnews.xml'],
        '경향신문': ['http://www.khan.co.kr/rss/rssdata/total_news.xml'],
        '한겨례': ['http://www.hani.co.kr/rss/'],
        '국민일보': ['http://rss.kmib.co.kr/data/kmibRssAll.xml'],
        '데일리경제': ['http://www.kdpress.co.kr/rss/allArticle.xml'],
        '매일경제': ['https://www.mk.co.kr/rss/40300001/'],
        '머니투데이': ['https://rss.mt.co.kr/mt_news.xml'],
        '세계일보': ['http://www.segye.com/Articles/RSSList/segye_recent.xml'],
        '오마이뉴스': ['http://rss.ohmynews.com/rss/ohmynews.xml'],
        '한국경제': ['http://rss.hankyung.com/new/news_main.xml'],
        '프레시안': ['http://www.pressian.com/data/rss/news.xml'],
        '시사인': ['https://www.sisain.co.kr/rss/allArticle.xml'],
        '뉴시스': ['http://www.newsis.com/RSS/sokbo.xml',
                'http://www.newsis.com/RSS/country.xml',
                'http://www.newsis.com/RSS/politics.xml',
                'http://www.newsis.com/RSS/square.xml',
                'http://www.newsis.com/RSS/economy.xml',
                'http://www.newsis.com/RSS/industry.xml',
                'http://www.newsis.com/RSS/society.xml',
                'http://www.newsis.com/RSS/international.xml',
                'http://www.newsis.com/RSS/culture.xml'],
        '아주경제': ['https://rss.ajunews.com/sokbo.xml',
                 'https://rss.ajunews.com/china.xml',
                 'https://rss.ajunews.com/politics.xml',
                 'https://rss.ajunews.com/economy.xml',
                 'https://rss.ajunews.com/industry.xml',
                 'https://rss.ajunews.com/society.xml',
                 'https://rss.ajunews.com/global.xml',
                 'https://rss.ajunews.com/itsience.xml',
                 'https://rss.ajunews.com/opinion.xml'],
        '전자신문': ['http://rss.etnews.com/Section901.xml',
                 'http://rss.etnews.com/Section902.xml',
                 'http://rss.etnews.com/02.xml',
                 'http://rss.etnews.com/20.xml'],
        'SBS': ['https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER',
                'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=02&plink=RSSREADER',
                'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=03&plink=RSSREADER',
                'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=07&plink=RSSREADER',
                'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=08&plink=RSSREADER',
                'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=14&plink=RSSREADER',
                'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=09&plink=RSSREADER'],
        'JTBC': ['http://fs.jtbc.joins.com/RSS/politics.xml',
                 'http://fs.jtbc.joins.com/RSS/economy.xml',
                 'http://fs.jtbc.joins.com/RSS/society.xml',
                 'http://fs.jtbc.joins.com/RSS/international.xml',
                 'http://fs.jtbc.joins.com/RSS/culture.xml',
                 'http://fs.jtbc.joins.com/RSS/entertainment.xml',
                 'http://fs.jtbc.joins.com/RSS/sports.xml'],
        '매일노동뉴스': ['http://www.labortoday.co.kr/rss/allArticle.xml']
        }
HEADERS = {
    'X-Requested-With': 'XMLHttpRequest',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
}


def print_status(*args):
    """
    Print a message with the current time.

    :param args: the list of arguments; a format string and elements.
    :return None
    """
    import datetime
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = args[0].format(*args[1:])
    print('[{}] {}'.format(time, message))


def get_html(publisher, url):
    _html = ""
    try:
        resp = requests.get(url, headers=HEADERS)
    except requests.ConnectionError:
        print_status(publisher, 'Connection Error!')
        return False
    except requests.exceptions.ChunkedEncodingError:
        print_status(publisher, 'Chunked Encoding Error!')
        return False
    resp.encoding = 'UTF-8'
    if resp.status_code == 200:
        _html = resp.text
    return _html


def change_datetime(date_info):
    result = dateparser.parse(date_info)
    return result.strftime('%Y-%m-%d %H:%M:%S')


def find_tag(item, tag):
    if tag is 'date':
        try:
            result = item.find('dc:date').text
        except AttributeError:
            try:
                result = item.find('pubDate').text
            except AttributeError:
                time = datetime.datetime.now()
                return time.strftime('%Y-%m-%d %H:%M:%S')

        return change_datetime(result)
    else:
        try:
            result = item.find(tag).text
        except AttributeError:
            return None

        if tag is 'description':
            result = re.sub('<table.*?>.*?</table>', "", result, 0, re.I | re.S)
        return result.strip()


def _create_news_dataframe(init=False, df=None, items=None, pub=None):
    if init:
        columns = ['date', 'title', 'author', 'link', 'description']
        return pd.DataFrame(columns=columns)
    else:
        for item in items:
            news_date = find_tag(item, 'date')
            news_title = find_tag(item, 'title')
            news_author = find_tag(item, 'author')
            news_link = find_tag(item, 'link')
            news_description = find_tag(item, 'description')

            temp_df = pd.DataFrame({"date": [news_date],
                                    "publisher": [pub],
                                    "title": [news_title],
                                    "author": [news_author],
                                    "link": [news_link],
                                    "description": [news_description]})

            df = df.append(temp_df, ignore_index=True)
        return df


def _remove_duplicates(df, publisher):
    db_links = mysql.read_publisher_links(publisher)
    merge_df = pd.merge(df, db_links, how='left', on=['link'])
    merge_df = merge_df[merge_df['dup'] != 1]
    return merge_df.drop(columns=['dup'])


def update_news(initialize, verbose, test):
    if test:
        markup = get_html('JTBC', URLS['JTBC정치'])
        soup = BeautifulSoup(markup, 'lxml-xml')
        news_item = soup.find_all('item')
        print(markup)
        print(soup)
        return
    if initialize:
        mysql.create_news_table()

    for publisher in URLS.keys():
        if verbose:
            print(publisher)

        news_df = _create_news_dataframe(init=True)

        for sub_section in URLS[publisher]:
            markup = get_html(publisher, sub_section)
            if not markup:
                continue

            soup = BeautifulSoup(markup, 'lxml-xml')
            news_item = soup.find_all('item')
            news_df = _create_news_dataframe(df=news_df, items=news_item,
                                             pub=publisher)
            if not initialize:
                news_df = _remove_duplicates(news_df, publisher)

        news_df.drop_duplicates(inplace=True, keep=False, subset=['link'])
        news_df.to_sql('news', mysql.get_engine(), if_exists='append', index=False)
