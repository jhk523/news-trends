import requests
import dateparser
import re
import pandas as pd
from bs4 import BeautifulSoup as bs

from newstrends.data.db import mysql

URLS = {'조선일보': 'http://www.chosun.com/site/data/rss/rss.xml',
        '동아일보': 'https://rss.donga.com/total.xml',
        '중앙일보': 'https://rss.joins.com/joins_news_list.xml',
        '노컷뉴스': 'http://rss.nocutnews.co.kr/nocutnews.xml',
        '경향신문': 'http://www.khan.co.kr/rss/rssdata/total_news.xml',
        '한겨례': 'http://www.hani.co.kr/rss/',
        '국민일보': 'http://rss.kmib.co.kr/data/kmibRssAll.xml',
        '데일리경제': 'http://www.kdpress.co.kr/rss/allArticle.xml',
        '매일경제': 'https://www.mk.co.kr/rss/40300001/',
        '머니투데이': 'https://rss.mt.co.kr/mt_news.xml',
        '세계일보': 'http://www.segye.com/Articles/RSSList/segye_recent.xml',
        '오마이뉴스': 'http://rss.ohmynews.com/rss/ohmynews.xml',
        '전자신문': 'http://rss.etnews.com/Section901.xml',
        '한국경제': 'http://rss.hankyung.com/new/news_main.xml',
        'SBS정치': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER',
        'SBS경제': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=02&plink=RSSREADER',
        'SBS사회': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=03&plink=RSSREADER',
        'SBS생활/문화': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=07&plink=RSSREADER',
        'SBS국제/글로벌': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=08&plink=RSSREADER',
        'SBS연예/방송': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=14&plink=RSSREADER',
        'SBS스포츠': 'https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=09&plink=RSSREADER',
        'JTBC정치': 'http://fs.jtbc.joins.com/RSS/politics.xml',
        'JTBC경제': 'http://fs.jtbc.joins.com/RSS/economy.xml',
        'JTBC사회': 'http://fs.jtbc.joins.com/RSS/society.xml',
        'JTBC국제': 'http://fs.jtbc.joins.com/RSS/international.xml',
        'JTBC문화': 'http://fs.jtbc.joins.com/RSS/culture.xml',
        'JTBC연예': 'http://fs.jtbc.joins.com/RSS/entertainment.xml',
        'JTBC스포츠': 'http://fs.jtbc.joins.com/RSS/sports.xml'
        }


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
    if tag is 'date':
        try:
            result = item.find('dc:date').text
        except AttributeError:
            result = item.find('pubDate').text

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
        markup = get_html(URLS['JTBC정치'])
        soup = bs(markup, 'lxml-xml')
        news_item = soup.find_all('item')
        print(markup)
        print(soup)
        return
    if initialize:
        mysql.create_news_table()

    for publisher in URLS.keys():
        if verbose:
            print(publisher)
        markup = get_html(URLS[publisher])

        news_df = _create_news_dataframe(init=True)
        soup = bs(markup, 'lxml-xml')
        news_item = soup.find_all('item')
        news_df = _create_news_dataframe(df=news_df, items=news_item,
                                         pub=publisher)
        if not initialize:
            news_df = _remove_duplicates(news_df, publisher)

        news_df.drop_duplicates(inplace=True, keep=False, subset=['link'])
        news_df.to_sql('news', mysql.ENGINE, if_exists='append', index=False)
