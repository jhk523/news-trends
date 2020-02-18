import json
import os

import pandas as pd
from sqlalchemy import create_engine

_ROOT_DIR = os.path.abspath(__file__ + "/../../../../")
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        db_info_path = os.path.join(_ROOT_DIR, 'data/db_info.json')
        db_info = json.loads(open(db_info_path).read())

        user = db_info['USER']
        password = db_info['PASSWORD']
        address = db_info['ADDRESS']
        port = db_info['PORT']
        db = db_info['DB']

        _URL = f'mysql+pymysql://{user}:{password}@{address}:{port}/{db}?charset=utf8mb4'
        _ENGINE = create_engine(_URL, echo=False, encoding='utf-8', pool_recycle=3600)
    return _ENGINE


def create_news_table():
    query = "create table if not exists news(" \
            "`date` DATETIME not null, " \
            "publisher VARCHAR(255), " \
            "title VARCHAR(255), " \
            "author VARCHAR(255), " \
            "link VARCHAR(255) UNIQUE, " \
            "description TEXT)"
    get_engine().execute(query)


def select_all_articles():
    sql = 'select title from news'
    # sql = 'select title, description from news'
    entries = get_engine().execute(sql)
    articles = []
    for entry in entries:
        articles.append(entry[0])
        # articles.append(entry[1])
    return articles


def select_articles(field=None, publishers=None):
    if field is None:
        field = '*'
    elif isinstance(field, list):
        field = ', '.join(field)

    sql = f'select {field} from news'
    if publishers is not None:
        pub_str = ', '.join(f'\'{e}\'' for e in publishers)
        sql += f' where publisher in ({pub_str})'

    entries = get_engine().execute(sql)
    if isinstance(field, str):
        return [e[0] for e in entries]
    return [e for e in entries]


def read_publisher_links(publisher):
    query = "select link from news where publisher='{}'"
    fetched = get_engine().execute(query.format(publisher)).fetchall()
    df = pd.DataFrame(fetched, columns=['link'])
    return df.assign(dup=1)
