import os
import json
import pandas as pd
from sqlalchemy import create_engine

ROOT_DIR = os.path.abspath(__file__ + "/../../../../")
DB_INFO_PATH = os.path.join(ROOT_DIR, 'db_info.json')
DB_INFO = json.loads(open(DB_INFO_PATH).read())

_USER = DB_INFO['USER']
_PASSWORD = DB_INFO['PASSWORD']
_ADDRESS = DB_INFO['ADDRESS']
_PORT = DB_INFO['PORT']
_DB = DB_INFO['DB']

_URL = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8mb4'.format(
    _USER, _PASSWORD, _ADDRESS, _PORT, _DB)
ENGINE = create_engine(_URL, echo=False, encoding='utf-8', pool_recycle=3600)


def create_news_table():
    query = "create table if not exists news(" \
            "`date` DATETIME not null, " \
            "publisher VARCHAR(255), " \
            "title VARCHAR(255), " \
            "author VARCHAR(255), " \
            "link VARCHAR(255) UNIQUE, " \
            "description TEXT)"
    ENGINE.execute(query)


def read_publisher_links(publisher):
    query = "select link from news where publisher='{}'"
    fetched = ENGINE.execute(query.format(publisher)).fetchall()
    df = pd.DataFrame(fetched, columns=['link'])
    return df.assign(dup=1)
