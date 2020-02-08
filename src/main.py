import requests
import dateparser
import re
# import click


def scrape_news(initialize=True, verbose=False):
    from newstrends.data.scraping.scrape_news import update_news

    update_news(initialize, verbose)


# @click.group()
def main():
    scrape_news()


if __name__ == '__main__':
    main()
