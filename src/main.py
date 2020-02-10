import click


@click.command()
@click.option('--initialize', type=bool, default=False, is_flag=True)
@click.option('--verbose', type=bool, default=False, is_flag=True)
def scrape_news(initialize=False, verbose=False):
    from newstrends.data.scraping.scrape_news import update_news

    update_news(initialize, verbose)


@click.group()
def main():
    scrape_news()


main.add_command(scrape_news)

if __name__ == '__main__':
    main()
