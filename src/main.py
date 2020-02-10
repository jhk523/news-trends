import click


@click.command()
@click.option('--initialize', type=bool, default=False, is_flag=True)
@click.option('--verbose', type=bool, default=False, is_flag=True)
@click.option('--test', type=bool, default=False, is_flag=True)
def scrape_news(initialize=False, verbose=False, test=False):
    from newstrends.data.scraping.scrape_news import update_news

    update_news(initialize, verbose, test)


@click.group()
def main():
    pass


main.add_command(scrape_news)

if __name__ == '__main__':
    main()
