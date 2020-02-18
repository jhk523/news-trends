import time

import click

TIMEOUT = 60.0


@click.command()
@click.option('--initialize', type=bool, default=False, is_flag=True)
@click.option('--verbose', type=bool, default=False, is_flag=True)
@click.option('--test', type=bool, default=False, is_flag=True)
def scrape_news(initialize=False, verbose=False, test=False):
    from newstrends.data.scrape import update_news

    while True:
        update_news(initialize, verbose, test)

        if initialize:
            break

        time.sleep(TIMEOUT)


@click.group()
def main():
    pass


main.add_command(scrape_news)

if __name__ == '__main__':
    main()
