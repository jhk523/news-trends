from newstrends import azure
from newstrends.data import mysql


def main():
    titles = mysql.select_all_titles(preprocess=True)
    titles = titles[:1000]
    scores = azure.compute_scores(titles)
    azure.print_scores(titles, scores)


if __name__ == '__main__':
    main()
