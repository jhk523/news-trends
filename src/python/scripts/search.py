from newstrends import azure
from newstrends.data import mysql


def main():
    titles = mysql.select_all_titles(preprocess=True)
    keyword = '추미애'
    titles_searched = [t for t in titles if t.find(keyword) >= 0]
    scores = azure.compute_scores(titles_searched)
    azure.print_scores(titles_searched, scores)


if __name__ == '__main__':
    main()
