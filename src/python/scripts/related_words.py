import os
from collections import defaultdict

import numpy as np
from konlpy import tag

from newstrends import utils
from newstrends.data import mysql

punctuation = ['\'', '"', '.', ',', '!', '?', '~', '…', '”', '“', '/', ';', ':', '(', ')', '▲', '·', '=', '‘', '’', '@',
               '[', ']', '<', '>', '{', '}', "'", "\"", ".", ",", "!", "?", "~", "…", "”", "“", "/", ";", ":", "(", ")",
               "▲", "·", "=", "‘", "’", "@", "[", "]", "<", ">", "{", "}"]
ignore_words = {'나', '너', '그', '그녀', '전', '후', '앞', '뒤', '것', '들', '등', '이', '저', '만', '수', '중', '할', '때', '도',
                '의', '너', '은', '는', '가', '곳', '포토', '일', '며', '고', '말', '해', '다', '하', '번', '기', '추', '코',
                'hspace', 'width', 'px'}


def print_dict(dictionary):
    for key in dictionary.keys():
        if len(dictionary[key]) > 0:
            print(f'{[key]}: {dictionary[key]}')


def parse_into_nouns(articles):
    articles = utils.preprocess(articles)
    model = tag.Hannanum()
    result = []
    for sentence in articles:
        words = []
        for word in model.nouns(sentence):
            for p in punctuation:
                word = word.replace(p, '')
            if word not in ignore_words:
                try:
                    int(word)
                except ValueError:
                    words.append(word)
        if len(words) > 0:
            result.append(words)
    return result


def related_words_with_given_keyword(words, sim_matrix):
    """
    USING SIMILARITY MATRIX

    :param words:           List of words(str)
    :param sim_matrix:      sim_matrix[i][j]는 words[i], words[j]가 동시에 등장하는 article 수

    :return: dictionary of related keywords containing {WORD}:{WORDS RELATED TO GIVEN KEYWORD}
    """
    num_words = len(words)
    words_dict = dict()
    sim_dict = dict()

    for i in range(num_words):
        for j in range(num_words):
            if i != j and sim_matrix[i][j] > 10:
                if words[i] in words_dict.keys():
                    num_count = len(words_dict[words[i]])
                    if num_count >= 10 and sim_dict[words[i]][-1] > sim_matrix[i][j]:
                        continue
                    elif num_count >= 10:
                        words_dict[words[i]][-1] = words[j]
                        sim_dict[words[i]][-1] = sim_matrix[i][j]
                        num_count = 9
                        while num_count > 0 and \
                                sim_dict[words[i]][num_count] > sim_dict[words[i]][num_count - 1]:
                            temp = sim_dict[words[i]][num_count]
                            sim_dict[words[i]][num_count] = sim_dict[words[i]][num_count - 1]
                            sim_dict[words[i]][num_count - 1] = temp
                            temp = words_dict[words[i]][num_count]
                            words_dict[words[i]][num_count] = words_dict[words[i]][num_count - 1]
                            words_dict[words[i]][num_count - 1] = temp
                            num_count -= 1
                    else:
                        num_count = len(words_dict[words[i]])
                        words_dict[words[i]].append(words[j])
                        sim_dict[words[i]].append(sim_matrix[i][j])
                        while num_count > 0 and \
                                sim_dict[words[i]][num_count] > sim_dict[words[i]][num_count - 1]:
                            temp = sim_dict[words[i]][num_count]
                            sim_dict[words[i]][num_count] = sim_dict[words[i]][num_count - 1]
                            sim_dict[words[i]][num_count - 1] = temp
                            temp = words_dict[words[i]][num_count]
                            words_dict[words[i]][num_count] = words_dict[words[i]][num_count - 1]
                            words_dict[words[i]][num_count - 1] = temp
                            num_count -= 1
                else:
                    words_dict[words[i]] = [words[j]]
                    sim_dict[words[i]] = [sim_matrix[i][j]]
    return words_dict


def run(use_contents=False, min_counts=10):
    if use_contents:
        entries = mysql.select_articles(field=['title', 'description'])
        titles = [e[0] for e in entries]
        contents = [e[1] for e in entries]

        parsed_titles = parse_into_nouns(titles)
        parsed_contents = parse_into_nouns(contents)
        words_list = [title + content for title, content in zip(parsed_titles, parsed_contents)]
    else:
        titles = mysql.select_all_titles(preprocess=False)
        words_list = parse_into_nouns(titles)

    word_list = [w for words in words_list for w in words]
    word_counts = defaultdict(lambda: 0)
    for w in word_list:
        word_counts[w] += 1
    words = [w for w in set(word_list) if word_counts[w] >= min_counts]
    words = np.array(words, dtype=str)

    num_contents = len(words_list)
    words_in_articles = np.zeros([words.shape[0], num_contents])
    words_frequency = np.zeros([words.shape[0]])
    sim_matrix = np.zeros([words.shape[0], words.shape[0]])
    for w_idx in range(words_in_articles.shape[0]):
        for i in range(len(words_list)):
            if words[w_idx] in words_list[i]:
                words_in_articles[w_idx][i] = 1
                words_frequency[w_idx] += 1

    for w1 in range(sim_matrix.shape[0]):
        for w2 in range(sim_matrix.shape[1]):
            sim_matrix[w1][w2] = np.inner(words_in_articles[w1], words_in_articles[w2])

    related_words_dict_2 = related_words_with_given_keyword(words, sim_matrix)

    print_dict(related_words_dict_2)

    out_path = '../../../out'
    os.makedirs(out_path, exist_ok=True)
    np.savetxt(os.path.join(out_path, 'words.tsv'), words, fmt='%s', delimiter='\t')
    np.savetxt(os.path.join(out_path, 'similarity.tsv'), sim_matrix, delimiter='\t')

    ############################################################################


def main():
    run()


if __name__ == '__main__':
    main()
