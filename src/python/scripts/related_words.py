import os
from collections import defaultdict

import numpy as np
from konlpy import tag

from newstrends import utils
from newstrends.data import mysql

punctuation = ['\'', '"', '.', ',', '!', '?', '~', '…', '”', '“', '/', ';', ':', '(', ')', '▲', '·', '=', '‘', '’', '@',
               '[', ']', '<', '>', '{', '}', "'", "\"", ".", ",", "!", "?", "~", "…", "”", "“", "/", ";", ":", "(", ")",
               "▲", "·", "=", "‘", "’", "@", "[", "]", "<", ">", "{", "}"]
ignore_words = ['나', '너', '그', '그녀', '전', '후', '앞', '뒤', '것', '들', '등', '이', '저', '만', '수', '중', '할', '때', '도',
                '의', '너', '은', '는', '가', '곳', '포토', '일', '며', '고', '말', '해', '다', '하', '번', '기', '추', '코',
                'hspace', 'width', 'px']


def print_dict(dictionary):
    for key in dictionary.keys():
        if len(dictionary[key]) > 0:
            print(f'{[key]}: {dictionary[key]}')


def preprocess(words_list):
    """
    :param words_list: Word lists returned from parser
    :return: Return words lists which have no redundancy and punctuation
    """
    new_words_list = []
    for words_idx in range(len(words_list)):
        new_words_list.append([])
        for word_idx in range(len(words_list[words_idx])):
            word = words_list[words_idx][word_idx]
            word_len = len(word)
            new_word = ''
            for i in range(word_len):
                if word[i].isdigit():
                    new_word = ''
                    break
                if word[i] in punctuation:
                    if len(new_word) > 0:
                        new_words_list[words_idx].append(new_word)
                        new_word = ''
                else:
                    new_word = new_word + word[i]
            if len(new_word) > 0:
                new_words_list[words_idx].append(new_word)

    for words in new_words_list:
        for word in words:
            if word in ignore_words:
                words.remove(word)
        if len(words) is 0:
            new_words_list.remove(words)

    return new_words_list


def similarity_matrix(vectors):
    """
    :param vectors: mxn Vectors containing values of each dimension
    :return: Matrix of size mxm such that each element represents similarity between two vectors
    """
    sim_matrix = np.zeros([vectors.shape[0], vectors.shape[0]])
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            sim_matrix[i][j] = np.inner(vectors[i], vectors[j])
    return sim_matrix


def get_tag_model(name):
    if name == 'kkma':
        model = tag.Kkma()
    elif name == 'hannanum':
        model = tag.Hannanum()
    elif name == 'komoran':
        model = tag.Komoran()
    elif name == 'okt':
        model = tag.Okt()
    else:
        raise ValueError()
    return model


def parse_by_konlpy(articles, package='hannanum'):
    articles = utils.preprocess(articles)
    model = get_tag_model(package)
    words_list = []
    for title in articles:
        words_list.append(model.nouns(title))
    return words_list


'''
def related_word_set(words, sim_matrix, words_frequency, threshold=10):
    """
    USING SIMILARITY MATRIX

    :param words:           List of words(str)
    :param sim_matrix:      sim_matrix[i][j]는 words[i], words[j]가 동시에 등장하는 article 수
    :param words_frequency: 단어 별 등장 frequency
    :param threshold:       Threshold to determine whether two words are related or not

    :return: dictionary of related keywords containing {WORD}:{WORDS RELATED TO GIVEN KEYWORD}
    """
    num_words = len(words)
    flag = list(-np.ones([num_words]))
    idx_dict = dict()
    words_dict = dict()

    for i in range(num_words):
        if flag[i] == -1:
            flag[i] = i
        for j in range(i+1, num_words):
            if sim_matrix[i][j] > threshold and \
                    sim_matrix[i][j] > min(words_frequency[i], words_frequency[j]) * 0.5:
                if flag[j] == -1:
                    if flag[i] in idx_dict.keys():
                        idx_dict[flag[i]].append(j)
                    else:
                        idx_dict[flag[i]] = [j]
                    flag[j] = flag[i]
                else:
                    if flag[i] in idx_dict.keys():
                        idx_dict[flag[j]] = idx_dict[flag[j]] + idx_dict[flag[i]]
                        for k in idx_dict[flag[i]]:
                            flag[k] = flag[j]
                    idx_dict[flag[j]].append(i)
                    flag[i] = flag[j]

    print('done')

    num_count = 0
    for word_idx in idx_dict.keys():
        words_dict[num_count] = [words[word_idx]]
        for i in idx_dict[word_idx]:
            words_dict[num_count].append(words[i])
        num_count += 1

    return words_dict
'''


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

        parsed_titles = preprocess(parse_by_konlpy(titles))
        parsed_contents = preprocess(parse_by_konlpy(contents))
        words_list = [title + content for title, content in zip(parsed_titles, parsed_contents)]
    else:
        titles = mysql.select_all_titles(preprocess=False)
        words_list = preprocess(parse_by_konlpy(titles))

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
