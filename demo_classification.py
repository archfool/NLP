# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:52:43 2019

@author: ruan
"""

import numpy as np
from pandas import DataFrame as dataframe, Series as series
import os
import time
import jieba
import re
import multiprocessing
import logging

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
import nn_lib

path_nlp = r'E:\\MachineLearning\\data\\nlp\\'
word_dim = 50
max_seq_num = 250
dim_lstm = 64
batch_size = 1024
learning_rate = 1e-5


def load_data(path_data):
    # 读取词向量
    wordVectors = np.load(path_data + 'wordVectors.npy')

    # 尝试读取处理好的语料
    try:
        corpus = np.load(path_data + 'idsMatrix.npy')
        x = corpus
        y = np.concatenate((np.ones((12500, 1)), np.zeros((12500, 1))), axis=1)
        y = np.concatenate((y, 1 - y), axis=0)
        return x, y, wordVectors
    except:
        pass

    # 在读取语料文件失败后，重新生成语料
    wordsList = np.load(path_data + 'wordsList.npy')
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]

    def cleanSentences(string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())

    pos_file_num = 0
    for file_name in os.listdir(path_data + 'pos'):
        pos_file_num += 1
    neg_file_num = 0
    for file_name in os.listdir(path_data + 'neg'):
        neg_file_num += 1

    corpus = np.zeros((pos_file_num + neg_file_num, max_seq_num), dtype='int32')
    file_count = 0

    def read_file(path_file, corpus, file_count):
        for file_name in os.listdir(path_file):
            if file_count % 100 == 0:
                logging.warning(file_count)
            with open(path_file + file_name, "r", encoding='utf-8') as f:
                print(path_file + file_name)
                indexCounter = 0
                line = f.readline()
                cleanedLine = cleanSentences(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        corpus[file_count][indexCounter] = wordsList.index(word)
                    except ValueError:
                        corpus[file_count][indexCounter] = 399999  # 未知的词
                    indexCounter = indexCounter + 1
                    if indexCounter >= max_seq_num:
                        break
                file_count = file_count + 1
        return corpus, file_count

    corpus, file_count = read_file(path_data + r'pos\\', corpus, file_count)
    corpus, file_count = read_file(path_data + r'neg\\', corpus, file_count)

    x = corpus
    y = np.concatenate((np.ones((pos_file_num, 1)), np.zeros((pos_file_num, 1))), axis=1)
    y = np.concatenate((y, 1 - y), axis=0)

    return x, y, wordVectors


if not 'x' in locals():
    x, y, w2v = load_data(path_nlp)

random_seed = int(time.time())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=random_seed)
model = neural_network(x_train, y_train, task_type='classification', \
                       model_type='rnn_nlp', loss_fun_type='cross_entropy', \
                       eval_score_type='accuracy', optimizer_type='Adam', \
                       model_parameter={'word2vec': w2v, \
                                        'keep_prob': 0.7, \
                                        'dim_lstm': dim_lstm}, \
                       hyper_parameter={'learning_rate': learning_rate, \
                                        'batch_size': batch_size, \
                                        'early_stop': None}, \
                       path_data=path_nlp)
model.train(transfer_learning=False, built_in_test=True, x_test=x_test, y_test=y_test)
# out=model.predict(x_test)


