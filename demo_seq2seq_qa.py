# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:31:32 2019

@author: ruan
"""

import codecs
import collections
from operator import itemgetter
import numpy as np
from pandas import DataFrame as dataframe
from pandas import Series as series
import os
import time
import tensorflow as tf
import logging
logging.basicConfig(level=logging.WARNING,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
import nn_lib

path_data = u'.\\data\\'
path_seq2seq = path_data+u'seq2seq_qa\\'
corpus_list = ['qingyun.tsv', 'chatterbot.tsv', 'ptt.tsv']
# 源序列词库相关参数
vocab_size = 10000
path_vocab = path_seq2seq+'vocab.pkl'
path_train_done_src = path_seq2seq+'train.en'

# 超参数
word_embd_dim = 100
dim_rnn = word_embd_dim
learning_rate = 0.001
batch_size = 128
# 其它参数
test_mode = False

# 读取样本数据
def load_file(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_q = []
    data_a = []
    for line in lines:
        count += 1
        if (test_mode is True) and (count == 1000):
            break
        line_ = line.strip().split('\t')
        if (len(line_))>=2:
            data_q.append(line_[0])
            data_a.append(line_[1])
        else:
            continue
    return data_q, data_a


#写入映射好id的样本数据
def write_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in data:
            f.write(sentence)
    return


#数据预处理
def preprocess_data(path_vocab, path_train_raw, path_train_done, src_or_tgt, max_len):
    # data = read_file(path_corpus)
    # word2id_vocab_,vocab_size_ = nn_lib.build_word2id_vocab(data,path_vocab,vocab_size=vocab_size)
    word2id_vocab, vocab_size = nn_lib.read_word2id_dict(path_vocab)
    data = load_file(path_train_raw)
    data_ = []
    for line in data:
        line_ = nn_lib.sentence2id(line, word2id_vocab)
        if 'source' == src_or_tgt:
            line_ = line_[:max_len-1]+[word2id_vocab['<EOS>']]
        elif 'target' == src_or_tgt:
            # line_ = [word2id_vocab['<SOS>']]+line_[:max_len-1]+[word2id_vocab['<EOS>']]
            line_ = line_[:max_len-1]+[word2id_vocab['<EOS>']]
        line_ = nn_lib.pad_seq(line_, max_len)
        data_.append(" ".join([str(id) for id in line_])+u"\n")
    write_file(path_train_done, data_)
    return data_


if __name__ == "__main__":
    word2id_vocab, vocab_size = nn_lib.read_word2id_dict(path_vocab)
    data_q = []
    data_a = []
    for file_name in corpus_list:
        print('loading file {}'.format(path_seq2seq+file_name))
        data_q_, data_a_ = load_file(path_seq2seq+file_name)
        data_q = data_q + data_q_
        data_a = data_a + data_a_
    word2id_vocab_, vocab_size_ = nn_lib.build_word2id_vocab(data=data_q, saved_path=path_vocab,
                                                             vocab_size=vocab_size, use_seg=True)
    print(1)





