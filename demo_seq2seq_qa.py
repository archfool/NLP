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
import pickle
import os
import time
import tensorflow as tf
import copy
import logging
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )

# 修改当前工作目录
os.chdir(u'E:\\MachineLearning\\nlp')
path_data = u'..\\data\\'
print(os.getcwd())
from neural_network import NeuralNetwork
import nn_lib

# 开关
flag_test = False
flag_build_vocab = False
flag_process_data = False

# 超参数
word_embd_dim = 100
dim_rnn = word_embd_dim
learning_rate = 1e-3
batch_size = 512
max_seq_len = 30
keep_prob = 0.95

path_seq2seq = path_data + u'seq2seq_qa\\'
path_corpus = path_data + u'seq2seq_qa\\corpus\\'
path_corpus_processed = path_data + u'seq2seq_qa\\corpus_processed\\'
corpus_list = ['qingyun.tsv'] if flag_test else ['qingyun.tsv', 'chatterbot.tsv', 'ptt.tsv']
# corpus_list = ['qingyun.tsv'] if flag_test else ['qingyun.tsv', 'chatterbot.tsv', 'ptt.tsv', 'weibo.tsv']
# 源序列词库相关参数
vocab_size = 10000
path_vocab = path_seq2seq + 'vocab.pkl'
path_train_done_src = path_seq2seq + 'train.en'
processed_corpus_names = ['x_train', 'x_test', 'x_vali', 'x_extended_train', 'x_extended_test', 'x_extended_vali',
                          'y_train', 'y_test', 'y_vali', 'y_extended_train', 'y_extended_test', 'y_extended_vali',
                          'vocab_extend_train', 'vocab_extend_test', 'vocab_extend_vali']


# 读取样本数据
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        data_q = []
        data_a = []
        while line:
            line_ = line.strip().split('\t')
            line = f.readline()
            if (len(line_)) >= 2:
                data_q.append(line_[0])
                data_a.append(line_[1])
            else:
                continue
    return data_q, data_a



# 数据预处理
def preprocess_data(vocab_size):
    # 读取语料数据
    data_q = []
    data_a = []
    for file_name in corpus_list:
        print('loading file {}'.format(path_corpus + file_name))
        data_q_, data_a_ = load_file(path_corpus + file_name)
        data_q = data_q + data_q_
        data_a = data_a + data_a_
    del data_q_, data_a_
    # 构建/读取字典
    if flag_build_vocab:
        word2id_vocab, vocab_size = nn_lib.build_word2id_vocab(data=data_q, saved_path=path_vocab,
                                                               vocab_size=vocab_size, use_seg=True)
    else:
        word2id_vocab, vocab_size = nn_lib.read_word2id_dict(path_vocab)
    # 转换语料文本至id值
    q_ids = []
    q_id_extendeds = []
    word2id_vocab_extends = []
    a_ids = []
    a_id_extendeds = []
    for q, a in zip(data_q, data_a):
        # 转换问题Q至ID
        q_seg = nn_lib.str_segment(q)
        q_id, q_id_extended, vocab_extend = nn_lib.sentence2id(sent=q_seg,
                                                               word2id_vocab=word2id_vocab,
                                                               build_extend_vocab=True)
        q_ids.append(q_id)
        q_id_extendeds.append(q_id_extended)
        word2id_vocab_extends.append(copy.copy(vocab_extend))
        # 转换答案A至ID
        a_seg = nn_lib.str_segment(a)
        a_id = nn_lib.sentence2id(sent=a_seg, word2id_vocab=word2id_vocab, build_extend_vocab=False)
        vocab_extend.update(word2id_vocab)
        a_id_extended = nn_lib.sentence2id(sent=a_seg, word2id_vocab=vocab_extend, build_extend_vocab=False)
        a_ids.append(a_id)
        a_id_extendeds.append(a_id_extended)
    del q, q_seg, q_id, q_id_extended, a, a_seg, a_id, a_id_extended
    # 序列补0，统计附加词表大小
    q_ids = nn_lib.pad_sequences(sequences=q_ids, max_seq_len=max_seq_len)
    q_id_extendeds = nn_lib.pad_sequences(sequences=q_id_extendeds, max_seq_len=max_seq_len)
    a_ids = nn_lib.pad_sequences(sequences=a_ids, max_seq_len=max_seq_len, add_sos=True)
    a_id_extendeds = nn_lib.pad_sequences(sequences=a_id_extendeds, max_seq_len=max_seq_len, add_sos=True)
    vocab_size_extened = max([len(i) for i in word2id_vocab_extends])
    # 规整数据
    q_ids = np.array(q_ids)
    q_id_extendeds = np.array(q_id_extendeds)
    a_ids = np.array(a_ids)
    a_id_extendeds = np.array(a_id_extendeds)
    word2id_vocab_extends = np.array(word2id_vocab_extends).reshape([-1, 1])
    # 构建训练集、测试集、验证集
    x, x_vali, x_extended, x_extended_vali, y, y_vali, y_extended, y_extended_vali, vocab_extend, vocab_extend_vali \
        = train_test_split(q_ids, q_id_extendeds, a_ids, a_id_extendeds, word2id_vocab_extends, test_size=1024*8)
    x_train, x_test, x_extended_train, x_extended_test, y_train, y_test, y_extended_train, y_extended_test, vocab_extend_train, vocab_extend_test \
        = train_test_split(x, x_extended, y, y_extended, vocab_extend, test_size=1024)
    del x, x_extended, y, y_extended, vocab_extend
    # 存储训练集、测试集、验证集
    for name in processed_corpus_names:
        with open(path_corpus_processed + name, 'wb') as file:
            pickle.dump(eval(name), file)
    return

def load_processed_corpus():
    corpus = {}
    for data_name in processed_corpus_names:
        with open(path_corpus_processed + data_name, 'rb') as file:
            data = pickle.load(file, encoding='utf-8')
            corpus[data_name] = data
    return corpus

# processed_corpus_names = ['x_train', 'x_test', 'x_vali', 'x_extended_train', 'x_extended_test', 'x_extended_vali',
#                           'y_train', 'y_test', 'y_vali', 'y_extended_train', 'y_extended_test', 'y_extended_vali',
#                           'vocab_extend_train', 'vocab_extend_test', 'vocab_extend_vali']
if __name__ == "__main__":
    if flag_process_data is True:
        preprocess_data(vocab_size)
    corpus = load_processed_corpus()
    data = [corpus['x_train'], corpus['x_extended_train'], corpus['vocab_extend_train'], corpus['y_train'], corpus['y_extended_train']]
    data_test = [corpus['x_test'], corpus['x_extended_test'], corpus['vocab_extend_test'], corpus['y_test'], corpus['y_extended_test']]
    model = NeuralNetwork(data=data,
                          model_type='seq2seq', loss_fun_type='cross_entropy_seq2seq',
                          model_parameter={'keep_prob': keep_prob,
                                           'word_embd_dim': word_embd_dim,
                                           'dim_rnn': dim_rnn,
                                           'use_same_word_embd': True,
                                           'encoder_word_embd_pretrain': None,
                                           'encoder_vocab_size': vocab_size,
                                           'target_seq_len_max': max_seq_len,
                                           'batch_size': batch_size},
                          hyper_parameter={'optimizer_type': 'Adam',
                                           'learning_rate': learning_rate,
                                           'eval_score_type': 'cross_entropy_seq2seq',
                                           'early_stop_rounds_train': 100,
                                           'built_in_test_interval': 1,
                                           'early_stop_rounds_test': 10},
                          other_parameter={'model_save_rounds': 20,
                                           'path_data': path_seq2seq}
                          )
    # 训练
    if True:
        model.train(transfer_learning=True, built_in_test=True, data_test=data_test)

    print('Task End.')



