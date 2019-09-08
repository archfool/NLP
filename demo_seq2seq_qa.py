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

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )

# 修改当前工作目录
os.chdir(u'E:\\MachineLearning\\nlp')
print(os.getcwd())
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
import nn_lib

# 开关
flag_test = True
flag_build_vocab = False
flag_process_data = False

# 超参数
word_embd_dim = 100
dim_rnn = word_embd_dim
learning_rate = 1e-3
batch_size = 128
max_seq_len = 50
keep_prob = 0.9

path_data = u'.\\data\\'
path_seq2seq = path_data + u'seq2seq_qa\\'
path_corpus = path_data + u'seq2seq_qa\\corpus\\'
path_corpus_processed = path_data + u'seq2seq_qa\\corpus_processed\\'
corpus_list = ['qingyun.tsv'] if flag_test else ['qingyun.tsv', 'chatterbot.tsv', 'ptt.tsv', 'weibo.tsv']
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
    a_ids = []
    a_id_extendeds = []
    word2id_vocab_extends = []
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
        a_id_extended = nn_lib.sentence2id(sent=a_seg, word2id_vocab=vocab_extend)
        a_ids.append(a_id)
        a_id_extendeds.append(a_id_extended)
    del q, q_seg, q_id, q_id_extended, a, a_seg, a_id, a_id_extended
    # 规整数据
    q_ids, src_seq_len = nn_lib.pad_sequences(sequences=q_ids, max_seq_len=max_seq_len)
    q_id_extendeds, src_seq_len = nn_lib.pad_sequences(sequences=q_id_extendeds, max_seq_len=max_seq_len)
    a_ids, tgt_seq_len = nn_lib.pad_sequences(sequences=a_ids, max_seq_len=max_seq_len)
    a_id_extendeds, tgt_seq_len = nn_lib.pad_sequences(sequences=a_id_extendeds, max_seq_len=max_seq_len)
    vocab_size_extened = max([len(i) for i in word2id_vocab_extends])
    q_ids = np.array(q_ids)
    q_id_extendeds = np.array(q_id_extendeds)
    a_ids = np.array(a_ids)
    a_id_extendeds = np.array(a_id_extendeds)
    word2id_vocab_extends = np.array(word2id_vocab_extends).reshape([-1, 1])
    x, x_vali, x_extended, x_extended_vali, y, y_vali, y_extended, y_extended_vali, vocab_extend, vocab_extend_vali \
        = train_test_split(q_ids, q_id_extendeds, a_ids, a_id_extendeds, word2id_vocab_extends, test_size=10000)
    x_train, x_test, x_extended_train, x_extended_test, y_train, y_test, y_extended_train, y_extended_test, vocab_extend_train, vocab_extend_test \
        = train_test_split(x, x_extended, y, y_extended, vocab_extend, test_size=10000)
    del x, x_extended, y, y_extended, vocab_extend
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
    model = NeuralNetwork(data_x=corpus['x_train'], data_y=corpus['y_train'], task_type='seq_generation',
                          model_type='seq2seq', loss_fun_type='cross_entropy_seq2seq',
                          eval_score_type='cross_entropy_seq2seq', optimizer_type='Adam',
                          model_parameter={'keep_prob': keep_prob,
                                           'x_id_extended': corpus['x_extended_train'],
                                           'y_id_extended': corpus['y_extended_train'],
                                           'vocab_size_extend': max([len(vocab) for vocab in corpus['vocab_extend_train']]),
                                           'word_embd_dim': word_embd_dim,
                                           'dim_rnn': dim_rnn,
                                           'use_same_word_embd': True,
                                           'encoder_word_embd_pretrain': None,
                                           'encoder_vocab_size': vocab_size,
                                           'target_seq_len_max': max_seq_len,
                                           'batch_size': batch_size},
                          hyper_parameter={'learning_rate': learning_rate,
                                           'early_stop_rounds': 150},
                          other_parameter={'model_save_rounds': 50,
                                           'path_data': path_seq2seq}
                          )
    #训练
    if True:
        other_feed = {
            'x_extended_train': corpus['x_extended_train'],
            'y_extended_train': corpus['y_extended_train'],
            'vocab_size_extend_train': max([len(vocab) for vocab in corpus['vocab_extend_train']]),
            'x_extended_infer': corpus['x_extended_test'],
            'y_extended_infer': corpus['y_extended_test'],
            'vocab_size_extend_infer': max([len(vocab) for vocab in corpus['vocab_extend_test']]),
        }
        model.train(transfer_learning=False, built_in_test=True,
                    x_test=corpus['x_test'], y_test=corpus['x_test'],
                    other_feed=other_feed)

    print('Task End.')



