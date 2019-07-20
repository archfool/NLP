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
path_seq2seq = path_data+u'seq2seq\\'
#中文词库相关参数
vocab_size_zh = 4000
path_corpus_zh = path_seq2seq+'train.txt.zh'
path_vocab_zh = path_seq2seq+'zh_vocab.pkl'
path_train_raw_zh =  path_corpus_zh
path_train_done_zh =  path_seq2seq+'train.zh'
#英文词库相关参数
vocab_size_en = 8000
path_corpus_en = path_seq2seq+'train.txt.en'
path_vocab_en = path_seq2seq+'en_vocab.pkl'
path_train_raw_en =  path_corpus_en
path_train_done_en =  path_seq2seq+'train.en'
#超参数
word_embd_dim = 50
dim_lstm = 50
learning_rate = 0.001
batch_size = 1024


#读取样本数据
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(line.strip().split())
    return data

#写入映射好id的样本数据
def write_file(file_path,data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in data:
            f.write(sentence)
    return


def to_delete():
#根据语料库建立词表
# def built_vocab(path_corpus, path_vocab, vocab_size):
#     #读取语料库单词
#     counter = collections.Counter()
#     with codecs.open(path_corpus, "r", "utf-8") as file_corpus:
#         for line in file_corpus:
#             for word in line.strip().split():
#                 counter[word] += 1
#     #根据词频进行排序
#     sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
#     sorted_words = [x[0] for x in sorted_word_to_cnt]
#     #添加UNK/SOS/EOS
#     sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
#     #截断超过词库数量的单词
#     if len(sorted_words) > vocab_size:
#         sorted_words = sorted_words[:vocab_size]
#     #输出词表
#     with codecs.open(path_vocab, 'w', 'utf-8') as file_vocab:
#         for word in sorted_words:
#             file_vocab.write(word + "\n")
#     return sorted_words

#对语料进行词库onehot映射
# def corpus_onehot(path_vocab, path_train_raw, path_train_done):
#     #读取词库，建立映射表
#     with codecs.open(path_vocab, "r", "utf-8") as file_vocab:
#         vocab = [w.strip() for w in file_vocab.readlines()]
#     word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
#     #将OOV词替换成UNK
#     def get_id(word):
#         return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]
#     #读取语料库，并进行处理
#     fin = codecs.open(path_train_raw, "r", "utf-8")
#     fout = codecs.open(path_train_done, 'w', 'utf-8')
#     for line in fin:
#         words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
#         # 将每个单词替换为词汇表中的编号
#         out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
#         fout.write(out_line)
#     fin.close()
#     fout.close()
#     return
    pass

#数据预处理
def preprocess_data(path_corpus, path_vocab, vocab_size, path_train_raw, path_train_done, target_seq, max_len):
    data = read_file(path_corpus)
    word2id_vocab_,vocab_size_ = nn_lib.build_word2id_vocab(data,path_vocab,vocab_size=vocab_size)
    word2id_vocab,vocab_size = nn_lib.read_word2id_dict(path_vocab)
    data = read_file(path_train_raw)
    data_ = []
    for line in data:
        line_ = nn_lib.sentence2id(line, word2id_vocab)
        if False == target_seq:
            line_ = line_[:max_len-1]+[word2id_vocab['<EOS>']]
        elif True == target_seq:
            line_ = [word2id_vocab['<SOS>']]+line_[:max_len-2]+[word2id_vocab['<EOS>']]
        line_ = nn_lib.pad_seq(line_, max_len)
        data_.append(" ".join([str(id) for id in line_])+u"\n")
    write_file(path_train_done, data_)
    return data,data_

if __name__ == "__main__":
    if False:
        preprocess_data(path_corpus_zh, path_vocab_zh, vocab_size_zh, path_train_raw_zh, path_train_done_zh, target_seq=True, max_len=100)
        preprocess_data(path_corpus_en, path_vocab_en, vocab_size_en, path_train_raw_en, path_train_done_en, target_seq=False, max_len=200)
    if True:
        x = read_file(path_train_done_en)
        x = np.array(x).astype(int)
        y = read_file(path_train_done_zh)
        y = np.array(y).astype(int)
        random_seed = int(time.time())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_seed)
        model = NeuralNetwork(x_train, y_train,
                              task_type='seq_generation', model_type='seq2seq',
                              loss_fun_type='cross_entropy', eval_score_type='cross_entropy', optimizer_type='Adam',
                              model_parameter={'keep_prob': 1.0,
                                               # 'is_predict':False;
                                               'dim_lstm': dim_lstm,
                                               'word_embd_dim': word_embd_dim,
                                               'encoder_word_embd_pretrain': None,
                                               'encoder_vocab_size':vocab_size_en,
                                               'decoder_word_embd_pretrain': None,
                                               'decoder_vocab_size': vocab_size_zh},
                              hyper_parameter={'learning_rate': learning_rate,
                                               'batch_size': batch_size,
                                               'model_save_rounds': 50,
                                               'early_stop_rounds': 150},
                              path_data=path_seq2seq)
        # 训练
        if True:
            model.train(transfer_learning=False, built_in_test=True, x_test=x_test, y_test=y_test)

        print(1)





