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
from sklearn.model_selection import train_test_split
import copy
import pickle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",)

# 修改当前工作目录
# todo change dir
os.chdir(u'E:\\MachineLearning\\nlp')
# os.chdir(u'D:\\nlp')
print(os.getcwd())
from neural_network import NeuralNetwork
import nn_lib

# 开关
flag_test = False
flag_build_vocab = False
flag_process_data = False
flag_train = False
flag_infer = True
flag_transfer_learning = True


# 超参数
word_embd_dim = 200
dim_rnn = word_embd_dim
learning_rate = 1e-3
batch_size = 128*2
keep_prob = 0.95

path_data = u'.\\data\\'
path_seq2seq = path_data+u'seq2seq_nmt\\'
path_corpus_processed = path_data + u'seq2seq_nmt\\corpus_processed\\'
processed_corpus_names = ['x_train', 'x_test', 'x_vali', 'x_extended_train', 'x_extended_test', 'x_extended_vali',
                          'y_train', 'y_test', 'y_vali', 'y_extended_train', 'y_extended_test', 'y_extended_vali',
                          'vocab_extend_train', 'vocab_extend_test', 'vocab_extend_vali']

# 源序列词库相关参数
vocab_size_src = 8000
path_corpus_src = path_seq2seq+'train.txt.en'
path_vocab_src = path_seq2seq+'vocab_en.pkl'
path_corpus_processed_src = path_seq2seq+'corpus_processed_en'
src_seq_len_max = 200
# 目标序列词库相关参数
vocab_size_tgt = 4000
path_corpus_tgt = path_seq2seq+'train.txt.zh'
path_vocab_tgt = path_seq2seq+'vocab_zh.pkl'
path_corpus_processed_tgt = path_seq2seq+'corpus_processed_zh'
tgt_seq_len_max = 100


# 读取样本数据
def read_file(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(line.strip().split())
    return data


#写入映射好id的样本数据
def write_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in data:
            f.write(sentence)
    return


# 构建字典
def build_vocab():
    global path_corpus_src, path_vocab_src, vocab_size_src
    global path_corpus_tgt, path_vocab_tgt, vocab_size_tgt
    data_en = read_file(path_corpus_src)
    word2id_vocab_en, vocab_size_src = nn_lib.build_word2id_vocab(data_en, path_vocab_src, vocab_size=vocab_size_src)
    data_zh = read_file(path_corpus_tgt)
    word2id_vocab_zh, vocab_size_tgt = nn_lib.build_word2id_vocab(data_zh, path_vocab_tgt, vocab_size=vocab_size_tgt)
    return word2id_vocab_en, word2id_vocab_zh


# 数据预处理
def preprocess_data():
    # 读取字典
    word2id_vocab_src, vocab_size_src = nn_lib.read_word2id_dict(path_vocab_src)
    word2id_vocab_tgt, vocab_size_tgt = nn_lib.read_word2id_dict(path_vocab_tgt)
    # 读取语料
    corpus_src = read_file(path_corpus_src)
    corpus_tgt = read_file(path_corpus_tgt)
    # 转换语料文本至id值
    src_ids = []
    src_id_extendeds = []
    word2id_vocab_extends = []
    tgt_ids = []
    tgt_id_extendeds = []
    for src, tgt in zip(corpus_src, corpus_tgt):
        # 转换src至ID
        src_id, src_id_extended, vocab_extend_raw = nn_lib.sentence2id(sent=src,
                                                                   word2id_vocab=word2id_vocab_src,
                                                                   build_extend_vocab=True)
        src_ids.append(src_id)
        src_id_extendeds.append(src_id_extended)
        vocab_extend = {key: value-len(word2id_vocab_src)+len(word2id_vocab_tgt) for key, value in vocab_extend_raw.items()}
        word2id_vocab_extends.append(copy.copy(vocab_extend))
        # 转换tgt至ID
        tgt_id = nn_lib.sentence2id(sent=tgt, word2id_vocab=word2id_vocab_tgt, build_extend_vocab=False)
        vocab_extend.update(word2id_vocab_tgt)
        tgt_id_extended = nn_lib.sentence2id(sent=tgt, word2id_vocab=vocab_extend, build_extend_vocab=False)
        tgt_ids.append(tgt_id)
        tgt_id_extendeds.append(tgt_id_extended)
    del src, src_id, src_id_extended, tgt, tgt_id, tgt_id_extended
    # 序列补0，统计附加词表大小
    src_ids = nn_lib.pad_sequences(sequences=src_ids, max_seq_len=src_seq_len_max)
    src_id_extendeds = nn_lib.pad_sequences(sequences=src_id_extendeds, max_seq_len=src_seq_len_max)
    tgt_ids = nn_lib.pad_sequences(sequences=tgt_ids, max_seq_len=tgt_seq_len_max, add_sos=True)
    tgt_id_extendeds = nn_lib.pad_sequences(sequences=tgt_id_extendeds, max_seq_len=tgt_seq_len_max, add_sos=True)
    vocab_size_extened = max([len(i) for i in word2id_vocab_extends])
    # 规整数据
    src_ids = np.array(src_ids)
    src_id_extendeds = np.array(src_id_extendeds)
    tgt_ids = np.array(tgt_ids)
    tgt_id_extendeds = np.array(tgt_id_extendeds)
    word2id_vocab_extends = np.array(word2id_vocab_extends).reshape([-1, 1])
    # 构建训练集、测试集、验证集
    x, x_vali, x_extended, x_extended_vali, y, y_vali, y_extended, y_extended_vali, vocab_extend, vocab_extend_vali \
        = train_test_split(src_ids, src_id_extendeds, tgt_ids, tgt_id_extendeds, word2id_vocab_extends, test_size=128*3)
    x_train, x_test, x_extended_train, x_extended_test, y_train, y_test, y_extended_train, y_extended_test, vocab_extend_train, vocab_extend_test \
        = train_test_split(x, x_extended, y, y_extended, vocab_extend, test_size=128*2)
    del x, x_extended, y, y_extended, vocab_extend
    # 存储训练集、测试集、验证集
    for name in processed_corpus_names:
        with open(path_corpus_processed + name, 'wb') as file:
            pickle.dump(eval(name), file)
    return


# 读取语料
def load_processed_corpus():
    corpus = {}
    for data_name in processed_corpus_names:
        with open(path_corpus_processed + data_name, 'rb') as file:
            data = pickle.load(file, encoding='utf-8')
            corpus[data_name] = data
    return corpus


# 机器翻译测试用例
def nmt(model, corpus_src,
        path_vocab_src=path_vocab_src, path_vocab_tgt=path_vocab_tgt,
        src_seq_len_max=src_seq_len_max):
    # 读取字典
    word2id_vocab_src, vocab_size_src = nn_lib.read_word2id_dict(path_vocab_src)
    word2id_vocab_tgt, vocab_size_tgt = nn_lib.read_word2id_dict(path_vocab_tgt)
    id2word_vocab_tgt = {value: key for key, value in word2id_vocab_tgt.items()}
    ids = []
    id_extendeds = []
    vocab_extends = []
    # 处理输入语料数据
    for sentence in corpus_src:
        sent = sentence.strip().split()
        id, id_extended, vocab_extend_raw = nn_lib.sentence2id(sent=sent, word2id_vocab=word2id_vocab_src, build_extend_vocab=True)
        ids.append(id)
        id_extendeds.append(id_extended)
        vocab_extend = {key: value-len(word2id_vocab_src)+len(word2id_vocab_tgt) for key, value in vocab_extend_raw.items()}
        vocab_extends.append(copy.copy(vocab_extend))
    # 序列补0，统计附加词表大小
    ids = nn_lib.pad_sequences(sequences=ids, max_seq_len=src_seq_len_max)
    id_extendeds = nn_lib.pad_sequences(sequences=id_extendeds, max_seq_len=src_seq_len_max)
    vocab_size_extened = max([len(i) for i in vocab_extends])
    # 规整数据
    ids = np.array(ids)
    id_extendeds = np.array(id_extendeds)
    vocab_extends = np.array(vocab_extends).reshape([-1, 1])
    data = [ids, id_extendeds, vocab_extends]
    # 进行预测，输出时序概率分布
    tgt_prob_seqs = model.infer(data=data)
    # 转换预测结果至自然语言语句
    tgt_seqs = []
    for seq in tgt_prob_seqs:
        seq = np.argmax(seq, axis=1)
        seq = [id2word_vocab_tgt[id] for id in seq]
        seq = np.array(seq).reshape([-1, 1])
        tgt_seqs.append(seq)
    corpus_tgt = np.concatenate(tgt_seqs, axis=1)
    corpus_tgt = [''.join([tmp for tmp in corpus_tgt[i, :] if tmp != '<PAD>']) for i in range(corpus_tgt.shape[0])]
    return corpus_tgt


def test():
    word2id_vocab_src, vocab_size_src = nn_lib.read_word2id_dict(path_vocab_src)
    word2id_vocab_tgt, vocab_size_tgt = nn_lib.read_word2id_dict(path_vocab_tgt)
    id2word_vocab_src = {value: key for key, value in word2id_vocab_src.items()}
    id2word_vocab_tgt = {value: key for key, value in word2id_vocab_tgt.items()}
    # ' '.join([id2word_vocab_src[i] for i in corpus['x_test'][100]])
    # ''.join([id2word_vocab_tgt[i] for i in corpus['y_test'][100]])
    return


if __name__ == "__main__":
    # 创建字典
    if flag_build_vocab is True:
        word2id_vocab_en, word2id_vocab_zh = build_vocab()
    # 预处理数据
    if flag_process_data is True:
        preprocess_data()
    corpus = load_processed_corpus()
    data = [corpus['x_train'], corpus['x_extended_train'], corpus['vocab_extend_train'], corpus['y_train'], corpus['y_extended_train']]
    data_test = [corpus['x_test'], corpus['x_extended_test'], corpus['vocab_extend_test'], corpus['y_test'], corpus['y_extended_test']]
    model = NeuralNetwork(data=data, task_type='seq_generation',
                          model_type='seq2seq', loss_fun_type='cross_entropy_seq2seq',
                          eval_score_type='cross_entropy_seq2seq', optimizer_type='Adam',
                          model_parameter={'keep_prob': keep_prob,
                                           'word_embd_dim': word_embd_dim,
                                           'dim_rnn': dim_rnn,
                                           'use_same_word_embd': False,
                                           'encoder_word_embd_pretrain': None,
                                           'encoder_vocab_size': vocab_size_src,
                                           'decoder_word_embd_pretrain': None,
                                           'decoder_vocab_size': vocab_size_tgt,
                                           'target_seq_len_max': tgt_seq_len_max,
                                           'batch_size': batch_size},
                          hyper_parameter={'learning_rate': learning_rate,
                                           'built_in_test_rounds': 10,
                                           'early_stop_rounds': 100},
                          other_parameter={'model_save_rounds': 10,
                                           'path_data': path_seq2seq}
                          )
    # 训练
    if flag_train:
        model.train(transfer_learning=flag_transfer_learning, built_in_test=False, data_test=data_test)
    # 预测
    if flag_infer:
        en = ['win',
              'go']
        zh = nmt(model=model, corpus_src=en)
        for sent in zh:
            print(sent)

    print('Task End.')


