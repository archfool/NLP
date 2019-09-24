# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:01:22 2019

@author: ruan
"""

import numpy as np
from pandas import DataFrame as dataframe
from pandas import Series as series
import os
import time
import jieba
import re
import multiprocessing
import sys, pickle, os, random
import tensorflow as tf
import logging

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )
from sklearn.model_selection import train_test_split

# 修改当前工作目录
# todo change dir
os.chdir(u'E:\\MachineLearning\\nlp')
# os.chdir(u'D:\\nlp')
print(os.getcwd())
from neural_network import NeuralNetwork
import nn_lib

# 开关
flag_build_vocab = False
flag_process_data = False
flag_train = False
flag_test = True
flag_output_para = False
flag_transfer_learning =True
flag_built_in_test = True

# 超参数
word_embd_dim = 300
dim_rnn = 300
learning_rate = 1e-4
batch_size = 1024
keep_prob = 0.9

path_data = u'.\\data\\'
path_ner = path_data + u'ner\\'
path_corpus_processed = path_data + u'ner\\corpus_processed\\'
processed_corpus_names = ['x_train', 'x_test', 'x_vali',
                          'y_train', 'y_test', 'y_vali']
##NER labels, BIO
label2id = {"O": 0,
            "B-PER": 1, "I-PER": 2,
            "B-LOC": 3, "I-LOC": 4,
            "B-ORG": 5, "I-ORG": 6
            }
id2label = {id: label for label, id in label2id.items()}


# 建立字典
def build_vocab():
    data = read_ner_corpus(path_ner + r'train_data')
    data_ = [sent for sent, label in data]
    word2id, vocab_size = nn_lib.build_word2id_vocab(data_,
                                                     path_ner + r'word2id.pkl',
                                                     retain_eng=False,
                                                     retain_digit=False)
    return word2id, vocab_size


# 读取原始语料库
def read_ner_corpus(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as f:
        lines = f.readlines()
    sent_, label_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            label_.append(label)
        else:
            data.append((sent_, label_))
            sent_, label_ = [], []
    return data


# 处理语料
def preprocess_data():
    # 读取语料
    data = read_ner_corpus(path_ner + r'train_data')
    # 读取字典
    word2id, vocab_size = nn_lib.read_word2id_dict(path_ner + r'word2id.pkl')
    # 转换语料至id
    seqs, labels = [], []
    for (sent_, label_) in data:
        sent_id_ = nn_lib.sentence2id(sent_, word2id)
        label_id_ = [label2id[label] for label in label_]
        seqs.append(sent_id_)
        labels.append(label_id_)
    max_seq_len = max([len(x) for x in labels])
    # 规整语料数据
    seqs = nn_lib.pad_sequences(seqs, max_seq_len)
    seqs = np.array(seqs)
    labels = nn_lib.pad_sequences(labels, max_seq_len)
    labels = np.array(labels)
    # 构建训练集、测试集、验证集
    x, x_vali, y, y_vali = train_test_split(seqs, labels, test_size=1024)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1024)
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


# 命名实体识别预测
def ner_predict(model, x, word2id, label2id, max_len=None, do_word2id=True):
    # 反映射
    id2word = {id: word for word, id in word2id.items()}
    id2label = {id: label for label, id in label2id.items()}
    # 获取最大seq长度
    if max_len == None:
        max_len = max(map(lambda seq: len(seq), x))
    # 规整输入文本
    if do_word2id == True:
        seqs = []
        word_list = []
        for seq in x:
            seq = list(seq)
            word_list.append(seq)
            seq = nn_lib.sentence2id(seq, word2id)
            seqs.append(seq)
        seqs = nn_lib.pad_sequences(seqs, max_len)
    else:
        seqs = x
        word_list = []
        for row in x:
            word_list.append(series(row).map(id2word).tolist())
    seqs = np.array(seqs)
    # 预测标签
    label_id_list = model.infer([seqs])
    # 拼接语料和标签
    corpus_labels = []
    for i in range(len(word_list)):
        corpus_label = []
        for j in range(len(word_list[i])):
            corpus_label.append((word_list[i][j], id2label[label_id_list[i][j]]))
        corpus_labels.append(corpus_label)

    return corpus_labels



if __name__ == '__main__':
    if flag_build_vocab is True:
        build_vocab()
    if flag_process_data is True:
        preprocess_data()
    word2id, vocab_size = nn_lib.read_word2id_dict(path_ner + r'word2id.pkl')
    corpus = load_processed_corpus()
    data = [corpus['x_train'], corpus['y_train']]
    data_test = [corpus['x_test'], corpus['y_test']]
    model = NeuralNetwork(data=data, task_type='classification',
                          model_type='bilstm_crf', loss_fun_type='bilstm_crf',
                          eval_score_type='bilstm_crf_loss', optimizer_type='Adam',
                          model_parameter={'word_embd_pretrain': None,
                                           'keep_prob': keep_prob,
                                           'vocab_num': vocab_size,
                                           'word_embd_dim': word_embd_dim,
                                           'label_num': len(label2id),
                                           'dim_rnn': dim_rnn,
                                           'batch_size': batch_size},
                          hyper_parameter={'learning_rate': learning_rate,
                                           'early_stop_rounds': 150,
                                           'built_in_test_rounds': 20},
                          other_parameter={'model_save_rounds': 50,
                                           'path_data': path_ner}
                          )
    # 训练
    if flag_train is True:
        model.train(data=data, transfer_learning=flag_transfer_learning, built_in_test=flag_built_in_test, data_test=data_test)
    # 预测
    if flag_test is True:
        x_tmp = ['我明天上午十点要去深圳福田区平安大厦拜访王先生卖平安福', '五一广场', '毛泽东雕像', '万科小区', '福州市一建宿舍', '新华书店', '福建新华发行集团', \
                 '火巷里', '工商银行宿舍阳光假日公寓', '福建省二轻宿舍', '新侨联广场', \
                 '福建省新闻出版局得贵巷小区', '水部综合大楼', '高工小区', '榕城花园', '阳光城一区', \
                 '福宏小区', '水涧新村', '省工商银行宿舍']
        corpus_labels = ner_predict(model, x_tmp, word2id, label2id, max_len=None, do_word2id=True)
    # 导出模型参数
    if flag_output_para is True:
        model.params_output()
