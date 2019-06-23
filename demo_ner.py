# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:01:22 2019

@author: ruan
"""

import numpy as np
from pandas import DataFrame as dataframe,Series as series
import os
import time
import jieba
import re
import multiprocessing
import sys, pickle, os, random
import tensorflow as tf
import logging
logging.basicConfig(level=logging.WARNING,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)
from sklearn.model_selection import train_test_split
from neural_network import neural_network

path_nlp = r'.\\data\\'
path_ner = path_nlp+r'ner\\'
dim_lstm = 128
learning_rate = 0.001
batch_size = 1024

## labels, BIO
label2id = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
id2label = {id:label for label,id in label2id.items()}

#读取字ID字典
def read_word2id_dict(path):
    with open(path, 'rb') as f:
        word2id = pickle.load(f)
    return word2id

#随机初始化词向量矩阵
def random_w2v_embedding(word2id, embedding_dim):
    """
    :param word2id:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

#预处理数据
def train_data_process(path_train_data,word2id,label2id):
    data = read_corpus(path_train_data)
    seqs, labels = [], []
    for (sent_, label_) in data:
        sent_id_ = sentence2id(sent_, word2id)
        label_id_ = [label2id[label] for label in label_]
        seqs.append(sent_id_)
        labels.append(label_id_)
    max_seq_len = max([len(x) for x in labels])
    seqs = pad_sequences(seqs,max_seq_len)[0]
    labels = pad_sequences(labels,max_seq_len)[0]
    return seqs,labels,max_seq_len

#读取用于训练的语料库
def read_corpus(corpus_path):
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

#将句子转化为ID序列
def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

#将句子补0至固定长度
def pad_sequences(sequences, max_seq_len, pad_mark=0):
#    max_len = max(map(lambda x : len(x), sequences))
    max_len = max_seq_len
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

#命名实体识别预测
def ner_predict(model, x, word2id, label2id, max_len=None, do_word2id=True):
    #反映射
    id2word = {id:word for word,id in word2id.items()}
    id2label = {id:label for label,id in label2id.items()}
    #获取最大seq长度
    if max_len==None:
        max_len = max(map(lambda seq : len(seq), x))
    #规整输入文本
    if do_word2id==True:
        seqs = []
        word_list = []
        for seq in x:
            seq = list(seq)
            word_list.append(seq)
            seq = sentence2id(seq, word2id)
            seqs.append(seq)
        seqs = pad_sequences(seqs,max_len)[0]
    else:
        seqs = x
        word_list = []
        for row in x:
            word_list.append(series(row).map(id2word).tolist())
    #预测标签
    label_id_list = model.predict(seqs)
    label_list = []
    for row in label_id_list:
        label_list.append(series(row).map(id2label).tolist())
    #文本-标签
    word_label_list = []
    for words,labels in zip(word_list,label_list):
        word_label = []
        for word,label in zip(words,labels):
            if label==label:
                word_label.append((word,label))
            else:
                break
        word_label_list.append(word_label)
    return word_label_list

'''
def word2id_build(word2id_path, corpus_path, min_count):
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, label_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(word2id_path, 'wb') as fw:
        pickle.dump(word2id, fw)
'''

if __name__=='__main__':
    if not 'x' in locals():
        word2id = read_word2id_dict(path_ner+r'word2id.pkl')
        w2v = random_w2v_embedding(word2id,100)
        x,y,max_seq_len = train_data_process(path_ner+r'train_data',word2id,label2id)
    random_seed = int(time.time())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=random_seed)
    model = neural_network(x_train,y_train,task_type='classification',\
                           model_type='bilstm_crf',loss_fun_type='bilstm_crf',\
                           eval_score_type='bilstm_crf_loss',optimizer_type='Adam',\
                           model_parameter={'word_embd_pretrain':None,\
                                            'keep_prob':0.95,\
                                            'vocab_num':3905,\
                                            'word_embd_dim':100,\
                                            'label_num':len(label2id),\
                                            'dim_lstm':dim_lstm},\
                           hyper_parameter={'learning_rate':learning_rate,\
                                            'batch_size':batch_size,\
                                            'early_stop_rounds':150},\
                           path_data=path_ner)
    #训练
    if True:
        model.train(transfer_learning=False,built_in_test=True,x_test=x_test,y_test=y_test)
    #预测
    if False:
        #模式一（常用）
        x_tmp = ['五一广场','毛泽东雕像','万科小区','福州市一建宿舍','新华书店','福建新华发行集团',\
             '火巷里','工商银行宿舍阳光假日公寓','福建省二轻宿舍','新侨联广场',\
             '福建省新闻出版局得贵巷小区','水部综合大楼','高工小区','榕城花园','阳光城一区',\
             '福宏小区','水涧新村','省工商银行宿舍']
        word_label_list_1 = ner_predict(model,x_tmp,word2id,label2id,max_len=None,do_word2id=True)
        #模式二（不常用）
        #word_label_list_2 = ner_predict(model,x_test,word2id,label2id,max_len=max_seq_len,do_word2id=False)
    #导出模型参数
    if False:
        model.params_output()


