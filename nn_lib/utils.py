# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:53:30 2019

@author: ruan
"""

import jieba
import jieba.posseg as psg
import sys
import pickle
import os
import random


#加载停用词表
def get_stopwords(path_data=None):
    if path_data==None:
        stopword_path = r'.\\data\\stopword.txt'
    else:
        stopword_path = path_data
    stopwords = [word.replace('\n', '') for word in open(stopword_path,encoding='utf-8').readlines()]
    return stopwords

#分词
def str_segment(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.lcut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.lcut(sentence)
    return seg_list

#建立带有词频的词表
def build_word2id_vocab(data, vocab_saved_path, vocab_size=None, language='chs', retain_eng=True, retain_digit=True):
    # data = read_corpus(corpus_path)
    #若词表数量参数为空，则根据语言类型，配置缺省值
    if None==vocab_size:
        if 'chs'==language:
            vocab_size = 4000
        elif 'eng'==language:
            vocab_size = 8000
    #简历word2id字典变量
    word2id = {}
    for sentence in data:
        sentence = list(sentence)
        for word in sentence:
            #替换特定领域的词汇为类型名
            if (False == retain_digit) and word.isdigit():
                word = '<NUM>'
            elif (False==retain_eng) and (('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a')):
                word = '<ENG>'
            #生成字典
            if word not in word2id:
                word2id[word] = 1
            else:
                word2id[word] += 1
    #根据词表大小，筛选出高频词
    word2id_list = sorted(word2id.items(), key=lambda word2id: word2id[1], reverse=True)[:vocab_size]
    #加上保留符号
    word2id_list = ['<PAD>','<SOS>','<EOS>','<UNK>']+[word for word,word_freq in word2id_list]
    #将词表由列表格式转换为字典格式
    word2id = {word:id for word,id in zip(word2id_list, range(len(word2id_list)))}
    with open(vocab_saved_path, 'wb') as file:
        pickle.dump(word2id, file)
    return word2id,len(word2id)

#读取编码词典
def read_word2id_dict(path):
    with open(path, 'rb') as f:
        word2id = pickle.load(f)
    vocab_size = len(word2id)
    return word2id,vocab_size

#将句子转化为ID序列
def sentence2id(sent, word2id_vocab, retain_eng=True, retain_digit=True):
    sent = list(sent)
    sentence_id = []
    for word in sent:
        if (False == retain_digit) and word.isdigit():
            word = '<NUM>'
        elif (False == retain_eng) and (('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a')):
            word = '<ENG>'
        if word not in word2id_vocab:
            word = '<UNK>'
        sentence_id.append(word2id_vocab[word])
    return sentence_id

#将一个句子补0至固定长度
def pad_seq(seq, max_len, pad_mark=0):
    seq = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
    return seq

#将句子序列补0至固定长度
def pad_sequences(sequences, max_seq_len, pad_mark=0):
    #sequences = [seq_1,seq_2,...,seq_n]
    #seq = [word_1,word_2,...,word_m]
    #max_len = max(map(lambda x : len(x), sequences))
    max_len = max_seq_len
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


if __name__=='__main__':
    data = []
    with open(r'..\\data\\ner\\train_data', encoding='utf-8') as f:
        lines = f.readlines()
    sent_, label_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            label_.append(label)
        else:
            # data.append((sent_, label_))
            data.append(sent_)
            sent_, label_ = [], []
    word2id,vocab_size = build_word2id_vocab(data, r'..\\data\\ner\\word2id.pkl',retain_eng=False,retain_digit=False)
    print(vocab_size)

