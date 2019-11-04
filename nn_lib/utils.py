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
import re
import traceback
import numpy as np
from sklearn.model_selection import train_test_split

word2id_list_const = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    '<unk>': 3
}

# 加载停用词表
def get_stopwords(path_data=None):
    if path_data==None:
        stopword_path = r'.\\data\\stopword.txt'
    else:
        stopword_path = path_data
    stopwords = [word.replace('\n', '') for word in open(stopword_path,encoding='utf-8').readlines()]
    return stopwords

def str_segment_init(add_words=None, stop_words=None):
    if (add_words is not None) and isinstance(add_words, (list, tuple)):
        for add_word in add_words:
            # print(add_word)
            jieba.add_word(add_word)
    return

# 分词
def str_segment(sentence, pos=False):
    sentence = sentence.strip()
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.lcut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.lcut(sentence)
    return seg_list


# 建立带有词频的词表
def build_word2id_vocab(data, saved_path, vocab_size=None, use_seg=False, reserved_words=[],
                        language='chs', retain_eng=True, retain_digit=True):
    #若词表数量参数为空，则根据语言类型，配置缺省值
    if None==vocab_size:
        if 'chs'==language:
            vocab_size = 4000
        elif 'eng'==language:
            vocab_size = 8000
        else:
            print('Absence of vocab_size and language!')
    #简历word2id字典变量
    word2id = {}
    for sentence in data:
        if use_seg is True:
            sentence = str_segment(sentence, pos=False)
        else:
            sentence = list(sentence)
        for word in sentence:
            word = word.lower()
            #替换特定领域的词汇为类型名
            if (False == retain_digit) and word.isdigit():
                word = '<num>'
            elif (False==retain_eng) and (('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a')):
                word = '<eng>'
            #生成字典
            if word not in word2id:
                word2id[word] = 1
            else:
                word2id[word] += 1
    #根据词表大小，筛选出高频词
    word2id_list = sorted(word2id.items(), key=lambda word2id: word2id[1], reverse=True)\
        [:vocab_size-len(word2id_list_const)-len([word for word in reserved_words if word not in word2id_list_const.keys()])]
    #加上保留符号
    word2id_list = [tag for tag in word2id_list_const.keys()]+reserved_words+\
                   [word for word, word_freq in word2id_list if word not in reserved_words]
    #将词表由列表格式转换为字典格式
    word2id = {word: id for word, id in zip(word2id_list, range(len(word2id_list)))}
    with open(saved_path, 'wb') as file:
        pickle.dump(word2id, file)
    return word2id, len(word2id)

# 读取编码词典
def read_word2id_dict(path):
    with open(path, 'rb') as f:
        word2id = pickle.load(f)
    vocab_size = len(word2id)
    return word2id, vocab_size


# 将句子转化为ID序列
def sentence2id(sent, word2id_vocab, add_eos=True, reverse=False,
                retain_eng=True, retain_digit=True, build_extend_vocab=False):
    sent = list(sent)
    sentence_id = []
    vocab_extend = {}
    sentence_id_extended = []
    vocab_size = len(word2id_vocab)
    for word in sent:
        word = word.lower()
        # 判断是否转换数字和英文字母
        if (False == retain_digit) and word.isdigit():
            word = '<num>'
        elif (False == retain_eng) and (('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a')):
            word = '<eng>'
        # 判断是否为OOV词
        if word in word2id_vocab:
            # 使用扩展词表
            if build_extend_vocab is True:
                word_id_extend = word2id_vocab[word]
            # 不使用扩展词表
            word_id = word2id_vocab[word]
        else:
            # 使用扩展词表
            if build_extend_vocab is True:
                if word not in vocab_extend:
                    vocab_extend[word] = len(vocab_extend)+vocab_size
                word_id_extend = vocab_extend[word]
            # 不使用扩展词表
            word = '<unk>'
            word_id = word2id_vocab[word]
        # 将得到的id添加到list
        # 使用扩展词表
        if build_extend_vocab is True:
            sentence_id.append(word_id)
            sentence_id_extended.append(word_id_extend)
        # 不使用扩展词表
        else:
            sentence_id.append(word_id)
     # 使用扩展词表
    if build_extend_vocab is True:
        if reverse is True:
            sentence_id = sentence_id[::-1]
            sentence_id_extended = sentence_id_extended[::-1]
        if add_eos is True:
            sentence_id.append(word2id_list_const['<eos>'])
            sentence_id_extended.append(word2id_list_const['<eos>'])
        return sentence_id, sentence_id_extended, vocab_extend
    # 不使用扩展词表
    else:
        if reverse is True:
            sentence_id = sentence_id[::-1]
        if add_eos is True:
            sentence_id.append(word2id_list_const['<eos>'])
        return sentence_id


# 将一个句子补0至固定长度
def pad_seq(seq, max_len, pad_mark=0):
    seq = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
    return seq


# 将句子序列补0至固定长度
def pad_sequences(sequences, max_seq_len, add_sos=False, pad_mark=0):
    #sequences = [seq_1,seq_2,...,seq_n]
    #seq = [word_1,word_2,...,word_m]
    #max_len = max(map(lambda x : len(x), sequences))
    max_len = max_seq_len
    seq_list = []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        # seq_[-1] = word2id_list_const['<eos>'] if seq_[-1] != 0 else 0
        if add_sos is True:
            seq_list.append([word2id_list_const['<sos>']]+seq_)
        else:
            seq_list.append(seq_)
        # seq_len_list.append(min(len(seq), max_len))
    return seq_list

# 划分训练集/测试集/验证集
def generate_train_test_vali(*arrays, **options):
    # 读取和校验参数
    file_path = options.get('file_path', None)
    corpus_names = options.get('corpus_names', None)
    data_test_size = options.get('data_test_size', None)
    data_vali_size = options.get('data_vali_size', None)
    if ((file_path is None)
            or (corpus_names is None)
            or (data_test_size is None)
            or (data_vali_size is None)):
        print('some para is missing !!!')
        return
    corpus_dict = {}
    # 划分验证集
    result = train_test_split(*arrays, test_size=data_vali_size)
    corpus_names_vali = [name+'_vali' for name in corpus_names]
    result_name = []
    for name, name_vali in zip(corpus_names, corpus_names_vali):
        result_name.append(name)
        result_name.append(name_vali)
    for name, data in zip(result_name, result):
        corpus_dict[name] = data
    # 划分训练集和测试集
    arrays = tuple([corpus_dict[name] for name in corpus_names])
    result = train_test_split(*arrays, test_size=data_test_size)
    corpus_names_train = [name+'_train' for name in corpus_names]
    corpus_names_test = [name+'_test' for name in corpus_names]
    result_name = []
    for name_train, name_test in zip(corpus_names_train, corpus_names_test):
        result_name.append(name_train)
        result_name.append(name_test)
    for name, data in zip(result_name, result):
        corpus_dict[name] = data
    # 删除原始数据
    for name in corpus_names:
        corpus_dict.pop(name)
    # 存储训练集、测试集、验证集
    for name in corpus_dict.keys():
        with open(file_path + name, 'wb') as file:
            pickle.dump(corpus_dict[name], file)
    return corpus_dict






# 提取变量名为字符串
pattren = re.compile(r'[\W+\w+]*?get_variable_name\((\w+)\)')
__get_variable_name__ = []
def get_variable_name(x):
    global __get_variable_name__
    if not __get_variable_name__:
        __get_variable_name__ = pattren.findall(traceback.extract_stack(limit=2)[0][3])
    return __get_variable_name__.pop(0)



if __name__=='__main__':
    a = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    corpus_names = ['q']
    corpus_dict = generate_train_test_vali(a, file_path=u'E:\\MachineLearning\\data\\', corpus_names=corpus_names, data_test_size=1, data_vali_size=1)
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






