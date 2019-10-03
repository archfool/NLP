# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:52:32 2019

@author: ruan
"""

import numpy as np
from pandas import Series as series
import os
import time
import jieba
import math
from opencc import OpenCC
from gensim.corpora import WikiCorpus
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )
import codecs

path_wiki = u'E:\\MachineLearning\\data\\wiki\\'
# path_wiki = u'..\\data\\wiki\\'
flag_test = False

'''
维基百科语料下载地址：
https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
'''
# word2vec算法相关参数
w2v_dim = 100
w2v_window = 10
# min_count参数配置过大会导致报错：you must first build vocabulary before training the model
w2v_min_count = 3
w2v_iter = 5
# batch_words参数对结果有极大的影响，原因未知，默认配置为10000。
# 我曾经参试配置成1000000，最后most_similar等函数输出的结果非常差。
w2v_batch_words = 1000
# skip-gram耗时为CBOW的大约3~5倍。工业界主流用sg，据说对低频词语的效果比较好。
w2v_sg = 1


# 计算余弦相似度
def simlarityCalu(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity


# 读取和处理语料
def preprocess_wiki_corpus(path_data_in=None, path_data_out=None):
    if path_data_in == None:
        corpus_path = path_wiki + u'zhwiki-latest-pages-articles.xml.bz2'
    else:
        corpus_path = path_data_in
    if path_data_out == None:
        corpus_processed_path = path_wiki + 'corpus_wiki.txt'
    else:
        corpus_processed_path = path_data_out
    cc = OpenCC('t2s')
    count = 0
    with open(corpus_processed_path, 'w', encoding='utf-8') as corpus_processed:
        corpus = WikiCorpus(corpus_path, lemmatize=False, dictionary={})
        for doc in corpus.get_texts():
            doc_new = [' '.join(jieba.cut(cc.convert(sent), cut_all=False)) for sent in doc]
            # doc_new = series(doc).apply(lambda x: ' '.join(jieba.cut(cc.convert(x), cut_all=False)))
            corpus_processed.write(' '.join(doc_new) + "\n")
            count += 1
            if (count % 100 == 0):
                logging.warning('Saved ' + str(count) + ' articles')
            if ((flag_test == True) and (count == 200)):
                return
    return


# 训练词向量
def train_word2vec(path_corpus=None, word2vec_dim=w2v_dim, path_w2v_model=None, path_w2v_vector=None):
    # 初始化缺省地址
    if path_corpus == None:
        path_corpus = path_wiki + r'corpus_wiki.txt'
    if path_w2v_model == None:
        path_w2v_model = path_wiki + r'wiki_zh_w2v_model'
    if path_w2v_vector == None:
        path_w2v_vector = path_wiki + r'wiki_zh_w2v_vector'
    # 训练模型
    logging.warning('begin word2vec')
    model = Word2Vec(LineSentence(path_corpus), sg=w2v_sg, size=word2vec_dim, \
                     window=w2v_window, min_count=w2v_min_count, \
                     batch_words=w2v_batch_words, iter=w2v_iter, \
                     seed=int(time.time()), workers=multiprocessing.cpu_count())
    logging.warning('end word2vec')
    # 保存模型
    model.save(path_w2v_model)
    model.wv.save_word2vec_format(path_w2v_vector, binary=False)
    logging.warning('saved word2vec model')
    return model


# 读取词向量文件
def load_w2v_vector(path_w2v_vector):
    w2v_vector = KeyedVectors.load_word2vec_format(path_w2v_vector, binary=False)
    return w2v_vector

# 根据词表字典和词向量，构建新词表
def rebuild_w2v_matrix(word2id_vocab, w2v_vector):
    id2word_vocab = {idx: word for word, idx in word2id_vocab.items()}
    dim = w2v_vector.vector_size
    vocab_size = len(word2id_vocab)
    stddev = math.sqrt(6 / (vocab_size + dim))
    w2v_matrix = np.random.normal(scale=stddev, size=[vocab_size, dim])
    for i in range(vocab_size):
        if id2word_vocab[i] in w2v_vector.index2word:
            w2v_matrix[i] =w2v_vector.get_vector(id2word_vocab[i])
    return w2v_matrix


# 读取腾讯词向量文件
def load_tencent_w2v_matrix(path_data=None):
    if path_data == None:
        w2v_path = path_wiki + r'Tencent_AILab_ChineseEmbedding.txt'
        if not os.path.exists(w2v_path):
            w2v_path = path_wiki + r'wiki_zh_vector'
    else:
        w2v_path = path_data
    w2v_matrix = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    return w2v_matrix


# word2vec测试用
def w2v_demo(model):
    model.vector_size
    model.index2word
    model.get_vector('数学')
    model.most_similar(u"数学")
    model.most_similar(positive=[u"皇上", u"女人"], negative=u"男人")
    model.doesnt_match(u'数学 物理 微积分 几何 代数 数论'.split())
    model.similarity(u'书籍', u'书本')
    model.similarity(u'逛街', u'书本')


if __name__ == '__main__':
    flag_test = True
    preprocess_wiki_corpus()
    model = train_word2vec()
    w2v_vector = load_w2v_vector(path_wiki + r'wiki_zh_w2v_vector'+'_{}'.format(w2v_dim))
    print('End Task !')




