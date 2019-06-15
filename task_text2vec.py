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
from opencc import OpenCC
from gensim.corpora import WikiCorpus
from gensim.models import KeyedVectors  
from gensim.models import Word2Vec  
from gensim.models import Doc2Vec  
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import logging
logging.basicConfig(level=logging.WARNING,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)
import codecs

path_nlp = r'E:\\MachineLearning\\data\\nlp\\'
flag_test = False

'''
维基百科语料下载地址：
https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
'''
#word2vec算法相关参数
w2v_dim = 100
w2v_window = 10
#min_count参数配置过大会导致报错：you must first build vocabulary before training the model
w2v_min_count = 3
w2v_iter = 10
#batch_words参数对结果有极大的影响，原因未知，默认配置为10000。
#我曾经参试配置成1000000，最后most_similar等函数输出的结果非常差。
w2v_batch_words = 1000
#skip-gram耗时为CBOW的大约3~5倍。网络上主流用sg，我也不知道为什么。
w2v_sg = 0

#doc2vec算法相关参数
d2v_dim = 100
d2v_window = 10
d2v_min_count = 3
d2v_epoch = 5
d2v_dm = 0

#计算余弦相似度
def simlarityCalu(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity

#读取和处理语料
def load_wiki_corpus(path_data_in=None, path_data_out=None, word2vec=True):
    if path_data_in==None:
        corpus_path = path_nlp+r'zhwiki-latest-pages-articles.xml.bz2'
    else:
        corpus_path = path_data_in
    if path_data_out==None:
        if word2vec==True:
            corpus_processed_path = path_nlp+'corpus_word2vec.txt'
        else:
            corpus_processed_path = path_nlp+'corpus_doc2vec.txt'
    else:
        corpus_processed_path = path_data_out
    cc=OpenCC('t2s')
    count = 0
    with open(corpus_processed_path, 'w', encoding='utf-8') as corpus_processed:
        corpus=WikiCorpus(corpus_path, lemmatize=False, dictionary={})
        if word2vec==True:
            for doc in corpus.get_texts():
                doc_new = series(doc).apply(lambda x : ' '.join(jieba.cut(cc.convert(x), cut_all=False)))
                corpus_processed.write(' '.join(doc_new)+"\n")
                count+=1
                if (count%100 == 0):
                    logging.warning('Saved '+str(count)+' articles')
                if ((flag_test==True) and (count==1000)):
                    return
        else:
            corpus.metadata = True
            for doc,(page_id,title) in corpus.get_texts():
                doc_new = TaggedDocument(words=[word for sentence in doc for word in jieba.cut(cc.convert(sentence))], tags=[cc.convert(title)])
                corpus_processed.write(' '.join(doc_new[0])+'\t'+'\t'.join(doc_new[1])+"\n")
                count+=1
                if (count%100 == 0):
                    logging.warning('Saved '+str(count)+' articles')
                if ((flag_test==True) and (count==1000)):
                    return
    return

#文本向量化训练
def generate_text2vec_model(path_data=None, dim=None, word2vec=True):
    if word2vec==True:
        fun = generate_word2vec_model
    else:
        fun = generate_doc2vec_model
    if dim==None:
        return fun(path_data)
    else:
        return fun(path_data,dim)

#训练词向量
def generate_word2vec_model(path_data=None, word2vec_dim=w2v_dim):
    if path_data==None:
        corpus_path = path_nlp+r'corpus_word2vec.txt'
    else:
        corpus_path = path_data
    #训练模型
    logging.warning('begin word2vec')
    model = Word2Vec(LineSentence(corpus_path), sg=w2v_sg, size=word2vec_dim,\
                     window=w2v_window, min_count=w2v_min_count,\
                     batch_words=w2v_batch_words, iter=w2v_iter,\
                     seed=int(time.time()), workers=multiprocessing.cpu_count())
    logging.warning('end word2vec')
    # 保存模型
    model.save(path_nlp+r'wiki_zh_w2v_model_{}'.format(word2vec_dim)) 
    model.wv.save_word2vec_format(path_nlp+r'wiki_zh_w2v_vector_{}'.format(word2vec_dim), binary=False)
    logging.warning('saved word2vec model')
    return model

#训练句向量
def generate_doc2vec_model(path_data=None, doc2vec_dim=d2v_dim):
    if path_data==None:
        corpus_path = path_nlp+r'corpus_doc2vec.txt'
    else:
        corpus_path = path_data
    #迭代输出语料
    class LineSentence_doc2vec():
        def __iter__(self):
            for doc in open(corpus_path,'r',encoding='utf-8').readlines():
                if doc.strip()!='':
                    words,tags = doc.split('\t',maxsplit=1)
                    words = words.split(' ')
                    tags = [tag.strip() for tag in tags.split('\t')]
                    yield TaggedDocument(words=words, tags=tags)
    #训练模型
    logging.warning('begin doc2vec')
    model = Doc2Vec(LineSentence_doc2vec(), dm=d2v_dm, vector_size=d2v_dim,\
                    window=d2v_window, min_count=d2v_min_count,\
                    dbow_words=2, epochs=d2v_epoch,\
                    seed=int(time.time()), workers=multiprocessing.cpu_count())
    logging.warning('end doc2vec')
    # 保存模型  
    model.save(path_nlp+r'wiki_zh_d2v_model_{}'.format(doc2vec_dim))
    logging.warning('saved doc2vec model')
    return model

#读取词向量文件
def load_w2v_vector(path_data=None):
    if path_data==None:
        w2v_path = path_nlp+r'Tencent_AILab_ChineseEmbedding.txt'
        if not os.path.exists(w2v_path):
            w2v_path = path_nlp+r'wiki_zh_vector'
    else:
        w2v_path = path_data
    w2v_vector = KeyedVectors.load_word2vec_format(w2v_path,binary=False)
    return w2v_vector

#对新文章进行doc2vec转换
def doc2vec(file_name):
    start_alpha = 0.01
    infer_epoch = 1000
    model = Doc2Vec.load(r'E:\MachineLearning\data\nlp\wiki_zh_d2v_model_100')
    doc = [w for x in codecs.open(file_name, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all

#word2vec测试用
def w2v_demo(w2v_model):
    w2v_model.vector_size
    w2v_model.index2word
    w2v_model.get_vector('数学')
    w2v_model.most_similar(u"数学")
    w2v_model.most_similar(positive=[ u"皇上",u"女人"],negative=u"男人")
    w2v_model.doesnt_match(u'数学 物理 微积分 几何 代数 数论'.split())
    w2v_model.similarity(u'书籍',u'书本')
    w2v_model.similarity(u'逛街',u'书本')


if __name__=='__main__':
    flag_test = True
#    load_wiki_corpus()
#    generate_text2vec_model()
#    load_wiki_corpus(word2vec=False)
#    generate_text2vec_model(word2vec=False)
    


