# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:51:34 2019

@author: ruan
"""

import math
import jieba
import jieba.posseg as psg
from jieba import analyse
from gensim import corpora, models
import functools
from nlp_lib import get_stopwords, str_segment

path_nlp = r'E:\\MachineLearning\\data\\nlp\\'

def load_data(path):
    pass


#TF-IDF类
class tf_idf(object):
    #输入参数：目标文档，语料库，关键词数量
    def __init__(self, doc=None, corpus=None, keyword_num=10):
        self.doc = doc
        self.corpus = corpus
        self.keyword_num = keyword_num
        self.idf_init = False
    
    #统计tf值
    def get_tf_dic(self, doc=None):
        if doc==None:
            doc = self.doc
        else:
            self.doc = doc
        tf_dic = {}
        for word in doc:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        word_count = len(doc)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / word_count
        #保存类共享数据
        self.tf_dic = tf_dic
        return tf_dic

    #idf值统计方法
    def get_idf_dic(self, corpus=None):
        if (corpus==None and self.idf_init==True):
            return self.idf_dic,self.default_idf
        if corpus==None:
            corpus = self.corpus
        idf_dic = {}
        # 总文档数
        doc_count = len(corpus)
        #每个词出现的文档数
        for doc in corpus:
            for word in set(doc):
                idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
        #按公式转换为idf值，分母加1进行平滑处理
        for k, v in idf_dic.items():
            idf_dic[k] = math.log(doc_count / (1.0 + v))
        #对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
        default_idf = math.log(doc_count / (1.0))
        #保存类共享数据
        self.idf_dic = idf_dic
        self.default_idf = default_idf
        self.idf_init = True
        return idf_dic,default_idf
    
    #按公式计算tf-idf
    def get_tfidf_dic(self,doc=None,corpus=None):
        self.get_tf_dic(doc)
        self.get_idf_dic(corpus)
        tfidf_dic = {}
        for word in self.doc:
            tf = self.tf_dic.get(word, 0)
            idf = self.idf_dic.get(word, self.default_idf)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf
        self.tfidf_dic = tfidf_dic
        return tfidf_dic
    
    #输出关键词
    def get_keyword(self,doc=None,corpus=None,keyword_num=None):
        tfidf_dic = self.get_tfidf_dic(doc,corpus)
        if keyword_num==None:
            keyword_num = self.keyword_num
        keyword_list = []
        #根据tf-idf排序，去排名前keyword_num的词作为关键词
        for keyword, value in sorted(tfidf_dic.items(), key=functools.cmp_to_key(self.cmp), reverse=True)[:self.keyword_num]:
            keyword_list.append((keyword,value))
            print(keyword_list[-1])
        return keyword_list
    
    #排序函数，用于topK关键词的按值排序
    def cmp(self, e1, e2):
        import numpy as np
        res = np.sign(e1[1] - e2[1])
        if res != 0:
            return res
        else:
            a = e1[0] + e2[0]
            b = e2[0] + e1[0]
            if a > b:
                return 1
            elif a == b:
                return 0
            else:
                return -1

# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(self.cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list

    #排序函数，用于topK关键词的按值排序
    def cmp(self, e1, e2):
        import numpy as np
        res = np.sign(e1[1] - e2[1])
        if res != 0:
            return res
        else:
            a = e1[0] + e2[0]
            b = e2[0] + e1[0]
            if a > b:
                return 1
            elif a == b:
                return 0
            else:
                return -1

#tfidf测试
def tfidf_demo(text):
    #停用词表
    stopwords = get_stopwords()
    #目标文档
    doc = str_segment(text)
    doc = [word for word in doc if word not in stopwords]
    #语料库
    path_corpus = path_nlp+'corpus.txt'
    corpus = []
    for line in open(path_corpus, 'r', encoding='utf-8'):
        seg_list = str_segment(line.strip())
        seg_list = [word for word in seg_list if word not in stopwords]
        corpus.append(seg_list)
    #tf_idf
    model = tf_idf(doc,corpus)
    keyword = model.get_keyword()
    return model,keyword

def textrank_demo(text, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')
    print()

def lsilsa_demo(text, model, keyword_num=10):
    #停用词表
    stopwords = get_stopwords()
    #目标文档
    doc = str_segment(text)
    doc = [word for word in doc if word not in stopwords]
    #语料库
    path_corpus = path_nlp+'corpus.txt'
    corpus = []
    for line in open(path_corpus, 'r', encoding='utf-8'):
        seg_list = str_segment(line.strip())
        seg_list = [word for word in seg_list if word not in stopwords]
        corpus.append(seg_list)
    topic_model = TopicModel(corpus, keyword_num, model=model)
    topic_model.get_simword(doc)
    return topic_model

if __name__=='__main__':
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'
#    model,keyword = tfidf_demo(text)
#    textrank_demo(text)
#    lsilsa_demo(text,'LSI')
    lsilsa_demo(text,'LSA')


