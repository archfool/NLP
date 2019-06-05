# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:53:30 2019

@author: ruan
"""

import jieba
import jieba.posseg as psg

#path_nlp = r'E:\\MachineLearning\\data\\nlp\\'

#加载停用词表
def get_stopwords(path_data=None):
    if path_data==None:
        stopword_path = path_nlp+r'stopword_hgd.txt'
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

