# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:51:01 2019

@author: ruan
"""

import numpy as np
import pandas as pd
from pandas import DataFrame as dataframe,Series as series
import os
import time
import re
import multiprocessing
import sys, pickle, os, random
import tensorflow as tf
import logging
logging.basicConfig(level=logging.WARNING,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)
from sklearn.model_selection import train_test_split
from neural_network import neural_network

path_data = r'E:\\MachineLearning\\data\\mr2mos_201905\\'
#RSRP
cols_feature1=['Cell 1st RSRP','Cell 2nd RSRP','Cell 3rd RSRP',\
               'Cell 4th RSRP','Cell 5th RSRP','Cell 6th RSRP']
rsrp_fillna = -150
rsrp_mean = -85
rsrp_std = 12
#RSRQ
cols_feature2=['Cell 1st RSRQ','Cell 2nd RSRQ','Cell 3rd RSRQ',\
               'Cell 4th RSRQ','Cell 5th RSRQ','Cell 6th RSRQ']
rsrq_fillna = -50
rsrq_mean = -7
rsrq_std = 3
#feature
feature_col = [\
#'No.',
#'UETime',
#'PCTime',
#'Lon',
#'Lat',
'RSRP',
'RSRQ',
'SINR',
'RSSI',
'Cell 1st RSRP',
'Cell 2nd RSRP',
'Cell 3rd RSRP',
'Cell 4th RSRP',
'Cell 5th RSRP',
'Cell 6th RSRP',
'Cell 1st PCI',
'Cell 2nd PCI',
'Cell 3rd PCI',
'Cell 4th PCI',
'Cell 5th PCI',
'Cell 6th PCI',
#'PCI',
'Cell 1st RSRQ',
'Cell 2nd RSRQ',
'Cell 3rd RSRQ',
'Cell 4th RSRQ',
'Cell 5th RSRQ',
'Cell 6th RSRQ',
'WideBand CQI code0',
'Frequency DL(MHz)',
'Voice RFC1889 Jitter',
'Voice Packet Loss Rate',
'Voice Packet Delay',
]
target_col = 'POLQA MOS SWB'
#超参数
learning_rate = 1e-5
batch_size = 1024*8

if not 'dt_data' in locals():
    dt_data = pd.read_csv(path_data+r'fuzhaogaosu.csv')
    dt_data.dropna(subset=['POLQA MOS SWB','RSRQ','Voice RFC1889 Jitter',\
                           'Voice Packet Loss Rate','Voice Packet Delay',\
                           'RSRP','RSRQ'], inplace=True)
    #给邻区RSRP为空值的样本，赋值-150dB
    for col in [x for x in dt_data.columns if re.search('RSRP',x)]:
        dt_data[col].fillna(value=rsrp_fillna, inplace=True)
        dt_data[col] = dt_data[col].apply(lambda x: (x-rsrp_mean)/rsrp_std)
    
    #给邻区RSRQ为空值的样本，赋值-30dB
    for col in [x for x in dt_data.columns if re.search('RSRQ',x)]:
        dt_data[col].fillna(value=rsrq_fillna, inplace=True)
        dt_data[col] = dt_data[col].apply(lambda x: (x-rsrq_mean)/rsrq_std)
    #    #将RSRP大于门限值（差值大于-6）的邻区标记为干扰邻区
    #    for i in range(1,7):
    #        dt_data['interfere_cell_{}'.format(i)] = (dt_data['Cell {} RSRP'.format(i)]-dt_data['RSRP']>-6).apply(int)
    #    dt_data['interfere_cell_count']=dt_data[['interfere_cell_{}'.format(i) for i in range(1,7)]].apply(sum, axis=1)
    #        #对EARFCN频点特征进行处理
    #        for col in [x for x in dt_data.columns if re.search('EARFCN',x)]:
    #            dt_data[re.sub('EARFCN','band_width',col)] = dt_data[col].map({1825:75.0,100:100.0,2452:25.0})
    #            dt_data[re.sub('EARFCN','band_width',col)] = dt_data[re.sub('EARFCN','band_width',col)].apply(lambda x: x if x in [25.0,75.0,100.0] else np.nan)
    #            dt_data[re.sub('EARFCN','band_width',col)] = dt_data[re.sub('EARFCN','band_width',col)].apply(lambda x: (x-50)/50)
    #        dt_data.dropna(subset=['band_width DL'], inplace=True)
    dt_data = dt_data[feature_col+[target_col]]
    dt_data.dropna(subset=dt_data.columns, inplace=True)
x = dt_data[feature_col]
y = dt_data[target_col]
random_seed = int(time.time())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=int(1e5), random_state=random_seed)
model = neural_network(x_train,y_train,task_type='regression',\
                       model_type='mlp',loss_fun_type='mse',\
                       eval_score_type='mse',optimizer_type='Adam',\
                       model_parameter={'dim':[100,100,100,100,100],\
                                        'keep_prob':[1.0],\
                                        'activation_fun':[tf.nn.relu]},\
                       hyper_parameter={'learning_rate':learning_rate,\
                                        'batch_size':batch_size,\
                                        'early_stop_rounds':150,\
                                        'built_in_test_epoch':50,\
                                        'model_save_epoch':100},\
                       path_data=path_data)
#model.train(transfer_learning=True,built_in_test=True,x_test=x_test,y_test=y_test)
y_test_ = model.predict(x_test)
#cmp = np.concatenate((np.array(y_test).reshape((-1,1)),np.array(y_test_)),axis=1)
#导出模型参数
#model.params_output()    


