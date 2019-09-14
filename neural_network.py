# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:23:25 2018

@author: ruan
"""

import numpy as np
import pandas as pd
from pandas import DataFrame as dataframe
from pandas import Series as series
import sklearn
import tensorflow as tf
import math
import os
import gc
import nn_model
import model_seq2seq
from sklearn.preprocessing import StandardScaler
import logging
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import nn_lib

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

flag_seq2seq = True

class NeuralNetwork(object):
    """none"""
    '''========================================================================='''
    '''===============================init part================================='''
    '''========================================================================='''
    def __init__(self, data_x, data_y, task_type=None,
                 model_type=None, model_parameter=None, hyper_parameter=None,
                 loss_fun_type=None, optimizer_type='Adam', eval_score_type=None,
                 other_parameter=None):
        # 规整样本特征值x和预测值y
        self.init_data(data_x, data_y)
        # 配置任务类型：回归，分类,文本生成，等
        if self.init_task_type(task_type) is None:
            return
        # 初始化模型参数
        self.init_model_para(model_type, model_parameter)
        # 配置目标函数类型
        if self.init_loss_fun_type(loss_fun_type) is None:
            return
        # 配置评估函数的类型
        if self.init_eval_score(eval_score_type, hyper_parameter=hyper_parameter) is None:
            return
        # 初始化优化器
        self.init_optimizer(optimizer_type, hyper_parameter)
        # 初始化其它参数
        if self.init_other_para(other_parameter=other_parameter) is None:
            return

    # 读取并规整样本数据
    def init_data(self, x=None, y=None):
        # 读取x
        if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
            x = np.array(x)
            if x.any() is False:
                print('input data x is empty!')
                return None
            else:
                try:
                    x = x.reshape([-1, self.dim_x])
                    self.x = x
                # 在self.dim_x还未定义时，初始化self.dim_x
                except:
                    if len(x.shape) == 1:
                        x = x.reshape([-1, 1])
                        print('input data_x has only one sample, change it automatically!')
                    self.x = x
                    self.dim_x = self.x.shape[1]
        else:
            x = None
        # 读取y
        if isinstance(y, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
            y = np.array(y)
            if not y.any():
                print('input data y is empty!')
                return None
            else:
                try:
                    y = y.reshape([-1, self.dim_y])
                    self.y = y
                # 在self.dim_y还未定义时，初始化self.dim_y
                except:
                    y = y.reshape([self.x.shape[0], -1])
                    self.y = y
                    self.dim_y = self.y.shape[1]
        else:
            y = None
        # 返回规整好的样本数据
        return x, y
    
    # 配置任务的类型：回归，二分类、多分类等等
    def init_task_type(self, task_type):
        if task_type in ['regression',
                         'classification',
                         'seq_generation']:
            self.task_type = task_type
            return self.task_type
        else:
            print('task_type error!')
            return None
    
    # 初始化其它参数
    def init_other_para(self, other_parameter):
        # 检测model_parameter参数是否为字典
        if not isinstance(other_parameter, dict):
            print('other_parameter is not dict type!!!')
            return
        # 配置存储数据的路径
        self.path_data = other_parameter.get('path_data', '')
        # 配置存储数据的周期
        self.model_save_rounds = other_parameter.get('model_save_rounds', 100)
        # 配置训练的epoch数
        self.epoch = other_parameter.get('epoch', 10000)
        return True

    '''========================================================================='''
    '''===============================train part================================'''
    '''========================================================================='''
    # 训练
    def train(self, x=None, y=None, transfer_learning=False,
              built_in_test=False, x_test=None, y_test=None, other_parameter=None,
              other_feed=None):
        # 读取样本特征值
        x, y = self.init_data(x, y)
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        
        # 配置参数
        self.built_in_test = built_in_test
        if (other_parameter is not None) and (isinstance(other_parameter, dict)):
            for key, value in other_parameter:
                exec('self.{}={}'.format(key, value))

        # 重置tf张量空间
        tf.reset_default_graph()
        
        # 构建模型
        self.model_out = self.creat_model()
        
        # 构建损失函数
        loss = self.loss_fun(self.model_out, other_feed=other_feed)
        
        # 优化器
        optimize_step = self.optimizer(loss)
        
        # 评估函数
        score = self.cal_score(y_true=self.y_ph, y_predict=self.model_out, score_type=self.eval_score_type)
        
        # 初始化内置验证集评估模块
        self.built_in_test_stop_flag = False
        if built_in_test is True:
            x_test, y_test = self.init_data(x_test, y_test)
            if flag_seq2seq:
                feed_dict = self.get_feed_dict({self.x_ph: x_test, self.y_ph: y_test,
                                                self.x_extended_ph: other_feed['x_extended_infer'],
                                                self.y_extended_ph: other_feed['y_extended_infer'],
                                                self.batch_size_ph: x_test.shape[0],
                                                self.vocab_size_extend_ph: other_feed['vocab_size_extend_infer']},
                                               train_or_infer='train', other_feed=other_feed)
                built_in_test_cal = self.cal_score(y_true=self.y_extended_ph, y_predict=self.model_out,
                                                   score_type=self.eval_score_type)
            else:
                feed_dict_test = self.get_feed_dict({self.x_ph: x_test, self.y_ph: y_test},
                                                    train_or_infer='train', other_feed=other_feed)
                built_in_test_cal = self.cal_score(y_true=self.y_ph, y_predict=self.model_out, score_type=self.eval_score_type)
            self.built_in_test_stop_score_list = []
        
        # 初始化模型存取器
        saver = tf.train.Saver(max_to_keep=10)
        if not os.path.exists(self.path_data+'model_save\\'):
            os.makedirs(self.path_data+'model_save\\')
        
        # 开始训练模型
        with tf.Session() as sess:
            # 判断是否进行迁移学习：是新建模型，还是读取已有的模型配置
            if transfer_learning is True:
                # 读取之前训练的模型参数
                model_path = tf.train.latest_checkpoint(self.path_data+'model_save\\')
                saver.restore(sess, model_path)
                self.step = int(model_path.split('-')[-1])+1
                print(model_path)
            else:
                # 初始化变量
                sess.run(tf.global_variables_initializer())
                # 初始化step计数器
                self.step = 0
            # 外循环，所有样本循环若干次
            for i in range(self.epoch):
                # todo shuffle
                # 将数据随机打乱
                # x = tf.random_shuffle(tf.concat((x, y), axis=1))
                if flag_seq2seq:
                    data_concat = sklearn.utils.shuffle(
                        np.concatenate((x, y, other_feed['x_extended_train'], other_feed['y_extended_train']), axis=1))
                else:
                    data_concat = sklearn.utils.shuffle(np.concatenate((x, y), axis=1))
                # y = x[:, -self.dim_y:]
                # x = x[:, :-self.dim_y]
                # 内循环，每次选取批量的样本进行迭代
                for step in range(len(x)//self.batch_size):
                    self.step = self.step+1
                    data_batch = data_concat[step*self.batch_size: (step+1)*self.batch_size]
                    if flag_seq2seq:
                        x_batch = data_batch[:, 0:self.dim_x]
                        y_batch = data_batch[:, self.dim_x:self.dim_x+self.dim_y]
                        x_extend_batch = data_batch[:, self.dim_x+self.dim_y:self.dim_x*2+self.dim_y]
                        y_extend_batch = data_batch[:, self.dim_x+self.dim_y:self.dim_x*2+self.dim_y]
                        # todo to delete
                        # vocab_size_extend_batch = data_batch[:, -1:]
                        feed_dict = self.get_feed_dict({self.x_ph: x_batch, self.y_ph: y_batch,
                                                        self.x_extended_ph: x_extend_batch,
                                                        self.y_extended_ph: y_extend_batch,
                                                        self.batch_size_ph: x_batch.shape[0],
                                                        self.vocab_size_extend_ph: other_feed['vocab_size_extend_train']},
                                                       train_or_infer='train', other_feed=other_feed)
                    else:
                        x_batch = data_batch[:, 0:self.dim_x]
                        y_batch = data_batch[:, self.dim_x:self.dim_y]
                        # batch_x = x[step*self.batch_size: (step+1)*self.batch_size]
                        # batch_y = y[step*self.batch_size: (step+1)*self.batch_size]
                        feed_dict = self.get_feed_dict({self.x_ph: x_batch, self.y_ph: y_batch},
                                                       train_or_infer='train', other_feed=other_feed)
                    _, loss_value, score_value = sess.run([optimize_step, loss, score], feed_dict=feed_dict)
                    # 每次迭代输出一次评估得分
                    print('step', self.step, self.eval_score_type, score_value)
                    # early_stop判断
                    self.early_stop_judge(score_value)
                    if self.early_stop_flag:
                        print('good model!')
                        break
                    # 周期性存储模型参数，并进行验证集验证
                    if self.step % self.model_save_rounds == 0:
                        saver.save(sess, self.path_data+u'model_save\\mnist.ckpt', global_step=self.step)
                        gc.collect()
                    if (built_in_test is True) and ((self.step % self.built_in_test_rounds) == 0):
                        built_in_test_score = sess.run(built_in_test_cal, feed_dict=feed_dict_test)
                        print('====== built_in_test', self.eval_score_type, built_in_test_score, ' ======')
                        self.built_in_test_stop_judge(built_in_test_score)
                        if self.built_in_test_stop_flag:
                            print('good model!')
                            break
                if self.early_stop_flag or self.built_in_test_stop_flag:
                    print('good model!')
                    break
            saver.save(sess, self.path_data+u'model_save\\mnist.ckpt', global_step=self.step)
            print('Train model end!')
        return
    
    '''========================================================================='''
    '''==============================predict part==============================='''
    '''========================================================================='''
    # 预测
    def infer(self, x, other_feed=None):
        # 规整特征数据
        x, _ = self.init_data(x=x)
        
        # 重置tf的向量空间
        tf.reset_default_graph()
        
        # 构建模型
        model_out = self.creat_model()
        # 初始化模型存取器
        model_path = tf.train.latest_checkpoint(self.path_data+u'model_save\\')
        # model_path = tf.train.latest_checkpoint(os.path.join(self.path_data,'model_save'))
        # saver = tf.train.import_meta_graph(model_path+u'.meta')
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            # 读取模型参数
            saver.restore(sess, model_path)
            # 进行预测
            feed_dict = self.get_feed_dict({self.x_ph: x}, train_or_infer='infer', other_feed=other_feed)
            result = sess.run(model_out, feed_dict=feed_dict)
            if self.model_type == 'bilstm_crf':
                logits = np.array(result)
                seq_lens = np.array(sess.run(self.seq_len, feed_dict=feed_dict))
                transition_score_matrix = np.array(sess.run(self.transition_score_matrix))
                label_list = []
                for logit, seq_len in zip(logits, seq_lens):
                    viterbi_seq, _ = nn_lib.viterbi_decode(logit[:seq_len], transition_score_matrix)
                    label_list.append(viterbi_seq)
                result = label_list
            # 打印变量
            if False:
                for var in tf.global_variables():
                    print(var)
                    var_value = sess.run(var)
                    print(var_value)
        return result
    
    '''========================================================================'''
    '''=============================evaluate part=============================='''
    '''========================================================================'''
    # 配置损失函数类型
    def init_loss_fun_type(self, loss_fun_type=None):
        if not loss_fun_type:
            if self.task_type == 'regression':
                self.loss_fun_type = 'mse'
            elif self.task_type == 'classification':
                self.loss_fun_type = 'cross_entropy'
            elif self.task_type == 'seq_generation':
                self.loss_fun_type = 'cross_entropy_seq2seq'
        else:
            self.loss_fun_type = loss_fun_type
        return self.loss_fun_type

    # 损失函数
    def loss_fun(self, model_out, other_feed=None):
        y_true = self.y_ph
        if self.loss_fun_type == 'mse':
            loss = tf.reduce_mean(tf.square(tf.subtract(model_out, y_true)))
        elif self.loss_fun_type == 'mae':
            loss = tf.reduce_mean(tf.abs(tf.subtract(model_out, y_true)))
        elif self.loss_fun_type == 'cross_entropy':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out, labels=y_true))
        elif self.loss_fun_type == 'cross_entropy_seq2seq':
            y_true = other_feed['y_extended_infer']
            y_true_seq_len = tf.cast(tf.reduce_sum(tf.sign(y_true), axis=1), tf.int32)
            # todo to delete
            # index_batch_num = tf.expand_dims(tf.range(model_out[0].shape[0].value), axis=1)
            index_batch_num = tf.expand_dims(tf.range(self.batch_size_ph), axis=1)
            y_true_list = [tf.reshape(y_true[:, step_num], [-1, 1]) for step_num in range(y_true.shape[1])]
            y_true_index = [tf.stack((index_batch_num, y_true_step), axis=2) for y_true_step in y_true_list]
            loss_raw = [tf.gather_nd(logit, index) for (logit, index) in zip(model_out, y_true_index)]

            y_seq_mask = tf.cast(array_ops.sequence_mask(y_true_seq_len, self.target_seq_len_max), tf.float32)
            loss_masked = tf.concat(loss_raw, axis=1) * y_seq_mask
            loss = -tf.reduce_sum(loss_masked)/tf.reduce_sum(y_seq_mask)
            # loss = -tf.reduce_mean(loss_raw)
            # 全局变量，传递给score用
            self.loss = loss
            # step_len_max = len(model_out)
            # batch_size = model_out[0].shape[0].value
            # vocab_size = model_out[0].shape[1].value
            # y_true = [tf.reshape(tf.one_hot(indices=y_true[:, step_num], depth= vocab_size), [batch_size, vocab_size])
            #           for step_num in range(step_len_max)]
            # loss_step = [tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y) for y, y_ in zip(y_true, model_out)]
            # loss = tf.reduce_mean(loss_step)
        elif self.loss_fun_type == 'bilstm_crf':
            log_likelihood, self.transition_score_matrix =\
                nn_lib.crf_log_likelihood(inputs=self.layer_output[-1],
                                          tag_indices=y_true,
                                          sequence_lengths=self.seq_len,
                                          transition_params=self.transition_score_matrix)
            loss = -tf.reduce_mean(log_likelihood)
            # 全局变量，传递给score用
            self.loss = loss
        else:
            loss = None
        return loss
    
    # 配置评估函数的类型和评估监控器
    def init_eval_score(self, eval_score_type, hyper_parameter=None):
        # 检测hyper_parameter参数是否为字典
        if not isinstance(hyper_parameter, dict):
            print('hyper_parameter is not dict type!!!')
            return None
        if eval_score_type is None:
            if self.task_type == 'regression':
                self.eval_score_type = 'mse'
            elif self.task_type == 'classification':
                if self.dim_y == 2:
                    self.eval_score_type = 'f1_score'
            elif self.task_type == 'seq_generation':
                self.eval_score_type = 'cross_entropy_seq2seq'
        else:
            self.eval_score_type = eval_score_type
        self.built_in_test_rounds = hyper_parameter.get('built_in_test_rounds', 30)
        self.early_stop_rounds = hyper_parameter.get('early_stop_rounds', None)
        self.early_stop_score_list = []
        self.early_stop_flag = False
        return self.eval_score_type
    
    # 评估模型
    def cal_score(self, y_true, y_predict, score_type=None):
        # 确认度量方式
        if not score_type:
            score_type = self.eval_score_type
        
        if score_type in ['precision', 'recall', 'accuracy', 'f1_score']:
            # 若任务是2分类问题，则需要将2维的onehot形式预测值转换为1维
            y_true = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
            y_predict = tf.cast(tf.argmax(y_predict, axis=1), tf.int32)
            
            ones = tf.ones_like(y_true)
            zeros = tf.zeros_like(y_true)

            tp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(y_true, ones),
                        tf.equal(y_predict, ones)),
                    tf.float32))
            tn = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(y_true, zeros),
                            tf.equal(y_predict, zeros)),
                        tf.float32))
            fp = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(y_true, zeros),
                            tf.equal(y_predict, ones)),
                        tf.float32))
            fn = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(y_true, ones),
                            tf.equal(y_predict, zeros)),
                        tf.float32))
            if score_type == 'precision':
                precision = tp/(tp+fp)
                return precision
            elif score_type == 'recall':
                recall = tp/(tp+fn)
                return recall
            elif score_type == 'accuracy':
                accuracy = (tp+tn)/(tp+tn+fp+fn)
                return accuracy
            elif score_type == 'f1_score':
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1_score = 2*precision*recall/(precision+recall)
                return f1_score
            else:
                print('score_type error!')
                return
        else:
            if score_type == 'mse':
                return tf.reduce_mean(tf.square(tf.subtract(y_true, y_predict)))
            elif score_type == 'mae':
                return tf.reduce_mean(tf.abs(tf.subtract(y_true, y_predict)))
            elif score_type == 'bilstm_crf_loss':
                # todo
                # 全局变量self.loss会导致built_in_test和train的score想混淆？？？！！！
                return self.loss
            elif score_type == 'cross_entropy_seq2seq':
                # todo
                return self.loss
            else:
                print('score_type error!')
                return
    
    # early_stop判定
    def early_stop_judge(self, new_score):
        # 判断是否使用early_stop
        if self.early_stop_rounds is None:
            self.early_stop_flag = False
            return
        # 判断评估函数值是大点好，还是小点好
        if self.eval_score_type in ['mse', 'mae', 'bilstm_crf_loss']:
            minimize = True
        elif self.eval_score_type in ['f1_score', 'precision', 'recall', 'accuracy']:
            minimize = False
        else:
            minimize = True
            print('eval_score_type error!')
        # 判断early_stop
        if len(self.early_stop_score_list) < self.early_stop_rounds:
            self.early_stop_score_list = self.early_stop_score_list + [new_score]
        else:
            if self.early_stop_rounds > 10:
                n_tmp = 5
                ths = (sum(sorted(self.early_stop_score_list[-n_tmp-1:])[1:n_tmp])+new_score)/n_tmp
            else:
                ths = new_score
            if minimize is True:
                self.early_stop_flag = ([x for x in self.early_stop_score_list if x > ths] == [])
            else:
                self.early_stop_flag = ([x for x in self.early_stop_score_list if x < ths] == [])
            self.early_stop_score_list = self.early_stop_score_list + [new_score]
            self.early_stop_score_list = self.early_stop_score_list[-self.early_stop_rounds:]
        return
    
    # built_in_test_stop判定
    def built_in_test_stop_judge(self, built_in_test_score):
        # 判断是否使用built_in_test_stop
        if self.built_in_test is False:
            self.built_in_test_stop_flag = False
            return
        if self.early_stop_rounds is None:
            return
        # 判断评估函数值是大点好，还是小点好
        if self.eval_score_type in ['mse', 'mae', 'bilstm_crf_loss']:
            minimize = True
        elif self.eval_score_type in ['f1_score', 'precision', 'recall', 'accuracy']:
            minimize = False
        else:
            print('eval_score_type error!')
            return
        # 判断built_in_test_stop
        stop_lst_len = 3
        compared_stop_lst_len = 50
        compared_score_list = self.early_stop_score_list[-compared_stop_lst_len:]
#         compared_score = sum(compared_score_list)*1.0/len(compared_score_list)
        if len(self.built_in_test_stop_score_list) < stop_lst_len:
            self.built_in_test_stop_score_list = self.built_in_test_stop_score_list + [built_in_test_score]
        else:
            self.built_in_test_stop_score_list = self.built_in_test_stop_score_list + [built_in_test_score]
            self.built_in_test_stop_score_list = self.built_in_test_stop_score_list[-stop_lst_len:]
            if minimize is True:
                ths = min(self.built_in_test_stop_score_list)
                compared_score = max(compared_score_list)
                self.built_in_test_stop_flag = (compared_score < ths)
            else:
                ths = max(self.built_in_test_stop_score_list)
                compared_score = min(compared_score_list)
                self.built_in_test_stop_flag = (compared_score > ths)
        return
    
    # 初始化优化器
    def init_optimizer(self, optimizer_type, hyper_parameter):
        # 检测hyper_parameter参数是否为字典
        if not isinstance(hyper_parameter, dict):
            print('hyper_parameter is not dict type!!!')
            return
        self.optimizer_type = optimizer_type
        if self.optimizer_type == 'Adam':
            self.learning_rate = hyper_parameter.get('learning_rate', 0.0001)
        # 用于稀疏矩阵
        elif self.optimizer_type == 'Ftrl':
            self.learning_rate = hyper_parameter.get('learning_rate', 0.001)
            self.l1_regularization = hyper_parameter.get('l1_regularization', 0.0001)
            self.l2_regularization = hyper_parameter.get('l2_regularization', 0.0)
    
    # 优化器
    def optimizer(self, loss):
        if self.optimizer_type == 'Adam':
            optimize_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # 用于稀疏矩阵
        elif self.optimizer_type == 'Ftrl':
            optimizer = tf.train.FtrlOptimizer(self.learning_rate, l1_regularization_strength=self.l1_regularization,
                                               l2_regularization_strength=self.l2_regularization)
            optimize_step = optimizer.minimize(loss)
        else:
            optimize_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimize_step
    
    '''========================================================================'''
    '''===============================model part==============================='''
    '''========================================================================'''
    # 初始化模型参数
    def init_model_para(self, model_type, model_parameter):
        # 检测model_parameter参数是否为字典
        if not isinstance(model_parameter, dict):
            print('model_parameter is not dict type!!!')
            return
        # 如果模型类型字段为空，则根据任务类型，配置为缺省模型类型
        if model_type is None:
            if self.task_type == 'regression':
                self.model_type = 'mlp'
            elif self.task_type == 'classification':
                self.model_type = 'mlp'
            elif self.task_type == 'seq_generation':
                self.model_type = 'seq2seq'
        else:
            self.model_type = model_type
        
        # 配置批处理样本个数
        self.batch_size = model_parameter.get('batch_size', 1024)
        # 确定模型当前处于训练模式还是预测模式
        self.train_or_infer = model_parameter.get('train_or_infer', 'infer')

        # 根据模型类型，初始化模型参数
        if self.model_type == 'test':
            pass
        elif self.model_type == 'mlp':
            # 获取keep_prob参数
            self.keep_prob = list(model_parameter.get('keep_prob', [1.0]))
            # 获取每层网络的神经元个数
            self.dim = [self.dim_x]+list(model_parameter.get('dim', []))+[self.dim_y]
            # 获取神经网络层数
            self.layer_num = len(self.dim)-1
            # 获取激活函数
            self.activation_fun = list(model_parameter.get('activation_fun', [tf.nn.relu]))
            if len(self.activation_fun) == 1:
                self.activation_fun = self.activation_fun*(self.layer_num-1)+[None]
            else:
                self.activation_fun[-1] = None
        elif self.model_type == 'rnn_nlp':
            self.keep_prob = model_parameter.get('keep_prob', 1.0)
            self.word2vec = model_parameter.get('word2vec')
            self.dim_rnn = model_parameter.get('dim_rnn', 128)
            self.layer_num = 1
        elif self.model_type == 'bilstm_crf':
            self.keep_prob = model_parameter.get('keep_prob', 1.0)
            self.word_embd_pretrain = model_parameter.get('word_embd_pretrain')
            self.dim_rnn = model_parameter.get('dim_rnn', 200)
            self.label_num = model_parameter.get('label_num', None)
            self.vocab_num = model_parameter.get('vocab_num', None)
            self.word_embd_dim = model_parameter.get('word_embd_dim', None)
            self.layer_num = 1
        elif self.model_type == 'seq2seq':
            self.keep_prob = model_parameter.get('keep_prob', 1.0)
            self.x_extended = model_parameter.get('x_id_extended', None)
            self.y_extended = model_parameter.get('y_id_extended', None)
            self.vocab_size_extend = model_parameter.get('vocab_size_extend', None)
            self.word_embd_dim = model_parameter.get('word_embd_dim', 300)
            self.dim_rnn = model_parameter.get('dim_rnn', 300)
            self.use_same_word_embd = model_parameter.get('use_same_word_embd', True)
            self.encoder_word_embd_pretrain = model_parameter.get('encoder_word_embd_pretrain', None)
            self.encoder_vocab_size = model_parameter.get('encoder_vocab_size', None)
            self.decoder_word_embd_pretrain = model_parameter.get('decoder_word_embd_pretrain', None)
            self.decoder_vocab_size = model_parameter.get('decoder_vocab_size', None)
            self.target_seq_len_max = model_parameter.get('target_seq_len_max', 50)
        elif self.model_type == 'fm':
            # 确定隐层向量维度
            self.dim_lv = model_parameter.get('lantent_vector_dim', self.x.shape[1])
        else:
            print('model_type error!')
            return
        # 记录各层神经网络的输出
        self.layer_output = []
    
    # 构建模型
    def creat_model(self):
        self.train_or_infer_ph = tf.placeholder(tf.string, name='train_or_infer')
        if self.model_type == 'mlp':
            self.x_ph = tf.placeholder('float32', [None, self.dim_x], name='x')
            self.y_ph = tf.placeholder('float32', [None, self.dim_y], name='y')
            self.keep_prob_ph = tf.placeholder('float32', [self.layer_num], name='keep_prob')
            self.layer_output = nn_model.mlp(self.x_ph, self.keep_prob_ph,
                                             self.activation_fun, self.layer_num, self.dim)
        elif self.model_type == 'rnn_nlp':
            self.x_ph = tf.placeholder('int32', [None, self.dim_x], name='x')
            self.y_ph = tf.placeholder('float32', [None, self.dim_y], name='y')
            self.keep_prob_ph = tf.placeholder('float32', name='keep_prob')
            self.layer_output = nn_model.rnn_nlp(self.x_ph, self.keep_prob_ph,
                                                 self.word2vec, self.dim_rnn, self.dim_y)
        elif self.model_type == 'bilstm_crf':
            self.x_ph = tf.placeholder('int32', [None, self.dim_x], name='x')
            self.y_ph = tf.placeholder('int32', [None, self.dim_y], name='y')
            self.keep_prob_ph = tf.placeholder('float32', name='keep_prob')
            self.layer_output, self.seq_len, self.transition_score_matrix = nn_model.bilstm_crf(
                self.x_ph, self.keep_prob_ph,
                self.dim_rnn, self.label_num,
                self.word_embd_pretrain, self.vocab_num,
                self.word_embd_dim
            )
        elif self.model_type == 'seq2seq':
            self.x_ph = tf.placeholder('int32', [None, self.dim_x], name='x')
            self.y_ph = tf.placeholder('int32', [None, self.dim_y], name='y')
            self.keep_prob_ph = tf.placeholder('float32', name='keep_prob')
            self.batch_size_ph = tf.placeholder('int32', name='batch_size')
            self.x_extended_ph = tf.placeholder('int32', [None, self.dim_x], name='x_extended')
            self.y_extended_ph = tf.placeholder('int32', [None, self.dim_x], name='y_extended')
            self.vocab_size_extend_ph = tf.placeholder('int32', name='y_extended')
            self.encoder, self.decoder = model_seq2seq.seq2seq(
                self.x_ph, self.y_ph, self.keep_prob_ph, self.train_or_infer_ph, self.batch_size_ph,
                self.x_extended_ph, self.y_extended_ph, self.vocab_size_extend_ph,
                self.word_embd_dim, self.dim_rnn, self.use_same_word_embd,
                self.encoder_word_embd_pretrain, self.encoder_vocab_size,
                self.decoder_word_embd_pretrain, self.decoder_vocab_size,
                self.target_seq_len_max)
            self.layer_output = self.encoder + self.decoder
        elif self.model_type == 'fm':
            self.x_ph = tf.placeholder('float32', [None, self.dim_x], name='x')
            self.y_ph = tf.placeholder('float32', [None, self.dim_y], name='y')
            self.layer_output = nn_model.fm(self.x_ph, self.dim_x, self.dim_y, self.dim_lv)
        else:
            print('model_type error!')
            return
        # 返回神经网络最后一层的输出值
        return self.layer_output[-1]

    # 获取feed参数
    def get_feed_dict(self, feed_dict_init, train_or_infer, other_feed=None):
        feed_dict = feed_dict_init
        # 1.标记当前模型是训练模式还是预测模式
        # 2.配置keep_prob
        if 'train' == train_or_infer:
            feed_dict[self.train_or_infer_ph] = 'train'
            feed_dict[self.keep_prob_ph] = self.keep_prob
        else:
            feed_dict[self.train_or_infer_ph] = 'infer'
            feed_dict[self.keep_prob_ph] = 1.0
        # 根据模型是否有dropout层，以及模型用于预测还是训练，选择传入的keep_prob参数
        if self.model_type == 'test':
            pass
        elif self.model_type == 'mlp':
            if 'infer' == train_or_infer:
                feed_dict[self.keep_prob_ph] = [1.0] * self.layer_num
            else:
                if len(self.keep_prob) == 1:
                    feed_dict[self.keep_prob_ph] = self.keep_prob * (self.layer_num - 1) + [1.0]
                else:
                    feed_dict[self.keep_prob_ph] = self.keep_prob[:-1] + [1.0]
        elif self.model_type == 'rnn_nlp':
            pass
        elif self.model_type == 'seq2seq':
            # todo add batch_size
            if 'infer' == train_or_infer:
                feed_dict[self.x_extended_ph] = other_feed['x_extended_infer']
                feed_dict[self.vocab_size_extend_ph] = other_feed['vocab_size_extend_infer']
            else:
                pass
                # feed_dict[self.x_extended_ph] = other_feed.get('x_extended_train', self.x_extended)
                # feed_dict[self.y_extended_ph] = other_feed.get('y_extended_train', self.y_extended)
                # feed_dict[self.vocab_size_extend_ph] = other_feed.get('vocab_size_extend_train', self.vocab_size_extend)
        elif self.model_type == 'bilstm_crf':
            pass
        return feed_dict

    '''========================================================================'''
    '''===============================other part==============================='''
    '''========================================================================'''
    # 导数神经网络参数
    def params_output(self):
        tf.reset_default_graph()
        self.creat_model()
        model_path = tf.train.latest_checkpoint(self.path_data+'model_save\\')
        saver = tf.train.Saver()
    #         sess = tf.InteractiveSession()
        with tf.Session() as sess:
            # 读取模型参数
            saver.restore(sess, model_path)
    #         tf.global_variables()
    #         sess=tf.InteractiveSession()
    #         sess.run(tf.global_variables_initializer())
            for var in tf.global_variables():
                print(var)
                var_value = sess.run(var)
                var_name = var.name.replace(':', '_').replace(r'/', '_')
                dataframe(var_value).to_csv(self.path_data+u'{}.csv'.format(var_name), sep=',', encoding='utf_8_sig')
        #     w_1=tf.get_default_graph().get_tensor_by_name('w:0')
        #     w_1=dataframe(w_1.eval())
        #     w_1.to_csv('.\para_output'+u'\\w_1'+u'.csv',sep=',',encoding='utf_8_sig')
        return
   

def demo_regression():
    tf.reset_default_graph()
    tf.global_variables()
    
    x = np.array([x for x in range(10000)]).reshape([-1, 1])
    x = np.concatenate((x, x), axis=1)
    std_x = StandardScaler()
    x = std_x.fit_transform(x)
    y = np.array([x for x in range(10000)]).reshape([-1, 1])
    std_y = StandardScaler()
    y = std_y.fit_transform(y)
    y[2500:7500, 0] = 1-y[2500:7500, 0]
    
    model = NeuralNetwork(
        x, y, task_type='regression',
        model_type='mlp', model_parameter={'dim':[10]*3, 'keep_prob':[0.9], 'activation_fun':[tf.nn.relu]},
        loss_fun_type='mse', eval_score_type='mse',
        optimizer_type='Adam', hyper_parameter={'batch_size': 1024, 'learning_rate': 0.01},
        path_data='')
    model.train(epoch=1000)
    out = model.infer(x)
    
    if False:
        for var in tf.global_variables():
            print(var)
#             var_value=sess.run(var)
#             print(var_value)
    
    return model, x, y, out
    
#     model=NeuralNetwork(x,y,dim=[5,5,5,2],batch_size=1024,keep_prob=1.0,\
#                          activation_fun=tf.nn.relu,model_type='mlp')
#     model.train()
#     tmp=np.array([x*1000 for x in range(11)]).reshape([-1,1])
#     out=model.predict(np.concatenate((tmp,tmp,tmp),axis=1))
#     print(np.argmax(out,axis=1))


if __name__ == '__main__':
    # model, x, y, out = demo_regression()
    logging.warning('end')



