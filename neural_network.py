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
from sklearn.preprocessing import StandardScaler
import logging
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import nn_lib
import nn_model


logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class NeuralNetwork(object):
    """none"""
    '''========================================================================='''
    '''===============================init part================================='''
    '''========================================================================='''
    def __init__(self, data, model_type, loss_fun_type,
                 model_parameter=None,
                 hyper_parameter=None,
                 other_parameter=None):
        """none"""
        '''======Part 1: 配置任务模型基础信息======'''
        # 规整样本特征值x和预测值y
        self.data = self.init_data(data)
        # 配置模型类型
        self.model_type = model_type
        # 配置损失函数类型
        self.loss_fun_type = loss_fun_type
        '''======Part 2: 记录模型参数和超参数======'''
        # 记录模型参数
        if not isinstance(model_parameter, dict):
            print('model_parameter is not dict type!!!')
            return
        self.model_parameter = model_parameter
        # 记录超参数
        if not isinstance(hyper_parameter, dict):
            print('hyper_parameter is not dict type!!!')
            return
        self.hyper_parameter = hyper_parameter
        '''======Part 3: 其它初始化内容======'''
        # 配置评估函数参数
        if self.init_eval_fun(hyper_parameter=self.hyper_parameter) is None:
            return
        # 初始化其它参数
        # 检测model_parameter参数是否为字典
        if not isinstance(other_parameter, dict):
            print('other_parameter is not dict type!!!')
            return
        # 配置存储数据的路径
        self.path_data = other_parameter.get('path_data', '')
        # 配置存储数据的周期
        self.model_save_rounds = other_parameter.get('model_save_rounds', 100)
        # if (other_parameter is not None) and (isinstance(other_parameter, dict)):
        #     for key, value in other_parameter:
        #         exec('self.{}={}'.format(key, value))

    # 读取并规整样本数据
    def init_data(self, data):
        if data is None:
            return None
        # 如果data不是一个列表（即为单个数据），将它放入列表里
        if not isinstance(data, list):
            data = [data]
        # 若数据维度列表还未被初始化（即第一次初始化数据），则构建空列表
        try:
            self.data_dim[0]
        except:
            self.data_dim = []
        # 遍历data列表，处理数据
        data_processed = []
        for i, d in enumerate(data):
            if isinstance(d, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
                d = np.array(d)
                if d.any() is False:
                    print('input data_{} is empty!'.format(i))
                    return None
                else:
                    # 如果实现初始化过data_dim，则通过data_dim来reshape，否则初始化data_dim
                    try:
                        d = d.reshape([-1, self.data_dim[i]])
                        data_processed.append(d)
                    except:
                        if len(d.shape) == 1:
                            d = d.reshape([-1, 1])
                            print('input data_{} has only one sample, change it automatically!'.format(i))
                        data_processed.append(d)
                        self.data_dim.append(d.shape[1])
            else:
                data_processed.append(None)
        return data_processed


    '''========================================================================='''
    '''===============================train part================================'''
    '''========================================================================='''
    # 训练模块
    def train(self, data=None, transfer_learning=False,
              built_in_test=False, data_test=None):
        # 读取训练集样本
        data = self.init_data(data)
        if data is None:
            data = self.data
        batch_sum = data[0].shape[0]

        # 重置tf张量空间
        tf.reset_default_graph()
        
        # 构建模型
        self.model_out = self.creat_model()
        
        # 构建损失函数
        loss = self.loss_fun(self.model_out)
        
        # 创建优化器
        optimize_step = self.build_optimizer(loss)
        
        # 评估函数
        score_train = self.cal_eval_score(y_true=self.y_ph, y_infer=self.model_out, score_type=self.eval_score_type)

        # 初始化内置验证集评估模块
        self.built_in_test = built_in_test
        if built_in_test is True:
            data_test = self.init_data(data_test)
            feed_dict_test = self.get_feed_dict(data=data_test, train_or_infer='infer')
            score_test = self.cal_eval_score(y_true=self.y_ph, y_infer=self.model_out, score_type=self.eval_score_type)

        # 初始化模型存取器
        saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)
        if not os.path.exists(self.path_data+'model_save\\'):
            os.makedirs(self.path_data+'model_save\\')
        
        # 开始训练模型
        with tf.Session() as sess:
            # 判断是否进行迁移学习
            if transfer_learning is True:
                try:
                    # 读取之前训练的模型参数
                    model_path = tf.train.latest_checkpoint(self.path_data + 'model_save\\')
                    saver.restore(sess, model_path)
                    self.step = int(model_path.split('-')[-1])
                    print(model_path)
                except:
                    # 初始化变量
                    sess.run(tf.global_variables_initializer())
                    # 初始化step计数器
                    self.step = 0
            else:
                # 初始化变量
                sess.run(tf.global_variables_initializer())
                # 初始化step计数器
                self.step = 0
            # 外循环，所有样本循环若干次
            for i in range(self.hyper_parameter.get('epoch', 10000)):
                # todo shuffle
                # 将数据随机打乱
                # x = tf.random_shuffle(tf.concat((x, y), axis=1))
                data_concat = sklearn.utils.shuffle(np.concatenate(data, axis=1))
                # 内循环，每次选取批量的样本进行迭代
                for step in range(batch_sum//self.batch_size):
                    self.step = self.step+1
                    # 提取批数据
                    data_batch_concat = data_concat[step*self.batch_size: (step+1)*self.batch_size]
                    data_batch = [data_batch_concat[:, sum(self.data_dim[:i]):sum(self.data_dim[:i+1])] for i in range(len(self.data_dim))]
                    feed_dict = self.get_feed_dict(data=data_batch, train_or_infer='train')
                    # 训练一次
                    # _, loss_value, score_value_train = sess.run([optimize_step, loss, score_train], feed_dict=feed_dict)
                    sess.run(optimize_step, feed_dict=feed_dict)
                    # 评估训练集效果
                    score_value_train = sess.run(score_train, feed_dict=feed_dict)
                    # 每次迭代输出一次训练集评估得分
                    print('step', self.step, self.eval_score_type, score_value_train)
                    # 训练集early_stop判断
                    self.early_stop_flag_train, self.early_stop_scores_train = \
                        self.early_stop_judge(score_value_train, self.early_stop_scores_train, self.early_stop_rounds_train)
                    if self.early_stop_flag_train:
                        print('good model!')
                        break
                    if False:
                        tmp = []
                        y_true = self.y_extended_ph[:, 1:]
                        tmp.append(y_true)
                        # 目标序列掩码
                        y_true_seq_len = tf.cast(tf.reduce_sum(tf.sign(y_true), axis=1), tf.int32)
                        tmp.append(y_true_seq_len)
                        y_seq_mask = tf.cast(array_ops.sequence_mask(y_true_seq_len, self.target_seq_len_max),
                                             tf.float32)
                        tmp.append(y_seq_mask)
                        # 目标序列的index
                        index_batch_num = tf.expand_dims(tf.range(self.batch_size_ph), axis=1)
                        y_true_list = [tf.reshape(y_true[:, step_num], [-1, 1]) for step_num in range(y_true.shape[1])]
                        y_true_index = [tf.stack((index_batch_num, y_true_step), axis=2) for y_true_step in y_true_list]
                        tmp.append(y_true_index)
                        # 目标序列真实值的概率
                        target_probs = [tf.gather_nd(logit, index) for (logit, index) in zip(self.model_out, y_true_index)]
                        tmp.append(target_probs)
                        # 取对数，加掩码，归一化
                        loss_log = -tf.log(target_probs)
                        tmp.append(loss_log)
                        observer, tmp1 = sess.run([self.layer_output, tmp], feed_dict=feed_dict)
                        tmp2 = np.concatenate([np.reshape(np.argmax(logit, axis=1), [-1,1]) for logit in observer[-1]], axis=1)
                        print('1')
                    # 周期性存储模型参数，并进行验证集验证
                    if self.step % self.model_save_rounds == 0:
                        saver.save(sess, self.path_data+u'model_save\\mnist.ckpt', global_step=self.step)
                        gc.collect()
                    # 内置验证集校验
                    if (built_in_test is True) and ((self.step % self.built_in_test_interval) == 0):
                        # 评估测试集效果
                        score_value_test = sess.run(score_test, feed_dict=feed_dict_test)
                        # 每次迭代输出一次测试集评估得分
                        print('====== built_in_test', self.eval_score_type, score_value_test, ' ======')
                        # 测试集early_stop判断
                        self.early_stop_flag_test, self.early_stop_scores_test = \
                            self.early_stop_judge(score_value_test, self.early_stop_scores_test, self.early_stop_rounds_test)
                        if self.early_stop_flag_test:
                            print('good model!')
                            break
                if self.early_stop_flag_train or self.early_stop_flag_test:
                    print('good model!')
                    break
            saver.save(sess, self.path_data+u'model_save\\mnist.ckpt', global_step=self.step)
            print('Train model end!')
        return
    
    # 优化器
    def build_optimizer(self, loss):
        self.optimizer_type = self.hyper_parameter.get('optimizer_type')
        if self.optimizer_type == 'Adam':
            self.learning_rate = self.hyper_parameter.get('learning_rate', 0.0001)
            optimize_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # 用于稀疏矩阵
        elif self.optimizer_type == 'Ftrl':
            self.learning_rate = self.hyper_parameter.get('learning_rate', 0.001)
            self.l1_regularization = self.hyper_parameter.get('l1_regularization', 0.0001)
            self.l2_regularization = self.hyper_parameter.get('l2_regularization', 0.0)
            optimizer = tf.train.FtrlOptimizer(self.learning_rate, l1_regularization_strength=self.l1_regularization,
                                               l2_regularization_strength=self.l2_regularization)
            optimize_step = optimizer.minimize(loss)
        else:
            optimize_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimize_step

    '''========================================================================='''
    '''==============================predict part==============================='''
    '''========================================================================='''
    # 预测模块
    def infer(self, data):
        # 规整特征数据
        data = self.init_data(data)
        batch_sum = data[0].shape[0]

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
            feed_dict = self.get_feed_dict(data=data, train_or_infer='infer')
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
    # 损失函数
    def loss_fun(self, y_infer):
        y_true = self.y_ph
        if self.loss_fun_type == 'mse':
            loss = tf.reduce_mean(tf.square(tf.subtract(y_true, y_infer)))
        elif self.loss_fun_type == 'mae':
            loss = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_infer)))
        elif self.loss_fun_type == 'cross_entropy':
            y_true = tf.reshape(tf.one_hot(indices=y_true, depth=self.label_num), [-1, self.label_num])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_infer, labels=y_true))
        elif self.loss_fun_type == 'cross_entropy_seq2seq':
            # 目标序列真实值
            y_true = self.y_extended_ph[:, 1:]
            # 目标序列掩码
            y_true_seq_len = tf.cast(tf.reduce_sum(tf.sign(y_true), axis=1), tf.int32)
            y_seq_mask = tf.cast(array_ops.sequence_mask(y_true_seq_len, self.target_seq_len_max), tf.float32)
            # 目标序列的index
            index_batch_num = tf.expand_dims(tf.range(self.batch_size_ph), axis=1)
            y_true_list = [tf.reshape(y_true[:, step_num], [-1, 1]) for step_num in range(y_true.shape[1])]
            y_true_index = [tf.stack((index_batch_num, y_true_step), axis=2) for y_true_step in y_true_list]
            # 目标序列真实值的概率
            target_probs = [tf.gather_nd(logit, index) for (logit, index) in zip(y_infer, y_true_index)]
            # 取对数，加掩码，归一化
            loss_log = -tf.log(target_probs)
            loss_log_masked = tf.concat(loss_log, axis=1) * y_seq_mask
            loss = tf.reduce_sum(loss_log_masked)/tf.reduce_sum(y_seq_mask)
        elif self.loss_fun_type == 'bilstm_crf':
            log_likelihood, self.transition_score_matrix =\
                nn_lib.crf_log_likelihood(inputs=self.layer_output[-1],
                                          tag_indices=y_true,
                                          sequence_lengths=self.seq_len,
                                          transition_params=self.transition_score_matrix)
            loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = None
        return loss
    
    # 配置评估函数的类型和评估监控器
    def init_eval_fun(self, hyper_parameter=None):
        # 检测hyper_parameter参数是否为字典
        if not isinstance(hyper_parameter, dict):
            print('hyper_parameter is not dict type!!!')
            return None
        # 评估函数类型
        self.eval_score_type = hyper_parameter.get('eval_score_type', None)
        # 训练集评估函数参数
        self.early_stop_rounds_train = hyper_parameter.get('early_stop_rounds_train', None)
        self.early_stop_flag_train = False
        self.early_stop_scores_train = []
        # 测试集评估函数参数
        self.built_in_test_interval = hyper_parameter.get('built_in_test_interval', 20)
        self.early_stop_rounds_test = hyper_parameter.get('early_stop_rounds_test', None)
        self.early_stop_flag_test = False
        self.early_stop_scores_test = []
        return self.eval_score_type
    
    # 评估模型
    def cal_eval_score(self, y_true, y_infer, score_type=None):
        # 确认度量方式
        if not score_type:
            score_type = self.eval_score_type
        
        if score_type in ['precision', 'recall', 'f1_score']:
            # 若任务是2分类问题，则需要将2维的onehot形式预测值转换为1维
            y_true = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
            y_infer = tf.cast(tf.argmax(y_infer, axis=1), tf.int32)
            # 构造true矩阵和false矩阵
            ones = tf.ones_like(y_true)
            zeros = tf.zeros_like(y_true)
            # 计算TP/TN/FP/FN
            tp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(y_true, ones),
                        tf.equal(y_infer, ones)),
                    tf.float32))
            tn = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(y_true, zeros),
                            tf.equal(y_infer, zeros)),
                        tf.float32))
            fp = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(y_true, zeros),
                            tf.equal(y_infer, ones)),
                        tf.float32))
            fn = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(y_true, ones),
                            tf.equal(y_infer, zeros)),
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
        elif score_type in ['cross_entropy', 'cross_entropy_seq2seq', 'bilstm_crf_loss']:
            return self.loss_fun(y_infer)
        else:
            if score_type == 'mse':
                return tf.reduce_mean(tf.square(tf.subtract(y_true, y_infer)))
            elif score_type == 'mae':
                return tf.reduce_mean(tf.abs(tf.subtract(y_true, y_infer)))
            elif score_type == 'accuracy':
                y_infer_idx = tf.reshape(tf.argmax(y_infer, axis=1, output_type=tf.int32), [-1, 1])
                count_correct = tf.reduce_sum(tf.cast(tf.equal(y_infer_idx, y_true), tf.float32))
                count_all = tf.reduce_sum(tf.cast(tf.add(tf.multiply(y_true, 0), 1), tf.float32))
                return count_correct/count_all

    # early_stop判定
    def early_stop_judge(self, new_score, early_stop_scores, early_stop_rounds):
        early_stop_flag = False
        # 判断是否使用early_stop
        if early_stop_rounds is None:
            return False, None
        # 判断评估函数值是大点好，还是小点好
        if self.eval_score_type in ['mse', 'mae', 'bilstm_crf_loss', 'cross_entropy', 'cross_entropy_seq2seq']:
            minimize = True
        elif self.eval_score_type in ['f1_score', 'precision', 'recall', 'accuracy']:
            minimize = False
        else:
            minimize = True
            print('eval_score_type error!')
        # 判断early_stop
        if len(early_stop_scores) < early_stop_rounds:
            early_stop_scores = early_stop_scores + [new_score]
        else:
            if early_stop_rounds > 10:
                n_tmp = 5
                ths = (sum(sorted(early_stop_scores[-n_tmp-1:])[1:n_tmp])+new_score)/n_tmp
            else:
                ths = new_score
            if minimize is True:
                early_stop_flag = ([x for x in early_stop_scores if x > ths] == [])
            else:
                early_stop_flag = ([x for x in early_stop_scores if x < ths] == [])
            early_stop_scores = early_stop_scores + [new_score]
            early_stop_scores = early_stop_scores[-early_stop_rounds:]
        return early_stop_flag, early_stop_scores

    '''========================================================================'''
    '''===============================model part==============================='''
    '''========================================================================'''
    # todo 添加新模型时，需要在creat_mode和get_feed_dict上添加相应的分支
    # 构建模型
    def creat_model(self):
        # 记录各层神经网络的输出
        self.layer_output = []
        # 配置批处理样本个数
        self.batch_size = self.model_parameter.get('batch_size', 1024)
        # 获取keep_prob参数
        self.keep_prob = self.model_parameter.get('keep_prob', [1.0])
        # 确定模型当前处于训练模式还是预测模式
        self.train_or_infer_ph = tf.placeholder(tf.string, name='train_or_infer')
        # keep_probability
        self.keep_prob_ph = tf.placeholder('float32', name='keep_prob')

        if self.model_type == 'mlp':
            # 修正keep_prob格式
            if isinstance(self.keep_prob, float):
                self.keep_prob = [self.keep_prob]
            # 获取每层网络的神经元个数
            self.dim = [self.data_dim[0]]+list(self.model_parameter.get('dim', []))+[self.data_dim[1]]
            # 获取神经网络层数
            self.layer_num = len(self.dim)-1
            # 获取激活函数
            self.activation_fun = list(self.model_parameter.get('activation_fun', [tf.nn.relu]))
            if len(self.activation_fun) == 1:
                self.activation_fun = self.activation_fun*(self.layer_num-1)+[None]
            else:
                self.activation_fun[-1] = None
            # 配置placeholder
            self.x_ph = tf.placeholder('float32', [None, self.data_dim[0]], name='x')
            self.y_ph = tf.placeholder('float32', [None, self.data_dim[1]], name='y')
            self.keep_prob_ph = tf.placeholder('float32', [self.layer_num], name='keep_prob')
            # 构建模型
            self.layer_output = nn_model.mlp(self.x_ph, self.keep_prob_ph,
                                             self.activation_fun, self.layer_num, self.dim)
        elif self.model_type == 'rnn_nlp':
            self.word2vec = self.model_parameter.get('word2vec')
            self.dim_rnn = self.model_parameter.get('dim_rnn', 128)
            self.layer_num = 1
            # 配置placeholder
            self.x_ph = tf.placeholder('int32', [None, self.data_dim[0]], name='x')
            self.y_ph = tf.placeholder('float32', [None, self.data_dim[1]], name='y')
            # 构建模型
            self.layer_output = nn_model.rnn_nlp(self.x_ph, self.keep_prob_ph,
                                                 self.word2vec, self.dim_rnn, self.data_dim[1])
        elif self.model_type == 'bilstm_crf':
            self.word_embd_pretrain = self.model_parameter.get('word_embd_pretrain')
            self.dim_rnn = self.model_parameter.get('dim_rnn', 200)
            self.label_num = self.model_parameter.get('label_num', None)
            self.vocab_num = self.model_parameter.get('vocab_num', None)
            self.word_embd_dim = self.model_parameter.get('word_embd_dim', None)
            self.layer_num = 1
            # 配置placeholder
            self.x_ph = tf.placeholder('int32', [None, None], name='x')
            self.y_ph = tf.placeholder('int32', [None, None], name='y')
            self.seq_len_max_ph = tf.placeholder('int32', name='seq_len_max')
            # 模型入参汇总
            self.params_inputs = [self.x_ph, self.keep_prob_ph,
                                  self.dim_rnn, self.label_num, self.seq_len_max_ph,
                                  self.word_embd_pretrain, self.vocab_num,
                                  self.word_embd_dim]
            # 构建模型
            self.layer_output, self.seq_len, self.transition_score_matrix = nn_model.bilstm_crf(
                self.train_or_infer_ph, self.x_ph, self.keep_prob_ph,
                self.dim_rnn, self.label_num, self.seq_len_max_ph,
                self.word_embd_pretrain, self.vocab_num,
                self.word_embd_dim
            )
        elif self.model_type == 'seq2seq':
            self.word_embd_dim = self.model_parameter.get('word_embd_dim', 300)
            self.dim_rnn = self.model_parameter.get('dim_rnn', 300)
            self.use_same_word_embd = self.model_parameter.get('use_same_word_embd', True)
            self.encoder_vocab_size = self.model_parameter.get('encoder_vocab_size', 0)
            self.encoder_word_embd_pretrain = self.model_parameter.get('encoder_word_embd_pretrain', tf.zeros([self.encoder_vocab_size, self.word_embd_dim], tf.int32))
            self.decoder_vocab_size = self.model_parameter.get('decoder_vocab_size', 0)
            self.decoder_word_embd_pretrain = self.model_parameter.get('decoder_word_embd_pretrain', tf.zeros([self.decoder_vocab_size, self.word_embd_dim], tf.int32))
            self.target_seq_len_max = self.model_parameter.get('target_seq_len_max', 50)
            # 配置placeholder
            self.x_ph = tf.placeholder('int32', [None, self.data_dim[0]], name='x')
            self.y_ph = tf.placeholder('int32', [None, self.data_dim[3]], name='y')
            self.batch_size_ph = tf.placeholder('int32', name='batch_size')
            self.x_extended_ph = tf.placeholder('int32', [None, self.data_dim[0]], name='x_extended')
            self.y_extended_ph = tf.placeholder('int32', [None, self.data_dim[3]], name='y_extended')
            self.vocab_size_extend_ph = tf.placeholder('int32', name='vocab_size_extend')
            # 模型入参出参映射
            self.inputs_map = {
                'train_or_infer': self.train_or_infer_ph, 'x_id': self.x_ph, 'y_id': self.y_ph,
                'keep_prob': self.keep_prob_ph, 'batch_size': self.batch_size_ph,
                'x_id_extended': self.x_extended_ph, 'y_id_extended': self.y_extended_ph,
                'vocab_size_extend': self.vocab_size_extend_ph,
                'word_embd_dim': tf.constant(self.word_embd_dim), 'dim_rnn': tf.constant(self.dim_rnn),
                'use_same_word_embd': tf.constant(self.use_same_word_embd),
                'encoder_word_embd_pretrain': self.encoder_word_embd_pretrain,
                'encoder_vocab_size': tf.constant(self.encoder_vocab_size),
                'decoder_word_embd_pretrain': self.decoder_word_embd_pretrain,
                'decoder_vocab_size': tf.constant(self.decoder_vocab_size),
                'target_seq_len_max': tf.constant(self.target_seq_len_max)
            }
            self.outputs_map = {
                'output': tf.placeholder('float32', [None, self.target_seq_len_max, None], name='output')
            }
            # 构建模型
            # self.encoder, self.decoder = nn_model.seq2seq(
            output = nn_model.seq2seq(
                self.train_or_infer_ph, self.x_ph, self.y_ph, self.keep_prob_ph, self.batch_size_ph,
                self.x_extended_ph, self.y_extended_ph, self.vocab_size_extend_ph,
                self.word_embd_dim, self.dim_rnn, self.use_same_word_embd,
                self.encoder_word_embd_pretrain, self.encoder_vocab_size,
                self.decoder_word_embd_pretrain, self.decoder_vocab_size,
                self.target_seq_len_max)
            # self.layer_output = self.encoder + self.decoder
            return output
        elif self.model_type == 'fm':
            # 确定隐层向量维度
            self.dim_lv = self.model_parameter.get('lantent_vector_dim', self.data[0].shape[1])
            # 配置placeholder
            self.x_ph = tf.placeholder('float32', [None, self.data_dim[0]], name='x')
            self.y_ph = tf.placeholder('float32', [None, self.data_dim[1]], name='y')
            # 构建模型
            self.layer_output = nn_model.fm(self.x_ph, self.data_dim[0], self.data_dim[1], self.dim_lv)
        else:
            print('model_type error!')
            return
        # 返回神经网络最后一层的输出值
        return self.layer_output[-1]

    # 获取feed参数
    def get_feed_dict(self, data, train_or_infer):
        feed_dict = {}
        # 1.标记当前模型是训练模式还是预测模式。2.配置keep_prob。
        if 'train' == train_or_infer:
            feed_dict[self.train_or_infer_ph] = 'train'
            feed_dict[self.keep_prob_ph] = self.keep_prob
        else:
            feed_dict[self.train_or_infer_ph] = 'infer'
            feed_dict[self.keep_prob_ph] = 1.0

        if self.model_type == 'test':
            pass
        elif self.model_type == 'mlp':
            feed_dict[self.x_ph] = data[0]
            feed_dict[self.y_ph] = data[1]
            if 'infer' == train_or_infer:
                feed_dict[self.keep_prob_ph] = [1.0] * self.layer_num
            else:
                if len(self.keep_prob) == 1:
                    feed_dict[self.keep_prob_ph] = self.keep_prob * (self.layer_num - 1) + [1.0]
                else:
                    feed_dict[self.keep_prob_ph] = self.keep_prob[:-1] + [1.0]
        elif self.model_type == 'rnn_nlp':
            feed_dict[self.x_ph] = data[0]
            feed_dict[self.y_ph] = data[1]
        elif self.model_type == 'seq2seq':
            # 传入batch_size
            feed_dict[self.batch_size_ph] = data[0].shape[0]
            # 传入数据源序列数据
            feed_dict[self.x_ph] = data[0]
            feed_dict[self.x_extended_ph] = data[1]
            # 传入扩展词表长度
            feed_dict[self.vocab_size_extend_ph] = max([len(vocab[0]) for vocab in data[2]])
            # 尝试传入目标序列数据
            try:
                feed_dict[self.y_ph] = data[3]
                feed_dict[self.y_extended_ph] = data[4]
            except:
                pass
        elif self.model_type == 'bilstm_crf':
            feed_dict[self.x_ph] = data[0]
            feed_dict[self.seq_len_max_ph] = data[0].shape[1]
            try:
                feed_dict[self.y_ph] = data[1]
            except:
                pass
        return feed_dict

    '''========================================================================'''
    '''===============================other part==============================='''
    '''========================================================================'''
    # 导出用于tf-serving的模型架构的ProtocolBuffer文件
    def export_model(self):
        tf.reset_default_graph()
        self.creat_model()
        model_path = tf.train.latest_checkpoint(self.path_data+'model_save\\')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 读取模型参数
            saver.restore(sess, model_path)
            # 创建模型输出builder
            path_model_export = self.path_data + 'tf_serving_model'
            if os.path.exists(path_model_export):
                # os.remove(path_model_export)
                os.removedirs(path_model_export)
            builder = tf.saved_model.builder.SavedModelBuilder(path_model_export)
            # 定义输入、输出、方法名
            inputs = {key: tf.saved_model.utils.build_tensor_info(value) for key, value in self.inputs_map.items()}
            outputs = {key: tf.saved_model.utils.build_tensor_info(value) for key, value in self.outputs_map.items()}
            model_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            # 张量图通过sess输入，张量通过model_signature_def_map输入
            builder.add_meta_graph_and_variables(
                sess=sess,
                # tags可以定义为任意字符串
                tags=[tf.saved_model.tag_constants.SERVING],
                # todo judge how can clear_devices work
                clear_devices=True,
                signature_def_map={
                    # todo choice which one? 好像可以任意定义，也有看到定义为predict的
                    # "predict_image": model_signature
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature
                }
            )
            builder.save(as_text=True)

        if False:
            # 转化tensor到模型支持的格式tensor_info，下面的reshape是因为只想输出单个结果数组，否则是二维的
            tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
            tensor_info_pro = tf.saved_model.utils.build_tensor_info(tf.reshape(values, [1]))
            tensor_info_classify = tf.saved_model.utils.build_tensor_info(tf.reshape(indices, [1]))
            # 定义方法名和输入输出
            signature_def_map = {
                "predict_image": tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={"image": tensor_info_x},
                    outputs={
                        "pro": tensor_info_pro,
                        "classify": tensor_info_classify
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )}
            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map=signature_def_map)
            builder.save()
        if False:
            from tensorflow.python.saved_model import (
                signature_constants, signature_def_utils, tag_constants, utils)
            model_path = "model"
            model_version = 1
            model_signature = signature_def_utils.build_signature_def(
                inputs={
                    "keys": utils.build_tensor_info(keys_placeholder),
                    "features": utils.build_tensor_info(inference_features)
                },
                outputs={
                    "keys": utils.build_tensor_info(keys_identity),
                    "prediction": utils.build_tensor_info(inference_op),
                    "softmax": utils.build_tensor_info(inference_softmax),
                },
                method_name=signature_constants.PREDICT_METHOD_NAME)
            export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder = saved_model_builder.SavedModelBuilder(export_path)
            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                clear_devices=True,
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        model_signature,
                },
                legacy_init_op=legacy_init_op)
            builder.save()
        return

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
            # 判断待存储参数的文件夹是否存在，若不存在，则新建一个。
            if not os.path.exists(self.path_data + 'model_paras\\'):
                os.makedirs(self.path_data + 'model_paras\\')
            # 遍历模型参数并存储到csv文件里
            for var in tf.global_variables():
                print(var)
                var_value = sess.run(var)
                var_name = var.name.replace(':', '_').replace(r'/', '_')
                dataframe(var_value).to_csv(self.path_data+u'model_paras\\{}.csv'.format(var_name), sep=',', encoding='utf_8_sig')
        #     w_1=tf.get_default_graph().get_tensor_by_name('w:0')
        #     w_1=dataframe(w_1.eval())
        #     w_1.to_csv('.\para_output'+u'\\w_1'+u'.csv',sep=',',encoding='utf_8_sig')
        return
   



if __name__ == '__main__':
    logging.warning('end')



