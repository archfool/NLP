# -*- coding: utf-8 -*-
"""
Created on Fri Aug  23 16:48:57 2019

@author: ruan
"""

import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import nn_lib


def bilstm_crf(x, keep_prob, dim_rnn, label_num, word_embd_pretrain=None, vocab_size=None, word_embd_dim=None):
    with tf.variable_scope('bilstm_layer'):
        # layer[0] 输入数据，已onehot，未embedding
        # x:[batch_size,step_len]
        layer_output = [x]
        # batch_len = layer_output[0].shape[0].value
        step_len = layer_output[0].shape[1].value
        seq_len = tf.cast(tf.reduce_sum(tf.sign(layer_output[0]), axis=1), tf.int32)
        # layer[1] 进行embedding
        # word_embd:[batch_size,step_len,embd_dim]
        if isinstance(word_embd_pretrain, np.ndarray):
            word_embd = tf.get_variable(name='word_embd', trainable=True,
                                        initializer=word_embd_pretrain)
            vocab_size = word_embd.shape[0].value
            word_embd_dim = word_embd.shape[1].value
        else:
            # 在word_embd_pretrain==None时，需要根据词表大小vocab_size和词向量维度word_embd_dim，建立嵌入词向量word2vec
            word_embd = tf.get_variable(
                name='word_embd', trainable=True,
                initializer=tf.random_uniform((vocab_size, word_embd_dim), minval=-1, maxval=1, dtype=tf.float32)
            )
        embedding_w2v = tf.nn.embedding_lookup(word_embd, layer_output[0])
        layer_output.append(embedding_w2v)
        # layer[2] 生成BiLSTM实体，并进行dropout
        # lstm_layer:[batch_size,step_len,dim_rnn*2]
        lstm_cell_fw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_rnn, state_is_tuple=True)
        lstm_cell_bw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_rnn, state_is_tuple=True)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_fw_raw, output_keep_prob=keep_prob)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_bw_raw, output_keep_prob=keep_prob)
        lstm_layer_raw, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, layer_output[1],
                                                            sequence_length=seq_len, dtype=tf.float32)
#        print("\n",type(states),states.__len__(),"\n")
#        print("\n",type(states[0]),states[0].__len__(),"\n")
#        print("\n",type(states[0][0]),states[0][0].shape,states[0][1].shape,"\n")
        lstm_layer = tf.concat(lstm_layer_raw, axis=2)
        layer_output.append(lstm_layer)
        # layer[3] 全连接层
        fc_hidden_raw = tf.reshape(layer_output[2], shape=[-1, dim_rnn*2])
        weight = tf.get_variable(
            name='weight', trainable=True,
            initializer=tf.truncated_normal([dim_rnn*2, label_num], stddev=math.sqrt(6/(dim_rnn*2+label_num)), dtype=tf.float32)
        )
        bias = tf.get_variable(
            name='bias', trainable=True,
            initializer=tf.truncated_normal([label_num], stddev=0.1, dtype=tf.float32)
        )
        fc_hidden = tf.matmul(fc_hidden_raw, weight) + bias
        layer_output.append(tf.reshape(fc_hidden, [-1, step_len, label_num]))
    with tf.variable_scope('crf_layer'):
        transition_score_matrix = tf.get_variable(name='transition_score_matrix',
                                                  trainable=True, shape=[label_num, label_num])
    return layer_output, seq_len, transition_score_matrix


