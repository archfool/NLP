import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import nn_lib


# 带有embedding的RNN模型
def rnn_nlp(x, keep_prob, word_embd_pretrain, dim_rnn, dim_y):
    with tf.variable_scope('rnn_nlp'):
        # layer[0] 输入数据，未embedding前
        # x:[batch,step]
        layer_output = [x]
        # layer[1] 进行embedding
        # embedding_w2v:[batch,step,embd]
        embedding_w2v = tf.nn.embedding_lookup(word_embd_pretrain, layer_output[0])
        layer_output.append(embedding_w2v)
        # layer[2] 生成lstm实体，并进行dropout
        lstm_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_rnn, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_raw, output_keep_prob=keep_prob)
        lstm_layer = tf.nn.dynamic_rnn(lstm_cell, layer_output[1], dtype=tf.float32)[0]
        layer_output.append(lstm_layer)
        # layer[3]
        # transpose:[step,batch,embd]
        transpose = tf.transpose(layer_output[2], [1, 0, 2])
        # transpose_gather:[batch,embd]提取lstm最后一个时刻的output数据
        transpose_gather = tf.gather(transpose, int(transpose.shape[0].value)-1)
        layer_output.append(transpose_gather)
        # layer[4] 全连接层
        weight = tf.get_variable(name='weight', trainable=True,
                    initializer=tf.truncated_normal([dim_rnn, dim_y], stddev=math.sqrt(6/(dim_rnn+dim_y)), dtype=tf.float32))
        bias = tf.get_variable(name='bias', trainable=True,
                               initializer=tf.truncated_normal([dim_y], stddev=0.1, dtype=tf.float32))
        layer_output.append((tf.matmul(layer_output[3], weight) + bias))
    return layer_output



