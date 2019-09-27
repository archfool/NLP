import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import nn_lib



# FM模型
def fm(x, dim_x, dim_y, dim_lv):
    with tf.variable_scope('fm'):
        # 构建神经网络
        layer_output = [x]
        # 线性项部分
        with tf.variable_scope('linear_part'):
            # 定义权重w
            weight = tf.Variable(tf.truncated_normal([dim_x, dim_y], stddev=0.1, dtype=tf.float32), name='weight')
            # 定义偏置b
            bias = tf.Variable(tf.truncated_normal([dim_y], stddev=0.1, dtype=tf.float32), name='bias')
            # 输出线性项
            linear_part = tf.add(tf.matmul(layer_output[0], weight), bias)
        # 交叉项部分
        with tf.variable_scope('interaction_part'):
            # 定义潜在向量
            lantent_vector = tf.Variable(tf.truncated_normal([dim_x, dim_lv, dim_y], stddev=0.1, dtype=tf.float32),
                                         name='lantent_vector')
            # 输出交叉项
            tmp = [None]*dim_y
            for i in range(dim_y):
                tmp[i] = tf.multiply(0.5,
                                     tf.reduce_sum(
                                         tf.subtract(
                                             tf.pow(tf.matmul(layer_output[0], lantent_vector[:, :, i]), 2),
                                             tf.matmul(tf.pow(layer_output[0], 2), tf.pow(lantent_vector[:, :, i], 2))),
                                         axis=1, keepdims=True))
            interaction_part = tf.concat([x for x in tmp], axis=1)
        # 合并线性项和交叉项
        layer_output.append(tf.add(linear_part, interaction_part))
        # tf.nn.softmax(layer_output[-1])
    return layer_output



