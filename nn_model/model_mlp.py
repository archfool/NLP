import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import nn_lib

# mlp模型
def mlp(x, keep_prob, activation_fun, layer_num, dim):
    # 构建神经网络
    layer_output = [x]
    for i in range(1, layer_num+1):
        layer_output.append(add_mlp_layer(layer_name='layer_{}'.format(i),
                                          inputs=layer_output[i-1],
                                          dim_in=dim[i-1],
                                          dim_out=dim[i],
                                          use_bias=True,
                                          activation_fun=activation_fun[i-1],
                                          keep_prob=keep_prob[i-1]))
    return layer_output


# mlp加一层
def add_mlp_layer(layer_name, inputs, dim_in, dim_out, use_bias=True, activation_fun=None, keep_prob=1.0):
    with tf.variable_scope(layer_name):
        # 定义权重w
        weight = tf.get_variable(
            name='weight', trainable=True,
            initializer=tf.truncated_normal([dim_in, dim_out], stddev=math.sqrt(6/(dim_in+dim_out)), dtype=tf.float32))
        if use_bias is True:
            # 定义偏置b
            bias = tf.get_variable(name='bias', trainable=True,
                                   initializer=tf.truncated_normal([dim_out], stddev=0.1, dtype=tf.float32))
            # 全连接层，计算x*w+b
            fc = tf.add(tf.matmul(tf.cast(inputs, tf.float32), weight), bias)
        else:
            fc = tf.matmul(tf.cast(inputs, tf.float32), weight)
        # 丢弃层
        dropout = tf.nn.dropout(fc, keep_prob)
        # 使用激活函数
        if activation_fun:
            activated = activation_fun(dropout)
        else:
            activated = dropout
        # 输出结果
        layer_output = activated
    return layer_output



