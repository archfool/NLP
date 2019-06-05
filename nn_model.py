# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:48:57 2019

@author: ruan
"""

import tensorflow as tf
import numpy as np
import math
from nn_model_crf import crf_log_likelihood,viterbi_decode

#mlp模型
def mlp(x,keep_prob,activation_fun,layer_num,dim):
    #构建神经网络
    layer_output = [x]
    for i in range(1,layer_num+1):
        layer_output.append(add_mlp_layer(layer_name='layer_{}'.format(i),
                                          inputs=layer_output[i-1],
                                          dim_in=dim[i-1],
                                          dim_out=dim[i],
                                          activation_fun=activation_fun[i-1],
                                          keep_prob=keep_prob[i-1]))
    return layer_output

#mlp加一层
def add_mlp_layer(layer_name,inputs,dim_in,dim_out,activation_fun=None,keep_prob=1.0):
    with tf.variable_scope(layer_name):
        #定义权重w
        weight = tf.get_variable(name='weight',trainable=True,\
            initializer=tf.truncated_normal([dim_in,dim_out],stddev=math.sqrt(6/(dim_in+dim_out)),dtype=tf.float32))
        #定义偏置b
        bias = tf.get_variable(name='bias',trainable=True,\
            initializer=tf.truncated_normal([dim_out],stddev=0.1,dtype=tf.float32))
        #全连接层，计算x*w+b
        fc = tf.add(tf.matmul(tf.cast(inputs,tf.float32),weight),bias)
        #丢弃层
        dropout = tf.nn.dropout(fc,keep_prob)
        #使用激活函数
        if activation_fun==None:
            activated = dropout
        else:
            activated = activation_fun(dropout)
        #输出结果
        layer_output = activated
    return layer_output

#带有embedding的RNN模型
def rnn_nlp(x,keep_prob,word2vec,dim_lstm,dim_y):
    with tf.variable_scope('rnn_nlp'):
        #layer[0] 输入数据，未embedding前
        #x:[batch,time]
        layer_output = [x]
        #layer[1] 进行embedding
        #embedding_w2v:[batch,time,embd]
        embedding_w2v = tf.nn.embedding_lookup(word2vec, layer_output[0])
        layer_output.append(embedding_w2v)
        #layer[2] 生成lstm实体，并进行dropout
        lstm_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm,state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_raw, output_keep_prob=keep_prob)
        lstm_layer = tf.nn.dynamic_rnn(lstm_cell,layer_output[1],dtype=tf.float32)[0]
        layer_output.append(lstm_layer)
        #layer[3]
        #transpose:[time,batch,embd]
        transpose = tf.transpose(layer_output[2], [1, 0, 2])
        #transpose_gather:[batch,embd]提取lstm最后一个时刻的output数据
        transpose_gather = tf.gather(transpose, int(transpose.get_shape()[0].value)-1)
        layer_output.append(transpose_gather)
        #layer[4] 全连接层
        weight = tf.get_variable(name='weight',trainable=True,\
                    initializer=tf.truncated_normal([dim_lstm, dim_y],stddev=math.sqrt(6/(dim_lstm+dim_y)),dtype=tf.float32))
        bias = tf.get_variable(name='bias',trainable=True,\
                    initializer=tf.truncated_normal([dim_y],stddev=0.1,dtype=tf.float32))
        layer_output.append((tf.matmul(layer_output[3], weight) + bias))
    return layer_output

#BiLSTM_CRF模型
def bilstm_crf(x,keep_prob,dim_lstm,label_num,word2vec=None,vocab_num=None,w2v_dim=None):
    with tf.variable_scope('bilstm_layer'):
        #layer[0] 输入数据，未embedding
        #x:[batch,time]
        layer_output = [x]
        #batch_len = layer_output[0].get_shape()[0].value
        time_len = layer_output[0].get_shape()[1].value
        seq_len = tf.cast(tf.reduce_sum(tf.sign(layer_output[0]), axis=1),tf.int32)
        #layer[1] 进行embedding
        #embedding_w2v:[batch,time,embd]
        if isinstance(word2vec,np.ndarray):
            word2vec_new = tf.get_variable(name='word2vec',trainable=True,\
                                initializer=word2vec)
        else:
            #在word2vec==None时，需要根据词表大小vocab_num和词向量维度w2v_dim，建立嵌入词向量word2vec
            word2vec_new = tf.get_variable(name='word2vec',trainable=True,\
                                initializer=tf.random_uniform((vocab_num,w2v_dim),-1,1,dtype=tf.float32))
        embedding_w2v = tf.nn.embedding_lookup(word2vec_new, layer_output[0])
        layer_output.append(embedding_w2v)
        #layer[2] 生成lstm实体，并进行dropout
        #lstm_layer:[batch,time,dim_lstm*2]
        lstm_cell_fw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm,state_is_tuple=True)
        lstm_cell_bw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm,state_is_tuple=True)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_fw_raw, output_keep_prob=keep_prob)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_bw_raw, output_keep_prob=keep_prob)
        lstm_layer_raw = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,lstm_cell_bw,\
                            layer_output[1],sequence_length=seq_len,dtype=tf.float32)[0]
        lstm_layer = tf.concat(lstm_layer_raw, axis=2)
        layer_output.append(lstm_layer)
        #layer[3] 全连接层
        fc_hidden_raw = tf.reshape(layer_output[2], shape=[-1, dim_lstm*2])
        weight = tf.get_variable(name='weight',trainable=True,\
                    initializer=tf.truncated_normal([dim_lstm*2, label_num],stddev=math.sqrt(6/(dim_lstm*2+label_num)),dtype=tf.float32))
        bias = tf.get_variable(name='bias',trainable=True,\
                    initializer=tf.truncated_normal([label_num],stddev=0.1,dtype=tf.float32))
        fc_hidden = tf.matmul(fc_hidden_raw, weight) + bias
        layer_output.append(tf.reshape(fc_hidden, [-1,time_len,label_num]))
    with tf.variable_scope('crf_layer'):
        transition_score_matrix = tf.get_variable(name='transition_score_matrix',\
                                    trainable=True,shape=[label_num,label_num])
    return layer_output,seq_len,transition_score_matrix

'''
def bilstm_crf_loss_fun(bilstm_output,y,seq_len,label_num):
    with tf.variable_scope('crf_layer'):
        #转移概率transition_probability和发射概率emission_probability
        seq_score = crf_sequence_score(inputs=bilstm_output,)
        
        
        
#  sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
#                                       transition_params)
#  log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)
#  log_likelihood = sequence_scores - log_norm
#  return log_likelihood, transition_params
        log_likelihood, transition_score_matrix = tf.contrib.crf.crf_log_likelihood(inputs=layer_output[3],
#                                                               tag_indices=tf.cast(y,tf.int32),
                                                               tag_indices=y,
                                                               sequence_lengths=seq_len)
        #loss = -tf.reduce_mean(log_likelihood)
        layer_output.append(log_likelihood)
    return layer_output
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
'''

'''        
class CrfForwardRnnCell(tf.nn.rnn_cell.RNNCell):
  """Computes the alpha values in a linear-chain CRF.
  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """
  def __init__(self, transition_score_matrix):
    """Initialize the CrfForwardRnnCell.
    Args:
      transition_score_matrix: A [label_num, label_num] matrix of binary potentials.
          This matrix is expanded into a [1, label_num, label_num] in preparation
          for the broadcast summation occurring within the cell.
    """
    self._transition_score_matrix = tf.expand_dims(transition_score_matrix, 0)
    self._label_num = transition_score_matrix.get_shape()[0].value

  @property
  def state_size(self):
    return self._label_num

  @property
  def output_size(self):
    return self._label_num

  def __call__(self, inputs, state, scope=None):
    """Build the CrfForwardRnnCell.
    Args:
      inputs: A [batch_size, label_num] matrix of unary potentials.
      state: A [batch_size, label_num] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.
    Returns:
      new_alphas, new_alphas: A pair of [batch_size, label_num] matrices
          values containing the new alpha values.
    """
    emission_score = inputs
    old_alphas = state
    old_alphas = tf.expand_dims(old_alphas, 2)

    # This addition op broadcasts self._transition_score_matrix along the zeroth
    # dimension and old_alphas along the second dimension. This performs the
    # multiplication of previous alpha values and the current binary potentials
    # in log space.
    transition_scores = old_alphas + self._transition_score_matrix
    new_alphas = emission_score + tf.reduce_logsumexp(transition_scores, [1])

    # Both the state and the output of this RNN cell contain the alphas values.
    # The output value is currently unused and simply satisfies the RNN API.
    # This could be useful in the future if we need to compute marginal
    # probabilities, which would require the accumulated alpha values at every
    # time step.
    return new_alphas, new_alphas
'''

#FM模型
def fm(x,dim_x,dim_y,dim_lv):
    with tf.variable_scope('fm'):
        #构建神经网络
        layer_output = [x]
        #线性项部分
        with tf.variable_scope('linear_part'):
            #定义权重w
            weight = tf.Variable(tf.truncated_normal([dim_x,dim_y],stddev=0.1,dtype=tf.float32),name='weight')
            #定义偏置b
            bias = tf.Variable(tf.truncated_normal([dim_y],stddev=0.1,dtype=tf.float32),name='bias')
            #输出线性项
            linear_part = tf.add(tf.matmul(layer_output[0],weight),bias)
        #交叉项部分
        with tf.variable_scope('interaction_part'):
            #定义潜在向量
            lantent_vector = tf.Variable(tf.truncated_normal([dim_x,dim_lv,dim_y],stddev=0.1,dtype=tf.float32),name='lantent_vector')
            #输出交叉项
            tmp = [None]*dim_y
            for i in range(dim_y):
                tmp[i] = tf.multiply(0.5,\
                             tf.reduce_sum(\
                                  tf.subtract(\
                                      tf.pow(tf.matmul(layer_output[0],lantent_vector[:,:,i]),2),\
                                      tf.matmul(tf.pow(layer_output[0],2),tf.pow(lantent_vector[:,:,i],2))),\
                                  axis=1,keepdims=True))
            interaction_part = tf.concat([x for x in tmp],axis=1)
        #合并线性项和交叉项
        layer_output.append(tf.add(linear_part,interaction_part))
        #tf.nn.softmax(layer_output[-1])
    return layer_output

def test(self):
    # 定义两层双向LSTM的模型结构
    with tf.name_scope("Bi-LSTM"):
        for idx, hiddenSize in enumerate(self.model.hiddenSizes):
            with tf.name_scope("Bi-LSTM" + str(idx)):
                # 定义前向LSTM结构
                lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                             output_keep_prob=self.dropoutKeepProb)
                # 定义反向LSTM结构
                lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                             output_keep_prob=self.dropoutKeepProb)
                # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                              self.embeddedWords, dtype=tf.float32,
                                                                              scope="bi-lstm" + str(idx))
                # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                self.embeddedWords = tf.concat(outputs_, 2)
    # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
    outputs = tf.split(self.embeddedWords, 2, -1)
    # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
    with tf.name_scope("Attention"):
        H = outputs[0] + outputs[1]
        # 得到Attention的输出
        output = self._attention(H)
        outputSize = self.model.hiddenSizes[-1]
    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.model.hiddenSizes[-1]
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.sequenceLength])
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.sequenceLength, 1]))
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        sentenceRepren = tf.tanh(sequeezeR)
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        return output        
