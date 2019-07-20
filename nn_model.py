# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:48:57 2019

@author: ruan
"""

import tensorflow as tf
import numpy as np
import math
from tensorflow.python.layers.core import Dense
import nn_lib


"""===================================================mlp模型==================================================="""


# mlp模型
def mlp(x, keep_prob, activation_fun, layer_num, dim):
    # 构建神经网络
    layer_output = [x]
    for i in range(1, layer_num+1):
        layer_output.append(add_mlp_layer(layer_name='layer_{}'.format(i),
                                          inputs=layer_output[i-1],
                                          dim_in=dim[i-1],
                                          dim_out=dim[i],
                                          activation_fun=activation_fun[i-1],
                                          keep_prob=keep_prob[i-1]))
    return layer_output


# mlp加一层
def add_mlp_layer(layer_name, inputs, dim_in, dim_out, activation_fun=None, keep_prob=1.0):
    with tf.variable_scope(layer_name):
        # 定义权重w
        weight = tf.get_variable(
            name='weight', trainable=True,
            initializer=tf.truncated_normal([dim_in, dim_out], stddev=math.sqrt(6/(dim_in+dim_out)), dtype=tf.float32))
        # 定义偏置b
        bias = tf.get_variable(name='bias', trainable=True,
                               initializer=tf.truncated_normal([dim_out], stddev=0.1, dtype=tf.float32))
        # 全连接层，计算x*w+b
        fc = tf.add(tf.matmul(tf.cast(inputs, tf.float32), weight), bias)
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


"""===================================================rnn模型==================================================="""


# 带有embedding的RNN模型
def rnn_nlp(x, keep_prob, word_embd_pretrain, dim_lstm, dim_y):
    with tf.variable_scope('rnn_nlp'):
        # layer[0] 输入数据，未embedding前
        # x:[batch,step]
        layer_output = [x]
        # layer[1] 进行embedding
        # embedding_w2v:[batch,step,embd]
        embedding_w2v = tf.nn.embedding_lookup(word_embd_pretrain, layer_output[0])
        layer_output.append(embedding_w2v)
        # layer[2] 生成lstm实体，并进行dropout
        lstm_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_raw, output_keep_prob=keep_prob)
        lstm_layer = tf.nn.dynamic_rnn(lstm_cell, layer_output[1], dtype=tf.float32)[0]
        layer_output.append(lstm_layer)
        # layer[3]
        # transpose:[step,batch,embd]
        transpose = tf.transpose(layer_output[2], [1, 0, 2])
        # transpose_gather:[batch,embd]提取lstm最后一个时刻的output数据
        transpose_gather = tf.gather(transpose, int(transpose.get_shape()[0].value)-1)
        layer_output.append(transpose_gather)
        # layer[4] 全连接层
        weight = tf.get_variable(name='weight', trainable=True,
                    initializer=tf.truncated_normal([dim_lstm, dim_y], stddev=math.sqrt(6/(dim_lstm+dim_y)), dtype=tf.float32))
        bias = tf.get_variable(name='bias', trainable=True,
                               initializer=tf.truncated_normal([dim_y], stddev=0.1, dtype=tf.float32))
        layer_output.append((tf.matmul(layer_output[3], weight) + bias))
    return layer_output


"""===================================================bilstm_crf模型==================================================="""


# BiLSTM_CRF模型
def bilstm_crf(x, keep_prob, dim_lstm, label_num, word_embd_pretrain=None, vocab_size=None, word_embd_dim=None):
    with tf.variable_scope('bilstm_layer'):
        # layer[0] 输入数据，已onehot，未embedding
        # x:[batch_size,step_len]
        layer_output = [x]
        # batch_len = layer_output[0].get_shape()[0].value
        step_len = layer_output[0].get_shape()[1].value
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
        # lstm_layer:[batch_size,step_len,dim_lstm*2]
        lstm_cell_fw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
        lstm_cell_bw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
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
        fc_hidden_raw = tf.reshape(layer_output[2], shape=[-1, dim_lstm*2])
        weight = tf.get_variable(
            name='weight', trainable=True,
            initializer=tf.truncated_normal([dim_lstm*2, label_num], stddev=math.sqrt(6/(dim_lstm*2+label_num)), dtype=tf.float32)
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


"""===================================================seq2seq模型==================================================="""


def seq2seq(x, y, keep_prob, is_predict, dim_lstm, word_embd_dim,
            encoder_word_embd_pretrain=None, encoder_vocab_size=None,
            decoder_word_embd_pretrain=None, decoder_vocab_size=None):
    with tf.variable_scope('encoder'):
        # encoder[0] 输入数据，已onehot，未embedding
        # x:[batch_size,step_len]
        # TODO: layer_output = [tf.convert_to_tensor(x)]
        encoder = [x]
        # batch_size = encoder[0].get_shape()[0].value
        batch_size = 1024
        print(type(encoder[0]))
        print(encoder[0].get_shape())
        encoder_seq_max_len = encoder[0].get_shape()[1].value
        encoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(encoder[0]), axis=1), tf.int32)
        # encoder[1] 对源序列数据进行embedding
        encoder_word_embd, encoder_vocab_size, word_embd_dim \
            = get_word_embd(encoder_word_embd_pretrain, encoder_vocab_size, word_embd_dim, name='encoder_word_embd')
        encoder_embedding_w2v = tf.nn.embedding_lookup(encoder_word_embd, encoder[0])
        encoder.append(encoder_embedding_w2v)
        # encoder[2] 构建encoder模型，并使用dynamic_rnn方法
        # lstm_layer:[batch_size,step_len,dim_lstm*2]
        lstm_cell_fw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
        lstm_cell_bw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_fw_raw, output_keep_prob=keep_prob)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_bw_raw, output_keep_prob=keep_prob)
        encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                          cell_bw=lstm_cell_bw,
                                                                          inputs=encoder[1],
                                                                          sequence_length=encoder_seq_len,
                                                                          dtype=tf.float32)
        print(encoder_states.get_shape())
        memory = tf.concat(encoder_outputs, axis=2)
        encoder.append(memory)
        '''
        lstm_state_as_tensor_shape = [num_layers, 2, batch_size, hidden_size]
        initial_state = tf.zeros(lstm_state_as_tensor_shape)
        unstack_state = tf.unstack(initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(num_layers)])
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state_out = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=tuple_state)
        '''
    with tf.variable_scope('decoder'):
        # TODO: layer_output = [tf.convert_to_tensor(y)]
        # decoder[0] 目标序列数据
        decoder = [y]
        # 提取word_embd矩阵
        decoder_word_embd, decoder_vocab_size, word_embd_dim \
            = get_word_embd(decoder_word_embd_pretrain, decoder_vocab_size, word_embd_dim, name='decoder_word_embd')
        # decoder[1] 构建encoder模型，并使用dynamic_rnn方法
        # 构建decoder模型
        decoder_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(dim_lstm)
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(cell=decoder_cell_raw, output_keep_prob=keep_prob)
        output_layer = None
        # 封装copynet模块
        decoder_cell = nn_lib.CopyNetWrapper(cell=decoder_cell,
                                             encoder_states=memory,
                                             encoder_input_ids=encoder[0],
                                             vocab_size=encoder_vocab_size,
                                             gen_vocab_size=decoder_vocab_size,
                                             encoder_state_size=decoder_cell.output_size*2,
                                             initial_cell_state=None)
        decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)
        if False == is_predict:
            # 获取目标序列信息
            # TODO: layer_output = [tf.convert_to_tensor(x)]
            decoder_seq_max_len = decoder[0].get_shape()[1].value
            decoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(decoder[0]), axis=1), tf.int32)
            # 对目标序列数据进行embedding
            decoder_embedding_w2v = tf.nn.embedding_lookup(decoder_word_embd, decoder[0])
            # 使用dynamic_rnn方法
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_w2v, decoder_seq_len)
            decoder_help = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state)
            decoder_outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
            decoder.append(decoder_outputs.rnn_output)
            sample_id = decoder_outputs.sample_id
            logits = decoder_outputs.rnn_output
            # logits = output_layer(outputs.rnn_output)
            print("\n", type(decoder_outputs), decoder_outputs.__len__(), "\n")
            print("\n", type(decoder_outputs[0]), decoder_outputs[0].__len__(), "\n")
            print("\n", type(decoder_outputs[1]), decoder_outputs[1].__len__(), "\n")
        elif True == is_predict:
            # 使用dynamic_rnn方法
            beam_width = 3
            # sos_ids = tf.fill([batch_size], decoder_vocab['<SOS>'])
            # eos_id = decoder_vocab['<EOS>']
            # '<SOS>'的词表id为1，'<EOS>'的词表id为2。
            sos_ids = tf.fill([batch_size], tf.cast(1,tf.int32))
            eos_id = tf.cast(2,tf.int32)
            decoder_help = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                embedding=decoder_word_embd,
                                                                start_tokens=sos_ids,
                                                                end_token=eos_id,
                                                                initial_state=decoder_initial_state,
                                                                beam_width=beam_width,
                                                                output_layer=output_layer,
                                                                length_penalty_weight=0.0)
            decoder_outputs, decoder__state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder_help,
                                                                                   maximum_iterations=encoder_seq_max_len*2)
            decoder.append(decoder_outputs)
            if beam_width > 0:
                logits = tf.no_op()
                sample_id = decoder_outputs.predicted_ids
            else:
                logits = decoder_outputs.rnn_output
                sample_id = decoder_outputs.sample_id
        else:
            print("value of predict is error!!!")
    return encoder, decoder
#     with tf.variable_scope('decoder_old'):
#         """定义解码器的word_embedding"""
#         decoder_word_embd, decoder_vocab_size, word_embd_dim \
#             = get_word_embd(decoder_word_embd_pretrain, decoder_vocab_size, word_embd_dim, name='decoder_word_embd')
#         """layer[3]"""
#         # 注意力机制
#         attention_mechanism = nn_lib.LuongAttention(
#                                 num_units=dim_lstm*2,
#                                 memory=layer_output[2],
#                                 memory_sequence_length=encoder_seq_len)
#         # 定义decoder结构
#         decoder_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(dim_lstm)
#         decoder_cell = nn_lib.AttentionWrapper(
#                             cell=decoder_cell_raw,
#                             attention_mechanism=attention_mechanism,
#                             attention_layer_size=dim_lstm*2)
#         # helper和decoder实体
#         decoder_initial_state = decoder_cell.zero_state(batch_size,dtype=tf.float32)
#         projection_layer = Dense(decoder_vocab_size, use_bias=False)
#         if True==predict:
#             # 在词表中，<UNK>为0，<SOS>为1，<EOS>为2
#             helper = nn_lib.GreedyEmbeddingHelper(decoder_word_embd,\
#                                                   tf.fill([batch_size],1),\
#                                                   2)
#             decoder = nn_lib.BasicDecoder(cell=decoder_cell,\
#                                           helper=helper,\
#                                           initial_state=decoder_initial_state,
#                                           output_layer=projection_layer)
#             maximum_iterations = tf.round(encoder_seq_max_len * 2)
#             decoder_outputs,_,_ = nn_lib.dynamic_decode(decoder,
#                                     maximum_iterations=maximum_iterations)
#             translations = decoder_outputs.sample_id
#         elif False==predict:
# #            decoder_step_len = y.get_shape()[1].value
#             decoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(y), axis=1),tf.int32)
#             decoder_embedding_w2v = tf.nn.embedding_lookup(decoder_word_embd, y)
#             helper = nn_lib.CopyNetTrainingHelper(inputs=decoder_embedding_w2v,\
#                                                   encoder_inputs_ids=x,\
#                                                   sequence_length=decoder_seq_len)
#             config={'encoder_max_seq_len':encoder_seq_max_len,
#                     'vocab_size':encoder_vocab_size}
#             decoder = nn_lib.CopyNetDecoder(config=config,
#                                             cell=decoder_cell,
#                                             helper=helper,
#                                             initial_state=decoder_initial_state,
#                                             encoder_outputs=encoder_outputs,
#                                             output_layer=projection_layer)
#             decoder_outputs,_,_ = nn_lib.dynamic_decode(decoder)
#             logits = decoder_outputs.rnn_output
#         else:
#             print("value of predict is error!!!")


"""===================================================FM模型==================================================="""


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


"""===================================================test==================================================="""


# def test(self):
#     # 定义两层双向LSTM的模型结构
#     with tf.name_scope("Bi-LSTM"):
#         for idx, hiddenSize in enumerate(self.model.hiddenSizes):
#             with tf.name_scope("Bi-LSTM" + str(idx)):
#                 # 定义前向LSTM结构
#                 lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
#                     tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
#                     output_keep_prob=self.dropoutKeepProb
#                 )
#                 # 定义反向LSTM结构
#                 lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
#                     tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
#                     output_keep_prob=self.dropoutKeepProb
#                 )
#                 # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
#                 # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
#                 # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
#                 outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
#                                                                                self.embeddedWords, dtype=tf.float32,
#                                                                                scope="bi-lstm" + str(idx))
#                 # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
#                 self.embeddedWords = tf.concat(outputs_, 2)
#     # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
#     outputs = tf.split(self.embeddedWords, 2, -1)
#     # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
#     with tf.name_scope("Attention"):
#         h = outputs[0] + outputs[1]
#         # 得到Attention的输出
#         output = self._attention(h)
#         outputSize = self.model.hiddenSizes[-1]
#     def _attention(self, h):
#         """
#         利用Attention机制得到句子的向量表示
#         """
#         # 获得最后一层LSTM的神经元数量
#         hiddenSize = self.model.hiddenSizes[-1]
#         # 初始化一个权重向量，是可训练的参数
#         W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
#         # 对Bi-LSTM的输出用激活函数做非线性转换
#         M = tf.tanh(h)
#         # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
#         # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
#         newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
#         # 对newM做维度转换成[batch_size, time_step]
#         restoreM = tf.reshape(newM, [-1, self.sequenceLength])
#         # 用softmax做归一化处理[batch_size, time_step]
#         self.alpha = tf.nn.softmax(restoreM)
#         # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
#         r = tf.matmul(tf.transpose(h, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.sequenceLength, 1]))
#         # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
#         sequeezeR = tf.squeeze(r)
#         sentenceRepren = tf.tanh(sequeezeR)
#         # 对Attention的输出可以做dropout处理
#         output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
#         return output


# idx to embedding
def get_word_embd(word_embd_pretrain=None, vocab_size=None, word_embd_dim=None, name='word_embd'):
    # word_embd:[batch_size,step_len,embd_dim]
    if word_embd_pretrain is not None:
        word_embd = tf.get_variable(name=name, trainable=True, initializer=word_embd_pretrain)
        vocab_size = word_embd.shape[0].value
        word_embd_dim = word_embd.shape[1].value
    elif (vocab_size is None) or (word_embd_dim is None):
        print("get_word_embd para error!!!")
        return
    else:
        # 在word_embd_pretrain==[]时，需要根据词表大小vocab_size和词向量维度word_embd_dim，建立嵌入词向量word2vec
        word_embd = tf.get_variable(name=name, trainable=True,
                                    initializer=tf.random_uniform(
                                        (vocab_size, word_embd_dim), minval=-1, maxval=1, dtype=tf.float32)
                                    )
    return word_embd, vocab_size, word_embd_dim



