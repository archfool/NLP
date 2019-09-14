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


def seq2seq(x_id, y_id, keep_prob, train_or_infer, batch_size,
            x_id_extended, y_id_extended, vocab_size_extend,
            word_embd_dim, dim_rnn, use_same_word_embd=False,
            encoder_word_embd_pretrain=None, encoder_vocab_size=None,
            decoder_word_embd_pretrain=None, decoder_vocab_size=None,
            target_seq_len_max=None):
    with tf.variable_scope('encoder') as scope_encoder:
        # encoder[0] [batch_size, source_seq_max_len]:源序列，未embedding，类型np.array
        encoder = [x_id]
        # batch_size = encoder[0].shape[0].value
        encoder_seq_max_len = encoder[0].shape[1].value
        encoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(encoder[0]), axis=1), tf.int32)
        # encoder[1] [batch_size, source_seq_max_len, word_embd_dim]：对源序列数据进行embedding
        encoder_word_embd, encoder_vocab_size, word_embd_dim \
            = creat_word_embd(encoder_word_embd_pretrain, encoder_vocab_size, word_embd_dim, name='encoder_word_embd')
        encoder_w2v = tf.nn.embedding_lookup(encoder_word_embd, encoder[0])
        encoder.append(encoder_w2v)
        # encoder[2] ([batch_size,source_seq_max_len,dim_rnn*2], state_shape):构建encoder模型，并使用dynamic_rnn方法
        encoder_cell_fw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_rnn, state_is_tuple=True)
        encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_cell_fw_raw, output_keep_prob=keep_prob)
        encoder_cell_bw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_rnn, state_is_tuple=True)
        encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_cell_bw_raw, output_keep_prob=keep_prob)
        encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                 cell_bw=encoder_cell_bw,
                                                                 inputs=encoder[1],
                                                                 sequence_length=encoder_seq_len,
                                                                 dtype=tf.float32)
        memory = tf.concat(encoder_outputs, axis=2)
        init_state = init_state_reconstruct(encoder_state=state, encoder_state_type='bilstm',
                                          decoder_state_type='lstm', fill_zero=False)
        # init_state = BiLSTM_init_state_to_LSTM(BiLSTM_state=state, fill_zero=False)
        encoder.append((memory, init_state))
    with tf.variable_scope('decoder') as scope_decoder:
        # decoder[0] [batch_size,target_seq_len_max]:目标序列，未embedding，类型np.array
        decoder = [y_id]
        decoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(decoder[0]), axis=1), tf.int32)
        if use_same_word_embd is True:
            decoder_word_embd = encoder_word_embd
            decoder_vocab_size = encoder_vocab_size
        else:
            decoder_word_embd, decoder_vocab_size, word_embd_dim \
                = creat_word_embd(decoder_word_embd_pretrain, decoder_vocab_size, word_embd_dim, name='decoder_word_embd')
        decoder_w2v = tf.nn.embedding_lookup(decoder_word_embd, decoder[0])
        decoder.append(decoder_w2v)
        # decoder[2] 构建decoder模型
        # outputs: list of [batch_size, dim_rnn] by length target_seq_len_max
        # aligns: list of [batch_size, encoder_step_len] by length target_seq_len_max
        # p_gens: list of [batch_size] by length target_seq_len_max
        with tf.variable_scope('dynamic_decoder') as scope_dynamic_decoder:
            decoder_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_rnn, state_is_tuple=True)
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(cell=decoder_cell_raw, output_keep_prob=keep_prob)
            outputs, aligns, p_gens, cell_state = dynamic_decoder(cell=decoder_cell, memory=memory,
                                                                  memory_sequence_length=encoder_seq_len,
                                                                  init_state=init_state, train_or_infer=train_or_infer,
                                                                  decoder_seq_len_max=target_seq_len_max,
                                                                  target_seq_embd=decoder[1],
                                                                  decoder_word_embd=decoder_word_embd)
        decoder.append((outputs, aligns, p_gens))
        # decoder[3] 计算Generator_Network和Pointer_Network的输出
        # vocab_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extend] by length target_seq_len_max
        # attention_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extend] by len target_seq_len_max
        # p_gens: list of [batch_size] by length target_seq_len_max
        with tf.variable_scope('Generator_Network') as scope_Generator_Network:
            weight = tf.get_variable(
                name='weight', trainable=True,
                initializer=tf.truncated_normal([decoder_cell.output_size, decoder_vocab_size],
                                                stddev=math.sqrt(6 / (decoder_cell.output_size + decoder_vocab_size)),
                                                dtype=tf.float32)
            )
            bias = tf.get_variable(
                name='bias', trainable=True,
                initializer=tf.truncated_normal([decoder_vocab_size], stddev=0.1, dtype=tf.float32)
            )
            vocab_scores = []
            for step_num, output in enumerate(outputs):
                if step_num > 0:
                    tf.get_variable_scope().reuse_variables()
                vocab_scores.append(tf.nn.xw_plus_b(output, weight, bias))  # apply the linear layer
            # vocab_distributions: list [batch_size, decoder_vocab_size] length of decoder_step_len
            vocab_distributions = [tf.nn.softmax(score) for score in vocab_scores]
            # vocab_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extend] by length decoder_step_len
            vocab_dist_extendeds = [tf.pad(vocab_dist, paddings=[[0, 0], [0, vocab_size_extend]]) for vocab_dist in vocab_distributions]
            # todo to delete
            # extend_zeros = tf.zeros((batch_size, vocab_size_extend))
            # vocab_dist_extendeds= [tf.concat([vocab_dist, extend_zeros], axis=1) for vocab_dist in vocab_distributions]

        with tf.variable_scope('Pointer_Network') as scope_Pointer_Network:
            index_batch_num = tf.range(batch_size)
            index_batch_num = tf.expand_dims(index_batch_num, 1)
            index_batch_num = tf.tile(index_batch_num, [1, encoder_seq_max_len])
            index = tf.stack((index_batch_num, x_id_extended), axis=2)
            # attention_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extend] by len decoder_step_len
            attention_dist_extendeds = [tf.scatter_nd(index, align, [batch_size, decoder_vocab_size+vocab_size_extend])
                                        for align in aligns]
        decoder.append((vocab_dist_extendeds, attention_dist_extendeds, aligns))
        # decoder[4] 计算模型的最终输出
        # final_distributions: list of [batch_size, decoder_vocab_size+vocab_size_extend] by len target_seq_len_max
        with tf.variable_scope('Switching_Network') as scope_Switching_Network:
            final_distributions = [vocab_dist*p_gen + attn_dist*(1-p_gen) for (p_gen, vocab_dist, attn_dist)
                                   in zip(p_gens, vocab_dist_extendeds, attention_dist_extendeds)]

        decoder.append(final_distributions)
    return encoder, decoder


# built word embedding
def creat_word_embd(word_embd_pretrain=None, vocab_size=None, word_embd_dim=None, name='word_embd'):
    # word_embd:[batch_size,step_len,embd_dim]
    if word_embd_pretrain is not None:
        word_embd = tf.get_variable(name=name, trainable=True, initializer=word_embd_pretrain)
        vocab_size = word_embd.shape[0].value
        word_embd_dim = word_embd.shape[1].value
    elif (word_embd_pretrain is None) and ((vocab_size is None) or (word_embd_dim is None)):
        print("get_word_embd para error!!!")
        return
    else:
        # 在word_embd_pretrain==[]时，需要根据词表大小vocab_size和词向量维度word_embd_dim，建立嵌入词向量word2vec
        initializer = tf.random_uniform(shape=(vocab_size, word_embd_dim), minval=-1, maxval=1, dtype=tf.float32)
        # initializer = tf.truncated_normal(shape=(vocab_size, word_embd_dim), mean=0.0, stddev=1.0, dtype=tf.float32)
        word_embd = tf.get_variable(name=name, trainable=True, initializer=initializer)
    return word_embd, vocab_size, word_embd_dim



def init_state_reconstruct(encoder_state, encoder_state_type, decoder_state_type, fill_zero=False):
    if 'bilstm'==str.replace(str.lower(encoder_state_type), '-', '') and 'lstm'==str.replace(str.lower(decoder_state_type), '-', ''):
        dim_rnn = encoder_state[0][0].shape[1].value
        ((c_fw, h_fw), (c_bw, h_bw)) = encoder_state
        c_concat = tf.concat((c_fw, c_bw), axis=1)
        # c_concat = tf.concat((tf.expand_dims(c_fw, axis=2), tf.expand_dims(c_bw, axis=2)), axis=2)
        c_linear = linear(input=c_concat, output_size=dim_rnn, name='init_state_c')
        c = tf.nn.relu(c_linear)
        # c = tf.reduce_mean(c_concat, axis=2)
        h_concat = tf.concat((h_fw, h_bw), axis=1)
        # h_concat = tf.concat((tf.expand_dims(h_fw, axis=2), tf.expand_dims(h_bw, axis=2)), axis=2)
        h_linear = linear(input=h_concat, output_size=dim_rnn, name='init_state_h')
        h = tf.nn.relu(h_linear)
        # h = tf.reduce_mean(h_concat, axis=2)
        decoder_init_state =  tf.contrib.rnn.LSTMStateTuple(c_linear, h_linear)
    else:
        print("code is in coding...................................................\n")
    '''
    an useful process method:
    lstm_state_as_tensor_shape = [num_layers, 2, batch_size, hidden_size]
    initial_state = tf.zeros(lstm_state_as_tensor_shape)
    unstack_state = tf.unstack(initial_state, axis=0)
    tuple_state = tuple(
        [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(num_layers)])
    inputs = tf.unstack(inputs, num=num_steps, axis=1)
    outputs, state_out = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=tuple_state)
    '''
    return decoder_init_state


# dynamic decoder
def dynamic_decoder(cell, memory, memory_sequence_length, init_state, train_or_infer, decoder_seq_len_max,
                    target_seq_embd, decoder_word_embd):
    """
    :param cell: rnn_cell
    :param memory: [batch_size, step_len, dim_encoder_rnn]
    :param memory_sequence_length: [batch_size]
    :param init_state: decoder_rnn_cell.state
    :param train_or_infer: str
    :param decoder_seq_len_max: int
    :param target_seq_embd:
    :param decoder_word_embd: 
    :return outputs: list of [batch_size, dim_rnn] by length target_seq_len_max
    :return aligns: list of [batch_size, encoder_step_len] by length target_seq_len_max
    :return p_gens: list of [batch_size] by length target_seq_len_max
    :return cell_state: decoder_rnn_cell.state
    """
    outputs = []
    aligns = []
    p_gens = []
    # 根据init_state是LSTM/GRU/RNN的state，提取batch_size
    if isinstance(init_state, (tuple, list)):
        batch_size = init_state[0].shape[0].value
    else:
        batch_size = init_state.shape[0].value
    cell_state = init_state
    # decoder's first step input is <SOS>
    cell_input = target_seq_embd[:, 0, :]
    '''
    //to delete
    tgt_sos = tf.tile(input=decoder_word_embd[1, :], multiples=[batch_size])
    cell_input = tf.reshape(tgt_sos, [batch_size, -1])
    # cell_input = tf.fill(dims=[batch_size, 1, 1], value=1)
    # cell_input = np.array(decoder_word_embd[1, :]*batch_size).reshape([batch_size, 1, -1])
    '''
    for step_num in range(decoder_seq_len_max):
        # print('=============dynamic_decoder:step_num{}============='.format(step_num))
        if step_num > 0:
            variable_scope.get_variable_scope().reuse_variables()
        cell_output, cell_state = cell(cell_input, cell_state)
        context, align = attention_mechanism(decoder_num_units=None, memory=memory, memory_sequence_length=memory_sequence_length,
                                             decoder_state=cell_state, mode=1)
        # aligns
        # PointerNetwork的所有信息，只来自于aligns
        aligns.append(align)
        # p_gens
        with tf.variable_scope('p_gen'):
          p_gen = linear(input=[context, cell_state.c, cell_state.h, cell_input], output_size=1, name='p_gen')
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)
        # outputs
        output = linear(input=[cell_output] + [context], output_size=cell.output_size, name='decoder_output')
        outputs.append(output)
        if 'train' == train_or_infer:
            # outputs的信息只用于GenerateNetwork，所以只利用了source序列的未扩展的embedding。
            if step_num != decoder_seq_len_max-1:
                cell_input = target_seq_embd[:, step_num+1, :]
        elif 'infer' == train_or_infer:
            cell_input = context
        else:
            cell_input = context
            # print('param train_or_infer uninitialled !!!')
    return outputs, aligns, p_gens, cell_state

# Attention Mechanism
def attention_mechanism(decoder_num_units, memory, memory_sequence_length, decoder_state, mode=1):
    """
    :param decoder_num_units: dim of decoder vector
    :param memory: [batch_size, encoder_step_len_max, dim_rnn*2]
    :param decoder_state: [batch_size, decode_state_size]
    :param memory_sequence_length: [batch_size]
    :param mode: bool
    :return: context: [batch_size, dim_rnn]
    :return: align: [batch_size, encoder_step_len]
    """
    # mode = 1
    # encoder : Bi-LSTM, decoder : LSTM
    # score = v * tanh(enc_h * w_h + dec_s * w_s + bias)
    with tf.variable_scope('encoder_feature'):
        # memory:[batch_size, step_len, dim_rnn * 2]
        batch_size = memory.shape[0].value
        step_max_len = memory.shape[1].value
        dim_bi_rnn = memory.shape[2].value
        if decoder_num_units is None:
            dim_rnn = int(dim_bi_rnn / 2)
        else:
            dim_rnn = int(decoder_num_units)

        # memory_expand: [batch_size, step_len, 1, dim_rnn*2]
        memory_expand = tf.expand_dims(memory, axis=2)
        w_memory = tf.get_variable(name="w_memory", trainable=True, shape=[1, 1, dim_bi_rnn, dim_rnn])
        # encoder_feature: [batch_size, step_len, 1, dim_rnn]
        encoder_feature = nn_ops.conv2d(memory_expand, w_memory, [1, 1, 1, 1], "SAME")
    with tf.variable_scope('decoder_feature'):
        # w_dec_state = tf.get_variable(name="w_dec_state", trainable=True, shape=[total_arg_size, output_size])
        # decoder_feature_raw: [batch_size, dim_rnn]
        decoder_feature_raw = linear(input=decoder_state, output_size=dim_rnn, name="w_dec_state")
        # decoder_feature: [batch_size, 1, 1, dim_rnn]
        decoder_feature = tf.expand_dims(tf.expand_dims(decoder_feature_raw, 1), 1)
    with tf.variable_scope('align'):
        # bias: [dim_rnn]
        bias = tf.get_variable(name='bias', trainable=True,
                               initializer=tf.truncated_normal([dim_rnn], stddev=0.1, dtype=tf.float32))
        # v: [dim_rnn]
        v = tf.get_variable(name="v", trainable=True, shape=[dim_rnn])
        # score: [batch_size, step_len]
        score = math_ops.reduce_sum(v * math_ops.tanh(encoder_feature + decoder_feature + bias), [2, 3])
        # 归一化
        # align: [batch_size, step_len]
        align = seq_mask_norm(score, memory_sequence_length, step_max_len)
    with tf.variable_scope('context'):
        # context: [batch_size, dim_rnn]
        context = math_ops.reduce_sum(array_ops.reshape(align, [-1, step_max_len, 1, 1]) * encoder_feature, [1, 2])
        # context = math_ops.reduce_sum(array_ops.reshape(align, [-1, step_max_len, 1, 1]) * memory_expand, [1, 2])
        # to delete context = array_ops.reshape(context, [-1, dim_rnn])
        # context = array_ops.reshape(context, [-1, memory.shape[2].value])
    return context, align

def linear(input, output_size, name=None):
    """
    :param input: a 2D Tensor or a list of 2D, [batch, n], Tensors.
    :param output_size: int, second dimension of W[i], the dim of output data.
    :param name: variable_scope
    :return: A 2D Tensor with shape [batch, output_size(dim)] equal to sum_i(input[i] * W[i]).
    Linear map: sum_i(input[i] * W[i]), where W[i] is a trainable matrix variable.
    """
    if input is None or (isinstance(input, (list, tuple)) and not input):
        raise ValueError("`input` must be specified")
    if not isinstance(input, (list, tuple)):
        input = [input]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in input]
    for shape in shapes:
        if len(shape) != 2:
            print("Linear is expecting 2D arguments: {}".format(str(shapes)))
            raise ValueError("Linear is expecting 2D arguments: {}".format(str(shapes)))
        if not shape[1]:
            print("Linear expects shape[1] of arguments: {}".format(str(shapes)))
            raise ValueError("Linear expects shape[1] of arguments: {}".format(str(shapes)))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    matrix = tf.get_variable(name=name, trainable=True, shape=[total_arg_size, output_size])
    # perhaps the list means multi-layers
    if len(input) == 1:
        output = tf.matmul(input[0], matrix)
    else:
        output = tf.matmul(tf.concat(axis=1, values=input), matrix)
    return output

# 对序列按序列实际长度取掩码，并进行归一化
def seq_mask_norm(seq, seq_len, max_len):
    """
    :param seq: [batch_size, max_seq_len]
    :param seq_len: [batch_size, seq_len]
    :param max_len: int
    :return: seq_norm: [batch_size, max_seq_len]
    """
    seq_padding_mask = tf.cast(array_ops.sequence_mask(seq_len, max_len), tf.float32)
    seq_tmp = nn_ops.softmax(seq)
    seq_tmp *= seq_padding_mask
    masked_sums = tf.reduce_sum(seq_tmp, axis=1)
    seq_norm = seq_tmp / tf.reshape(masked_sums, [-1, 1])
    return seq_norm


