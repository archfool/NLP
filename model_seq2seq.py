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


def seq2seq(x_id, y_id, keep_prob, train_or_infer,
            word_embd_dim, dim_rnn, use_same_word_embd=False,
            encoder_word_embd_pretrain=None, encoder_vocab_size=None,
            decoder_word_embd_pretrain=None, decoder_vocab_size=None,
            x_id_extened=None, y_id_extened=None, vocab_size_extened=None,
            target_seq_len_max=None):
    with tf.variable_scope('encoder'):
        # encoder[0] [batch_size, source_seq_max_len]:源序列，已onehot，未embedding，类型np.array
        encoder = [x_id]
        batch_size = encoder[0].shape[0].value
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
    with tf.variable_scope('decoder'):
        # decoder[0] [batch_size,target_seq_len_max]:目标序列，已onehot，未embedding，类型np.array
        if 'train'==train_or_infer:
            decoder = [y_id]
            decoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(decoder[0]), axis=1), tf.int32)
        elif 'infer'==train_or_infer:
            decoder = [np.array([])]
        else:
            print('param train_or_infer value is error!!!')
        # decoder[1] [batch_size, target_seq_len_max, word_embd_dim]：对源序列数据进行embedding
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
        # vocab_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extened] by length target_seq_len_max
        # attention_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extened] by len target_seq_len_max
        # p_gens: list of [batch_size] by length target_seq_len_max
        with tf.variable_scope('Generator_Network'):
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
            # vocab_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extened] by length decoder_step_len
            extend_zeros = tf.zeros((batch_size, vocab_size_extened))
            vocab_dist_extendeds= [tf.concat([vocab_dist, extend_zeros], axis=1) for vocab_dist in vocab_distributions]
        with tf.variable_scope('Pointer_Network'):
            index_batch_num = tf.range(batch_size)
            index_batch_num = tf.expand_dims(index_batch_num, 1)
            index_batch_num = tf.tile(index_batch_num, [1, encoder_seq_max_len])
            index = tf.stack((index_batch_num, x_id_extened), axis=2)
            # attention_dist_extendeds: list [batch_size, decoder_vocab_size+vocab_size_extened] by len decoder_step_len
            attention_dist_extendeds = [tf.scatter_nd(index, align, [batch_size, decoder_vocab_size+vocab_size_extened])
                                        for align in aligns]
        decoder.append((vocab_dist_extendeds, attention_dist_extendeds, aligns))
        # decoder[4] 计算模型的最终输出
        # final_distributions: list of [batch_size, decoder_vocab_size+vocab_size_extened] by len target_seq_len_max
        with tf.variable_scope('Switching_Network'):
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
    if 'bilstm'==str.replace(str.lower(encoder_state_type), '-', '')\
        and 'lstm'==str.replace(str.lower(decoder_state_type), '-', ''):
        dim_rnn = encoder_state[0][0].shape[1].value
        ((c_fw, h_fw), (c_bw, h_bw)) = encoder_state
        c_concat = tf.concat((tf.expand_dims(c_fw, axis=2), tf.expand_dims(c_bw, axis=2)), axis=2)
        c_linear = linear(input=c_concat, output_size=dim_rnn, name='init_state_c')
        c = tf.nn.relu(c_linear)
        # c = tf.reduce_mean(c_concat, axis=2)
        h_concat = tf.concat((tf.expand_dims(h_fw, axis=2), tf.expand_dims(h_bw, axis=2)), axis=2)
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
    batch_size = len(init_state)
    cell_state = init_state
    # decoder's first step input is <SOS>
    cell_input = tf.fill(dims=[batch_size, 1, 1], value=1)
    # cell_input = np.array(decoder_word_embd[1, :]*batch_size).reshape([batch_size, 1, -1])
    for step_num in range(decoder_seq_len_max):
        if step_num > 0:
            variable_scope.get_variable_scope().reuse_variables()
        cell_output, cell_state = cell(cell_input, cell_state)
        context, align = attention_mechanism(decoder_num_units=None, memory=memory, memory_sequence_length=memory_sequence_length,
                                             decoder_state=cell_state, mode=1)
        # aligns
        aligns.append(align)
        # p_gens
        with tf.variable_scope('p_gen'):
          p_gen = linear([context, cell_state.c, cell_state.h, cell_input], 1)
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)
        # outputs
        output = linear([cell_output] + [context], cell.output_size, True)
        outputs.append(output)
        if 'train' == train_or_infer:
            cell_input = target_seq_embd[:, step_num, :]
        elif 'infer' == train_or_infer:
            cell_input = context
        else:
            print('param train_or_infer is error!!!')
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
        decoder_feature_raw = linear(decoder_state, dim_rnn, name="w_dec_state")
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
        context = math_ops.reduce_sum(array_ops.reshape(align, [batch_size, -1, 1, 1]) * memory_expand, [1, 2])
        context = array_ops.reshape(context, [-1, dim_rnn])
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
        input = list(input)

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in input]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
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


# todo delete or rewrite?
class helper(object):
    def __init__(self, inputs, sequence_length):
        self.inputs = inputs
        self.sequence_length = sequence_length
        self.inputs_step = tf.unstack(inputs, axis=1)
        self.step_len_max = len(self.inputs_step)

    def __iter__(self):
        self.i = 1
        return self

    def __next__(self):
        if self.i <= len(self.inputs):
            output = self.inputs[0]
            self.inputs = self.inputs[1:]
            self.i +=1
            return output
        else:
            raise StopIteration

    def get_data(self):
        if len(self.inputs_step) > 0:
            output = self.inputs_step[0]
            self.inputs_step = self.inputs_step[1:]
            return output
        else:
            print('================out of index=================')
            return []



def seq2seq_old(x, y, keep_prob, batch_size, train_or_infer,
            dim_lstm, word_embd_dim,
            encoder_word_embd_pretrain=None, encoder_vocab_size=None,
            decoder_word_embd_pretrain=None, decoder_vocab_size=None):
    with tf.variable_scope('encoder'):
        # encoder[0] 输入数据，已onehot，未embedding，类型np.array
        # x:[batch_size,step_len]
        encoder = [x]
        encoder_seq_max_len = encoder[0].shape[1].value
        encoder_seq_len = tf.cast(tf.reduce_sum(tf.sign(encoder[0]), axis=1), tf.int32)
        # encoder[1] 对源序列数据进行embedding
        encoder_word_embd, encoder_vocab_size, word_embd_dim \
            = creat_word_embd(encoder_word_embd_pretrain, encoder_vocab_size, word_embd_dim, name='encoder_word_embd')
        encoder_embedding_w2v = tf.nn.embedding_lookup(encoder_word_embd, encoder[0])
        encoder.append(encoder_embedding_w2v)
        # encoder[2] 构建encoder模型，并使用dynamic_rnn方法
        # lstm_layer:[batch_size,step_len,dim_lstm*2]
        lstm_cell_fw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
        lstm_cell_bw_raw = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim_lstm, state_is_tuple=True)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_fw_raw, output_keep_prob=keep_prob)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_bw_raw, output_keep_prob=keep_prob)
        encoder_outputs, (encoder_state_fw, encoder_state_bw) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                            cell_bw=lstm_cell_bw,
                                            inputs=encoder[1],
                                            sequence_length=encoder_seq_len,
                                            dtype=tf.float32)
        memory = tf.concat(encoder_outputs, axis=2)
        encoder.append(memory)
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
    with tf.variable_scope('decoder'):
        # TODO: layer_output = [tf.convert_to_tensor(y)]
        # decoder[0] 目标序列数据
        decoder = [y]
        # 提取word_embd矩阵
        decoder_word_embd, decoder_vocab_size, word_embd_dim \
            = creat_word_embd(decoder_word_embd_pretrain, decoder_vocab_size, word_embd_dim, name='decoder_word_embd')
        # decoder[1] 构建encoder模型，并使用dynamic_rnn方法
        # 构建decoder模型
        decoder_cell_raw = tf.nn.rnn_cell.BasicLSTMCell(dim_lstm, state_is_tuple=True)
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
        decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=(encoder_state_fw, encoder_state_bw))
        # 训练模式
        if 'train' == train_or_infer:
            # 获取目标序列信息
            # TODO: layer_output = [tf.convert_to_tensor(x)]
            decoder_seq_len_max = decoder[0].shape[1].value
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
        # 预测模式
        elif 'infer' == train_or_infer:
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
            decoder_outputs, decoder__state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder_help, maximum_iterations=encoder_seq_max_len*2)
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
# #            decoder_step_len = y.shape[1].value
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








