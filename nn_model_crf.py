# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for constructing a linear-chain CRF.

The following snippet is an example of a CRF layer on top of a batched sequence
of unary scores (logits for every word). This example also decodes the most
likely sequence at test time. There are two ways to do decoding. One
is using crf_decode to do decoding in Tensorflow , and the other one is using
viterbi_decode in Numpy.

log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
    unary_scores, gold_tags, sequence_lengths)

loss = tf.reduce_mean(-log_likelihood)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Decoding in Tensorflow.
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
    unary_scores, transition_params, sequence_lengths)

tf_viterbi_sequence, tf_viterbi_score, _ = session.run(
    [viterbi_sequence, viterbi_score, train_op])

# Decoding in Numpy.
tf_unary_scores, tf_sequence_lengths, tf_transition_params, _ = session.run(
    [unary_scores, sequence_lengths, transition_params, train_op])
for tf_unary_scores_, tf_sequence_length_ in zip(tf_unary_scores,
                                                 tf_sequence_lengths):
# Remove padding.
tf_unary_scores_ = tf_unary_scores_[:tf_sequence_length_]

# Compute the highest score and its tag sequence.
tf_viterbi_sequence, tf_viterbi_score = tf.contrib.crf.viterbi_decode(
    tf_unary_scores_, tf_transition_params)
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import numpy as np

#from tensorflow.python.framework import tf
#from tensorflow.python.framework import dtypes
from tensorflow.python.layers import utils
#from tensorflow.python.ops import tf
#from tensorflow.python.ops import control_flow_ops
#from tensorflow.python.ops import gen_tf
#from tensorflow.python.ops import tf
#from tensorflow.python.ops import rnn
#from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import variable_scope as vs

#__all__ = [
#    "crf_sequence_score", "crf_log_norm", "crf_log_likelihood",
#    "crf_unary_score", "crf_binary_score", "CrfForwardRnnCell",
#    "viterbi_decode", "crf_decode", "CrfDecodeForwardRnnCell",
#    "CrfDecodeBackwardRnnCell"
#]


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
  """Computes the unnormalized score for a tag sequence.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the unnormalized score.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
  Returns:
    sequence_scores: A [batch_size] vector of unnormalized sequence scores.
  """
  # If max_seq_len is 1, we skip the score calculation and simply gather the
  # unary potentials of the single tag.
  def _single_seq_fn():
    batch_size = tf.shape(inputs, out_type=tag_indices.dtype)[0]
    example_inds = tf.reshape(
        tf.range(batch_size, dtype=tag_indices.dtype), [-1, 1])
    sequence_scores = tf.gather_nd(
        tf.squeeze(inputs, [1]),
        tf.concat([example_inds, tag_indices], axis=1))
    sequence_scores = tf.where(tf.less_equal(sequence_lengths, 0),
                                      tf.zeros_like(sequence_scores),
                                      sequence_scores)
    return sequence_scores

  def _multi_seq_fn():
    # Compute the scores of the given tag sequence.
    unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
    binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                     transition_params)
    sequence_scores = unary_scores + binary_scores
    return sequence_scores

  return utils.smart_cond(
      pred=tf.equal(inputs.shape[1].value or tf.shape(inputs)[1],
                          1),
      true_fn=_single_seq_fn,
      false_fn=_multi_seq_fn)


def crf_log_norm(inputs, sequence_lengths, transition_params):
  """Computes the normalization for a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
  Returns:
    log_norm: A [batch_size] vector of normalizers for a CRF.
  """
  # Split up the first and rest of the inputs in preparation for the forward
  # algorithm.
  first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = tf.squeeze(first_input, [1])

  # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
  # the "initial state" (the unary potentials).
  def _single_seq_fn():
    log_norm = tf.reduce_logsumexp(first_input, [1])
    # Mask `log_norm` of the sequences with length <= zero.
    log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                               tf.zeros_like(log_norm),
                               log_norm)
    return log_norm

  def _multi_seq_fn():
    """Forward computation of alpha values."""
    rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])

    # Compute the alpha values in the forward algorithm in order to get the
    # partition function.
    forward_cell = CrfForwardRnnCell(transition_params)
    # Sequence length is not allowed to be less than zero.
    sequence_lengths_less_one = tf.maximum(
        tf.constant(0, dtype=sequence_lengths.dtype),
        sequence_lengths - 1)
    _, alphas = tf.nn.dynamic_rnn(
        cell=forward_cell,
        inputs=rest_of_input,
        sequence_length=sequence_lengths_less_one,
        initial_state=first_input,
        dtype=tf.float32)
    log_norm = tf.reduce_logsumexp(alphas, [1])
    # Mask `log_norm` of the sequences with length <= zero.
    log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                               tf.zeros_like(log_norm),
                               log_norm)
    return log_norm

  max_seq_len = tf.shape(inputs)[1]
  return tf.cond(pred=tf.equal(max_seq_len, 1),
                               true_fn=_single_seq_fn,
                               false_fn=_multi_seq_fn)


def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
  """Computes the log-likelihood of tag sequences in a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the log-likelihood.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix, if available.
  Returns:
    log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
      each example, given the sequence of tag indices.
    transition_params: A [num_tags, num_tags] transition matrix. This is either
        provided by the caller or created in this function.
  """
  with tf.variable_scope('crf_layer'):
      # Get shape information.
      num_tags = inputs.get_shape()[2].value
    
      # Get the transition matrix if not provided.
      if transition_params is None:
        transition_params = tf.get_variable("transitions", [num_tags, num_tags])
    
      sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                           transition_params)
      log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)
    
      # Normalize the scores to get the log-likelihood per example.
      log_likelihood = sequence_scores - log_norm
  return log_likelihood, transition_params


def crf_unary_score(tag_indices, sequence_lengths, inputs):
  """Computes the unary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
  Returns:
    unary_scores: A [batch_size] vector of unary scores.
  """
  batch_size = tf.shape(inputs)[0]
  max_seq_len = tf.shape(inputs)[1]
  num_tags = tf.shape(inputs)[2]

  flattened_inputs = tf.reshape(inputs, [-1])

  offsets = tf.expand_dims(
      tf.range(batch_size) * max_seq_len * num_tags, 1)
  offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
  # Use int32 or int64 based on tag_indices' dtype.
  if tag_indices.dtype == tf.int64:
    offsets = tf.to_int64(offsets)
  flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

  unary_scores = tf.reshape(
      tf.gather(flattened_inputs, flattened_tag_indices),
      [batch_size, max_seq_len])

  masks = tf.sequence_mask(sequence_lengths,
                                  maxlen=tf.shape(tag_indices)[1],
                                  dtype=tf.float32)

  unary_scores = tf.reduce_sum(unary_scores * masks, 1)
  return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
  """Computes the binary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    binary_scores: A [batch_size] vector of binary scores.
  """
  # Get shape information.
  num_tags = transition_params.get_shape()[0]
  num_transitions = tf.shape(tag_indices)[1] - 1

  # Truncate by one on each side of the sequence to get the start and end
  # indices of each transition.
  start_tag_indices = tf.slice(tag_indices, [0, 0],
                                      [-1, num_transitions])
  end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

  # Encode the indices in a flattened representation.
  flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
  flattened_transition_params = tf.reshape(transition_params, [-1])

  # Get the binary scores based on the flattened representation.
  binary_scores = tf.gather(flattened_transition_params,
                                   flattened_transition_indices)

  masks = tf.sequence_mask(sequence_lengths,
                                  maxlen=tf.shape(tag_indices)[1],
                                  dtype=tf.float32)
  truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
  binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
  return binary_scores


class CrfForwardRnnCell(tf.nn.rnn_cell.RNNCell):
  """Computes the alpha values in a linear-chain CRF.

  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """

  def __init__(self, transition_params):
    """Initialize the CrfForwardRnnCell.

    Args:
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
    """
    self._transition_params = tf.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.

    Returns:
      new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """
    state = tf.expand_dims(state, 2)

    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension. This performs the
    # multiplication of previous alpha values and the current binary potentials
    # in log space.
    transition_scores = state + self._transition_params
    new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])

    # Both the state and the output of this RNN cell contain the alphas values.
    # The output value is currently unused and simply satisfies the RNN API.
    # This could be useful in the future if we need to compute marginal
    # probabilities, which would require the accumulated alpha values at every
    # time step.
    return new_alphas, new_alphas

def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

'''

class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
  """Computes the forward decoding in a linear-chain CRF.
  """

  def __init__(self, transition_params):
    """Initialize the CrfDecodeForwardRnnCell.

    Args:
      transition_params: A [num_tags, num_tags] matrix of binary
        potentials. This matrix is expanded into a
        [1, num_tags, num_tags] in preparation for the broadcast
        summation occurring within the cell.
    """
    self._transition_params = tf.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfDecodeForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      scope: Unused variable scope of this cell.

    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    state = tf.expand_dims(state, 2)                         # [B, O, 1]

    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension.
    # [B, O, 1] + [1, O, O] -> [B, O, O]
    transition_scores = state + self._transition_params             # [B, O, O]
    new_state = inputs + tf.reduce_max(transition_scores, [1])  # [B, O]
    backpointers = tf.argmax(transition_scores, 1)
    backpointers = tf.cast(backpointers, dtype=dtypes.int32)    # [B, O]
    return backpointers, new_state


class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
  """Computes backward decoding in a linear-chain CRF.
  """

  def __init__(self, num_tags):
    """Initialize the CrfDecodeBackwardRnnCell.

    Args:
      num_tags: An integer. The number of tags.
    """
    self._num_tags = num_tags

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return 1

  def __call__(self, inputs, state, scope=None):
    """Build the CrfDecodeBackwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.
      scope: Unused variable scope of this cell.

    Returns:
      new_tags, new_tags: A pair of [batch_size, num_tags]
        tensors containing the new tag indices.
    """
    state = tf.squeeze(state, axis=[1])                # [B]
    batch_size = tf.shape(inputs)[0]
    b_indices = tf.range(batch_size)                    # [B]
    indices = tf.stack([b_indices, state], axis=1)     # [B, 2]
    new_tags = tf.expand_dims(
        gen_tf.gather_nd(inputs, indices),             # [B]
        axis=-1)                                              # [B, 1]

    return new_tags, new_tags


def crf_decode(potentials, transition_params, sequence_length):
  """Decode the highest scoring sequence of tags in TensorFlow.

  This is a function for tensor.

  Args:
    potentials: A [batch_size, max_seq_len, num_tags] tensor of
              unary potentials.
    transition_params: A [num_tags, num_tags] matrix of
              binary potentials.
    sequence_length: A [batch_size] vector of true sequence lengths.

  Returns:
    decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                Contains the highest scoring tag indices.
    best_score: A [batch_size] vector, containing the score of `decode_tags`.
  """
  # If max_seq_len is 1, we skip the algorithm and simply return the argmax tag
  # and the max activation.
  def _single_seq_fn():
    squeezed_potentials = tf.squeeze(potentials, [1])
    decode_tags = tf.expand_dims(
        tf.argmax(squeezed_potentials, axis=1), 1)
    best_score = tf.reduce_max(squeezed_potentials, axis=1)
    return tf.cast(decode_tags, dtype=dtypes.int32), best_score

  def _multi_seq_fn():
    """Decoding of highest scoring sequence."""

    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    num_tags = potentials.get_shape()[2].value

    # Computes forward decoding. Get last score and backpointers.
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
    initial_state = tf.squeeze(initial_state, axis=[1])  # [B, O]
    inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
    # Sequence length is not allowed to be less than zero.
    sequence_length_less_one = tf.maximum(0, sequence_length - 1)
    backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O], [B, O]
        crf_fwd_cell,
        inputs=inputs,
        sequence_length=sequence_length_less_one,
        initial_state=initial_state,
        time_major=False,
        dtype=dtypes.int32)
    backpointers = gen_tf.reverse_sequence(  # [B, T - 1, O]
        backpointers, sequence_length_less_one, seq_dim=1)

    # Computes backward decoding. Extract tag indices from backpointers.
    crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
    initial_state = tf.cast(tf.argmax(last_score, axis=1),  # [B]
                                  dtype=dtypes.int32)
    initial_state = tf.expand_dims(initial_state, axis=-1)  # [B, 1]
    decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
        crf_bwd_cell,
        inputs=backpointers,
        sequence_length=sequence_length_less_one,
        initial_state=initial_state,
        time_major=False,
        dtype=dtypes.int32)
    decode_tags = tf.squeeze(decode_tags, axis=[2])  # [B, T - 1]
    decode_tags = tf.concat([initial_state, decode_tags],   # [B, T]
                                   axis=1)
    decode_tags = gen_tf.reverse_sequence(  # [B, T]
        decode_tags, sequence_length, seq_dim=1)

    best_score = tf.reduce_max(last_score, axis=1)  # [B]
    return decode_tags, best_score

  return utils.smart_cond(
      pred=tf.equal(potentials.shape[1].value or
                          tf.shape(potentials)[1], 1),
      true_fn=_single_seq_fn,
      false_fn=_multi_seq_fn)
'''