"""Module for RNN Wrappers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
RNNCell = rnn_cell.RNNCell


class Stack_Residual_RNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, use_residual_connections = True, state_is_tuple=True, num_gpus=0):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
      use_residetual_connections: add previous input to the next stacked layer. Allows for much deeper networks.
      Recommended to use if layer number is larger than three.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    self._cells = cells
    self._state_is_tuple = state_is_tuple
    self._use_residual_connections = use_residual_connections
    self._num_gpus = num_gpus
    if not state_is_tuple:
      if any(nest.is_sequence(seq=c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      if self._use_residual_connections:
        past_inp = tf.zeros_like(tensor=cur_inp)
      new_states = []
      if self._num_gpus:
        for i, cell in enumerate(self._cells):
          with tf.device('/device:GPU:' + str(i % 8)):
            with vs.variable_scope("Cell%d" % i):
              if self._state_is_tuple:
                if not nest.is_sequence(seq=state):
                  raise ValueError(
                      "Expected state to be a tuple of length %d, but received: %s"
                      % (len(self.state_size), state))
                cur_state = state[i]
              else:
                cur_state = array_ops.slice(input_=state, begin=[0, cur_state_pos], size=[-1, cell.state_size])
                cur_state_pos += cell.state_size
              if self._use_residual_connections:
                cur_inp += past_inp
                past_inp = cur_inp
              cur_inp, new_state = cell(cur_inp, cur_state)
              new_states.append(new_state)
      else:
        with tf.device('/cpu:0'):
          for i, cell in enumerate(self._cells):
            with vs.variable_scope("Cell%d" % i):
              if self._state_is_tuple:
                if not nest.is_sequence(seq=state):
                  raise ValueError(
                    "Expected state to be a tuple of length %d, but received: %s"
                    % (len(self.state_size), state))
                cur_state = state[i]
              else:
                cur_state = array_ops.slice(input_=state, begin=[0, cur_state_pos], size=[-1, cell.state_size])
                cur_state_pos += cell.state_size
              if self._use_residual_connections:
                cur_inp += past_inp
                past_inp = cur_inp
              cur_inp, new_state = cell(cur_inp, cur_state)
              new_states.append(new_state)
    new_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat(axis=1, values=new_states))
    return cur_inp, new_states