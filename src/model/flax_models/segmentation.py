from typing import List, Tuple

import jax
from flax import linen as nn
from flax import serialization
from jax import numpy as jnp


class RNN(nn.Module):
  @nn.compact
  def __call__(self, carry, x):
    LSTM = nn.scan(nn.LSTMCell,
                   variable_broadcast="params",
                   split_rngs={"params": False},
                   in_axes=1,
                   out_axes=1)
    return LSTM()(carry, x)
  
  @staticmethod
  def initialize_carry(batch_dim: Tuple, hidden_size: int):
    return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dim, hidden_size)
  
class SegmentationModel(nn.Module):
  hidden_size: int
  
  @nn.compact
  def __call__(self, x):
    batch_size = x.shape[0]
    rnn = RNN()
    init_carry = rnn.initialize_carry((batch_size,), self.hidden_size)
    _, y = rnn(init_carry, x)
    y = nn.Dense(features=1)(y)
    return nn.sigmoid(y)

class Segmentation:
    def __init__(self,
                 params_file: str,
                 model: SegmentationModel,
                 max_len: int,
                 embedding_size: int):
      
        self.model = model
        
        with open(params_file, 'rb') as f:
            params_bin = f.read()

        init_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, max_len, embedding_size)))['params']
        self.params = serialization.from_bytes(init_params, params_bin)
    
    
    def get_segmentation(self, embeddings: jnp.ndarray) -> List[int]:
        embeddings = jnp.expand_dims(embeddings, axis=0)
        preds = jnp.squeeze(self.model.apply({'params': self.params}, embeddings))
        indices = [i for i, x in enumerate(preds) if x >= 0.5]
        indices.append(len(preds))
        return indices

