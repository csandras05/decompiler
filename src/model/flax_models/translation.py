import functools
from typing import Any, List

import jax
from flax import linen as nn
from flax import serialization
from jax import numpy as jnp


class EncoderLSTM(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        return nn.LSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size)


class Encoder(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        lstm = EncoderLSTM()
        init_carry = lstm.initialize_carry(batch_size, self.hidden_size)
        final_state, _ = lstm(init_carry, inputs)
        return final_state


class DecoderLSTM(nn.Module):
    vocab_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False, 'lstm': True})
    @nn.compact
    def __call__(self, carry, x):
        lstm_state, last_prediction = carry
        x = last_prediction
        lstm_state, x = nn.LSTMCell()(lstm_state, x)
        x = nn.Dense(features=self.vocab_size)(x)
        y = nn.log_softmax(x)
        predicted_token = jnp.argmax(y, axis=-1)
        prediction = jax.nn.one_hot(predicted_token, num_classes=self.vocab_size)
        
        return (lstm_state, prediction), y


class Decoder(nn.Module):
    init_state: Any
    vocab_size: int

    @nn.compact
    def __call__(self, inputs):
        lstm = DecoderLSTM(vocab_size=self.vocab_size)
        init_carry = (self.init_state, inputs[:, 0])
        _, y = lstm(init_carry, inputs)
        return y


class Seq2seq(nn.Module):
    hidden_size: int
    vocab_size: int
    sos_id: int
    max_output_len: int

    @nn.compact
    def __call__(self, encoder_inputs):
        init_decoder_state = Encoder(hidden_size=self.hidden_size)(encoder_inputs)
        batch_size = encoder_inputs.shape[0]
        decoder_inputs = self.sos_id * jnp.ones((batch_size, self.max_output_len))
        decoder_inputs = jax.nn.one_hot(decoder_inputs, self.vocab_size)
        y = Decoder(init_state=init_decoder_state, vocab_size=self.vocab_size)(decoder_inputs)

        return y    

class Translation:
    def __init__(self, params_file: str, model: Seq2seq):
        self.model = model
        self.token_to_id = {'<PAD>': 0,
                            '<SOS>': 1,
                            'VAR': 2,
                            '=': 3,
                            'NUM': 4,
                            ';': 5,
                            '(': 6,
                            '%': 7,
                            ')': 8,
                            '&': 9,
                            '/': 10,
                            '-': 11,
                            '*': 12,
                            '^': 13,
                            '+': 14,
                            '>>': 15,
                            '|': 16,
                            '<<': 17}
        self.id_to_token = list(self.token_to_id.keys())
        
        # TODO: place these values in a config file
        self.max_input_len = 36
        self.max_output_len = 29
        
        with open(params_file, 'rb') as f:
            params_bin = f.read()
        init_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, self.max_input_len, 128)))['params']
        self.params = serialization.from_bytes(init_params, params_bin)
        
        with open(params_file, 'rb') as f:
            params_bin = f.read()

        init_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 62, 128)))['params']
        self.params = serialization.from_bytes(init_params, params_bin)
    
    def _id_seq_to_c_line(self, id_seq) -> str:
        until = jnp.where(id_seq == self.token_to_id[';'])[0][0] + 1
        tokens = map(lambda x: self.id_to_token[int(x)], id_seq[1:until])
        return ' '.join(tokens)
    
    def _pad_input(self, input_data, max_len):
        return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0]), (0, 0)]) for x in input_data])
    
    def translate(self, embeddings) -> List[str]:
        padded = self._pad_input(embeddings,
                                 self.max_input_len)
        logits = self.model.apply({'params': self.params}, padded)
        preds = jnp.argmax(logits, axis=-1)
        return list(map(self._id_seq_to_c_line, preds))
