import jax
import pytest
from jax import numpy as jnp
from model.flax_models import translation


@pytest.mark.parametrize('hidden_size,batch_size,max_len,embedding_size', [
    (256, 4, 40, 128)
])
def test_encoder_returns_correct_output_shape(hidden_size, batch_size, max_len, embedding_size):
    model = translation.Encoder(hidden_size)
    output, _ = model.init_with_output(jax.random.PRNGKey(0),
                                       jnp.ones((batch_size, max_len, embedding_size)))   
    assert output[0].shape == (batch_size, hidden_size)
    assert output[1].shape == (batch_size, hidden_size)

@pytest.mark.parametrize('hidden_size,batch_size,max_input_len,max_output_len,embedding_size,vocab_size', [
    (256, 4, 40, 40, 128, 18)
])
def test_decoder_returns_correct_output_shape(hidden_size, batch_size, max_input_len, max_output_len, embedding_size, vocab_size):
    encoder = translation.Encoder(hidden_size)
    init_decoder_state, _ = encoder.init_with_output(jax.random.PRNGKey(0),
                                       jnp.ones((batch_size, max_input_len, embedding_size)))
    decoder_inputs = jnp.ones((batch_size, max_output_len))
    decoder_inputs = jax.nn.one_hot(decoder_inputs, vocab_size)
    decoder = translation.Decoder(init_decoder_state, vocab_size)
    output, _ = decoder.init_with_output(jax.random.PRNGKey(0), decoder_inputs)
    assert output.shape == (batch_size, max_output_len, vocab_size)
