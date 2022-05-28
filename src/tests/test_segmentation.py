import jax
import pytest
from jax import numpy as jnp
from model.flax_models import segmentation


@pytest.mark.parametrize('hidden_size,batch_size,max_len,embedding_size', [
    (256, 4, 40, 128)
])
def test_segmentation_model_returns_correct_output_shape(hidden_size, batch_size, max_len, embedding_size):
    model = segmentation.SegmentationModel(hidden_size)
    output, _ = model.init_with_output(jax.random.PRNGKey(0),
                                       jnp.ones((batch_size, max_len, embedding_size)))   
    assert output.shape == (batch_size, max_len, 1)
