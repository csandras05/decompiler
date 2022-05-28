import functools

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from model.train import training, translation_train


@pytest.mark.usefixtures
def test_train_step_updates_params(translation_train_state):
    batch_x = np.random.rand(2, 5, 128)
    batch_y = np.random.randint(0, 10, (2, 40))
    lengths = np.array([15, 20])
    error_fn = functools.partial(translation_train.cross_entropy_loss, token_cnt=2)
    metric_fn = functools.partial(translation_train.compute_metrics, token_cnt=2)
    new_state, _ = training.train_step(state=translation_train_state,
                                             batch_x=batch_x,
                                             batch_y=batch_y,
                                             lengths=lengths,
                                             error_fn=error_fn,
                                             metric_fn=metric_fn)
    
    old_param_values = jax.tree_leaves(translation_train_state.params)
    new_param_values = jax.tree_leaves(new_state.params)
    for old_array, new_array in zip(old_param_values, new_param_values):
        assert not np.allclose(old_array, new_array)


def test_pad_input():
    input = [jnp.ones((5, 3)), jnp.ones((8, 3)), jnp.ones((10, 3))]
    padded = translation_train.pad_input(input, 8)
    assert jnp.all(padded == jnp.array([jnp.vstack([jnp.ones((5, 3)), jnp.zeros((3, 3))]),
                                        jnp.ones((8, 3)),
                                        jnp.ones((8, 3))]))

def test_pad_output():
    input = [jnp.ones(5), jnp.ones(8), jnp.ones(10)]
    padded = translation_train.pad_output(input, 8)
    assert jnp.all(padded == jnp.array([jnp.hstack([jnp.ones(5), jnp.zeros(3)]),
                                        jnp.ones(8),
                                        jnp.ones(8)]))

def test_mask_sequences():
    batch = jnp.ones((3, 5))
    lengths = jnp.array([3, 5, 2])
    expected = jnp.array([[1, 1, 1, 0, 0],
                          [1, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0]])
    assert jnp.all(translation_train.mask_sequences(batch, lengths) == expected)

def test_metrics():
    logits = jnp.log(jnp.array([[[0.3, 0.7], [0.1, 0.9], [0.1, 0.9]],
                                 [[0.4, 0.6], [0.1, 0.9], [0.1, 0.9]]]))
    labels = jnp.array([[0, 1, 1],
                        [1, 1, 0]])
    lengths = jnp.array([3, 2])
    metrics = translation_train.compute_metrics(logits=logits, labels=labels, lengths=lengths, token_cnt=2)
    print(f"{metrics['loss'] = }")
    assert metrics['loss'] == - (jnp.log(0.9) * 3 + jnp.log(0.3) + jnp.log(0.6)) / 6
    assert metrics['accuracy'] == 0.5
