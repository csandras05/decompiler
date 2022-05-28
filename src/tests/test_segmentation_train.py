import jax
import numpy as np
import pytest
from jax import numpy as jnp
from model.train import segmentation_train, training


@pytest.mark.usefixtures
def test_train_step_updates_params(segmentation_train_state):
    batch_x = np.random.rand(2, 5, 128)
    batch_y = np.random.randint(0, 2, (2, 5))
    lengths = np.array([3, 5])
    new_state, _ = training.train_step(state=segmentation_train_state,
                                             batch_x=batch_x,
                                             batch_y=batch_y,
                                             lengths=lengths,
                                             error_fn=segmentation_train.binary_cross_entropy_loss,
                                             metric_fn=segmentation_train.compute_metrics)
    
    old_param_values = jax.tree_leaves(segmentation_train_state.params)
    new_param_values = jax.tree_leaves(new_state.params)
    for old_array, new_array in zip(old_param_values, new_param_values):
        assert not np.allclose(old_array, new_array)
        
        
def test_pad_input():
    input = [jnp.ones((5, 3)), jnp.ones((8, 3)), jnp.ones((10, 3))]
    padded = segmentation_train.pad_input(input, 8)
    assert jnp.all(padded == jnp.array([jnp.vstack([jnp.ones((5, 3)), jnp.zeros((3, 3))]),
                                        jnp.ones((8, 3)),
                                        jnp.ones((8, 3))]))

def test_pad_labels():
    input = [jnp.ones(5), jnp.ones(8), jnp.ones(10)]
    padded = segmentation_train.pad_label(input, 8)
    assert jnp.all(padded == jnp.array([jnp.hstack([jnp.ones(5), jnp.zeros(3)]),
                                        jnp.ones(8),
                                        jnp.ones(8)]))

def test_mask_sequences():
    batch = jnp.ones((3, 5))
    lengths = jnp.array([3, 5, 2])
    expected = jnp.array([[1, 1, 1, 0, 0],
                          [1, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0]])
    assert jnp.all(segmentation_train.mask_sequences(batch, lengths) == expected)

def test_metrics():
    logits = jnp.array([[0.7, 1.0, 1.0],
                        [0.6, 1.0, 1.0]])
    labels = jnp.array([[0, 1, 1],
                        [1, 1, 1]])
    lengths = jnp.array([3, 2])
    metrics = segmentation_train.compute_metrics(logits=logits, labels=labels, lengths=lengths)
    print(f"{metrics['loss'] = }")
    assert metrics['loss'] == -(jnp.log(0.3) + jnp.log(0.6)) / 2
    assert metrics['accuracy'] == 0.5
