import os

import jax
import numpy as np
import optax
from flax.training import train_state
from jax import numpy as jnp
from matplotlib import pyplot as plt
from model import utils
from model.flax_models.segmentation import SegmentationModel
from model.train import training
from sklearn.model_selection import train_test_split

CUR_DIR = os.path.dirname(__file__)

def mask_sequences(sequence_batch, lengths):
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis])

def binary_cross_entropy_loss(*, logits, labels, lengths):
  probs = logits * labels + (1-logits) * (1-labels)
  x = jnp.log(probs)
  x = mask_sequences(x, lengths)
  return -jnp.mean(jnp.sum(x, axis=-1))

def compute_metrics(*, logits, labels, lengths):
    loss = binary_cross_entropy_loss(logits=logits, labels=labels, lengths=lengths)
    token_accuracy = (jnp.round(logits) == labels)
    sequence_accuracy = (jnp.sum(mask_sequences(token_accuracy, lengths), axis=-1) == lengths)
    accuracy = jnp.mean(sequence_accuracy)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

def create_train_state(rng, optimizer, learning_rate, hidden_size, max_len, embedding_size):
    model = SegmentationModel(hidden_size)
    params = model.init(rng, jnp.ones((1, max_len, embedding_size)))['params']
    tx = optimizer(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def pad_input(input_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0]), (0, 0)]) for x in input_data])

def pad_label(label_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0])]) for x in label_data])


if __name__ == '__main__':

    data = utils.load_json(f'{CUR_DIR}/data.json')
    config = utils.load_yaml(f'{CUR_DIR}/segmentation_train.yaml')
     
    train_x, test_x, train_y, test_y = train_test_split(data['embedding'], data['labels'],
                                                        test_size=config['test_ratio'],
                                                        random_state=0)
    
    train_lengths = jnp.array(list(map(len, train_x)))
    test_lengths = jnp.array(list(map(len, test_x)))
        
    train_x = pad_input([jnp.array(d) for d in train_x], config['max_len'])
    train_y = pad_label([jnp.array(d) for d in train_y], config['max_len'])
    test_x = pad_input([jnp.array(d) for d in test_x], config['max_len'])
    test_y = pad_label([jnp.array(d) for d in test_y], config['max_len'])
    
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng=rng,
                               optimizer=training.OPTIMIZERS[config['optimizer']],
                               learning_rate=config['learning_rate'],
                               hidden_size=config['hidden_size'],
                               max_len=config['max_len'],
                               embedding_size=config['embedding_size'])
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, config['num_epochs'] + 1):
        rng, input_rng = jax.random.split(rng)
        state, metrics = training.train_epoch(state,
                                              train_x, train_y, train_lengths,
                                              config['batch_size'], epoch, input_rng,
                                              binary_cross_entropy_loss,
                                              compute_metrics)
        train_losses.append(metrics['loss'])
        train_accuracies.append(metrics['accuracy'])
        
        test_loss, test_accuracy = training.eval_model(state,
                                                       test_x, test_y, test_lengths,
                                                       compute_metrics)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'\ttest epoch: {epoch}, loss: {test_loss:.5f}, accuracy: {test_accuracy*100:.2f}')

    training.visualize(train_losses=train_losses, test_losses=test_losses,
                       train_accuracies=train_accuracies, test_accuracies=test_accuracies)
    
    
    training.export('segmentation_model_', state.params)
