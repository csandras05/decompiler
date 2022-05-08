import os

import jax
import numpy as np
import optax
from flax.training import train_state
from jax import numpy as jnp
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

def create_train_state(rng, learning_rate, hidden_size):
    model = SegmentationModel(hidden_size)
    params = model.init(rng, jnp.ones((1, 90, 128)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def pad_input(input_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0]), (0, 0)]) for x in input_data])

def pad_label(label_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0])]) for x in label_data])


if __name__ == '__main__':

    data = utils.load_json(f'{CUR_DIR}/data.json')      

    train_x, test_x, train_y, test_y = train_test_split(data['embedding'], data['labels'],
                                                        test_size=0.2,
                                                        random_state=0)
    
    train_lengths = jnp.array(list(map(len, train_x)))
    test_lengths = jnp.array(list(map(len, test_x)))
    
    max_len = 90 # TODO: config file...
    
    train_x = pad_input([jnp.array(d) for d in train_x], max_len)
    train_y = pad_label([jnp.array(d) for d in train_y], max_len)
    test_x = pad_input([jnp.array(d) for d in test_x], max_len)
    test_y = pad_label([jnp.array(d) for d in test_y], max_len)
    
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.003 # TODO: config file...
    hidden_size = 256 # TODO: config file...
    batch_size = 4 # TODO: config file...
    num_epochs = 10  # TODO: config file...
    
    state = create_train_state(rng=rng, learning_rate=learning_rate, hidden_size=hidden_size)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, metrics = training.train_epoch(state,
                                              train_x, train_y, train_lengths,
                                              batch_size, epoch, input_rng,
                                              binary_cross_entropy_loss,
                                              compute_metrics)
        train_losses.append(metrics['loss'])
        train_accuracies.append(metrics['accuracy'])
        
        test_loss, test_accuracy = training.eval_model(state,
                                                       test_x, test_y, test_lengths,
                                                       compute_metrics)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'\ttest epoch: {epoch}, loss: {test_loss}, accuracy: {test_accuracy*100:.2f}')

    training.visualize(train_losses, test_losses, 'Loss', 'upper right')
    training.visualize(train_accuracies, test_accuracies, 'Accuracy', 'lower right')

    training.export('segmentation_model_', state.params)
