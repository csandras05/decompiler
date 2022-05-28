import functools
import os
from typing import Callable, Dict, List

import jax
import numpy as np
from flax.training import train_state
from jax import numpy as jnp
from model import utils
from model.flax_models.translation import Seq2seq
from model.train import training
from sklearn.model_selection import train_test_split  # type: ignore

CUR_DIR = os.path.dirname(__file__)


class Vocab:
    token_cnt = 2
    token_to_idx = {'<PAD>': 0,
                    '<SOS>': 1}
    idx_to_token = {0: '<PAD>',
                    1: '<SOS>'}

    def __init__(self, train_y):
        for token in (' '.join(train_y)).split():
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.token_cnt
                self.idx_to_token[self.token_cnt] = token
                self.token_cnt += 1

    def idx_seq_to_c_line(self, idx_seq):
        tokens = map(lambda x: self.idx_to_token[int(x)], idx_seq)
        return ' '.join(tokens)



def pad_input(input_data: List[jnp.ndarray], max_len: int) -> jnp.ndarray:
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0]), (0, 0)])
                      if max_len > x.shape[0] else x[:max_len] for x in input_data])

def pad_output(output_data: List[jnp.ndarray], max_len: int) -> jnp.ndarray:
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0])])
                      if max_len > x.shape[0] else x[:max_len] for x in output_data])


def mask_sequences(sequence_batch: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    return sequence_batch * (
        lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis])
  
  
def cross_entropy_loss(*,
                       logits: jnp.ndarray,
                       labels: jnp.ndarray,
                       lengths: jnp.ndarray,
                       token_cnt: int) -> float:
    
    one_hot_labels = jax.nn.one_hot(labels, num_classes=token_cnt)
    xe = jnp.sum(one_hot_labels * logits, axis=-1)
    masked_xe = jnp.mean(mask_sequences(xe, lengths))
    return -masked_xe


def compute_metrics(*,
                    logits: jnp.ndarray,
                    labels: jnp.ndarray,
                    lengths: jnp.ndarray,
                    token_cnt: int) -> Dict[str, float]:
    
    loss = cross_entropy_loss(logits=logits, labels=labels, lengths=lengths, token_cnt=token_cnt)

    token_accuracy = jnp.argmax(logits, -1) == labels
    sequence_accuracy = (
        jnp.sum(mask_sequences(token_accuracy, lengths), axis=-1) == lengths
    )
    accuracy = jnp.mean(sequence_accuracy)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics



def create_train_state(rng: jax.random.KeyArray,
                       optimizer: Callable,
                       learning_rate: float,
                       hidden_size: int,
                       vocab: Vocab,
                       max_input_len: int,
                       max_output_len: int,
                       embedding_size: int) -> train_state.TrainState:
  
    model = Seq2seq(hidden_size=hidden_size,
                    vocab_size=vocab.token_cnt,
                    sos_id = vocab.token_to_idx['<SOS>'],
                    max_output_len=max_output_len)
    
    params = model.init(rng, jnp.ones((1, max_input_len, embedding_size)))['params']
    tx = optimizer(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



if __name__ == '__main__':

    data = utils.load_json(f'{CUR_DIR}/data.json')   
    config = utils.load_yaml(f'{CUR_DIR}/translation_train.yaml')   

    indices = [[i for i, x in enumerate(l) if x == 1] for l in data['labels']]
    input = [embedding[cur:nxt]
             for (embedding, i) in zip(data['embedding'], indices)
             for (cur, nxt) in zip(i[1:], i[2:])]  
    output: List[List[str]] = sum([masked_c.splitlines()[2:-1] for masked_c in data['masked_c']], [])
    
    train_x, test_x, train_y, test_y = train_test_split(input, output,
                                                        test_size=config['test_ratio'],
                                                        random_state=0)
    vocab = Vocab(train_y)
    
    train_y_token_ids = []
    for c_line in train_y:
        token_ids = list(map(lambda x: vocab.token_to_idx[x], c_line.split()))
        train_y_token_ids.append(jnp.array([vocab.token_to_idx['<SOS>']] + token_ids))

    test_y_token_ids = []
    for c_line in test_y:
        token_ids = list(map(lambda x: vocab.token_to_idx[x], c_line.split()))
        test_y_token_ids.append(jnp.array([vocab.token_to_idx['<SOS>']] + token_ids))
    
    train_lengths = jnp.array(list(map(lambda x: x.shape[0], train_y_token_ids)))
    test_lengths = jnp.array(list(map(lambda x: x.shape[0], test_y_token_ids)))
   
    
    train_x_padded = pad_input([jnp.array(d) for d in train_x], config['max_input_len'])
    train_y_padded = pad_output([jnp.array(d) for d in train_y_token_ids], config['max_output_len'])
    test_x_padded = pad_input([jnp.array(d) for d in test_x], config['max_input_len'])
    test_y_padded = pad_output([jnp.array(d) for d in test_y_token_ids], config['max_output_len'])  

    rng = jax.random.PRNGKey(0)

    state = create_train_state(rng=rng,
                               optimizer=training.OPTIMIZERS[config['optimizer']],
                               learning_rate=config['learning_rate'],
                               hidden_size=config['hidden_size'],
                               vocab=vocab,
                               max_input_len=config['max_input_len'],
                               max_output_len=config['max_output_len'],
                               embedding_size=config['embedding_size'])
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    metric_fn = functools.partial(compute_metrics, token_cnt=vocab.token_cnt)
    error_fn = functools.partial(cross_entropy_loss, token_cnt=vocab.token_cnt)
    
    for epoch in range(1, config['num_epochs'] + 1):
        rng, input_rng = jax.random.split(rng)
        state, metrics = training.train_epoch(state,
                                              train_x_padded, train_y_padded, train_lengths,
                                              config['batch_size'], epoch, input_rng,
                                              error_fn,
                                              metric_fn)
        train_losses.append(metrics['loss'])
        train_accuracies.append(metrics['accuracy'])
        
        test_loss, test_accuracy = training.eval_model(state,
                                                       test_x_padded, test_y_padded, test_lengths,
                                                       functools.partial(compute_metrics, token_cnt=vocab.token_cnt))
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'\ttest epoch: {epoch}, loss: {test_loss:.5f}, accuracy: {test_accuracy*100:.2f}')

    training.visualize(train_losses=train_losses, test_losses=test_losses,
                       train_accuracies=train_accuracies, test_accuracies=test_accuracies)

    training.export('translation_model_', state.params)
