import functools
import os

import jax
import numpy as np
import optax
from flax.training import train_state
from jax import numpy as jnp
from model import utils
from model.flax_models.translation import Seq2seq
from model.train import training
from sklearn.model_selection import train_test_split

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



def pad_input(input_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0]), (0, 0)]) for x in input_data])

def pad_output(output_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0])]) for x in output_data])


def mask_sequences(sequence_batch, lengths):
    return sequence_batch * (
        lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis])
  
def cross_entropy_loss(*, logits, labels, lengths, token_cnt):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=token_cnt)
    xe = jnp.sum(one_hot_labels * logits, axis=-1)
    masked_xe = jnp.mean(mask_sequences(xe, lengths))
    return -masked_xe

def compute_metrics(*, logits, labels, lengths, token_cnt):
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



def create_train_state(rng,
                       learning_rate: float,
                       hidden_size: int,
                       vocab: Vocab,
                       max_input_len: int = 40,
                       max_output_len: int = 40):
  
    model = Seq2seq(hidden_size=hidden_size,
                    vocab_size=vocab.token_cnt,
                    sos_id = vocab.token_to_idx['<SOS>'],
                    max_output_len=max_output_len)
    params = model.init(rng, jnp.ones((1, max_input_len, 128)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



if __name__ == '__main__':

    data = utils.load_json(f'{CUR_DIR}/data.json')      

    indices = [[i for i, x in enumerate(l) if x == 1] for l in data['labels']]
    input = [embedding[cur:nxt]
             for (embedding, i) in zip(data['embedding'], indices)
             for (cur, nxt) in zip(i[1:], i[2:])]  
    output = sum([masked_c.splitlines()[2:-1] for masked_c in data['masked_c']], [])
    
    train_x, test_x, train_y, test_y = train_test_split(input, output,
                                                        test_size=0.2,
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
    
    max_len=40
    train_x_padded = pad_input([jnp.array(d) for d in train_x], max_len)
    train_y_padded = pad_output([jnp.array(d) for d in train_y_token_ids], max_len)
    test_x_padded = pad_input([jnp.array(d) for d in test_x], max_len)
    test_y_padded = pad_output([jnp.array(d) for d in test_y_token_ids], max_len)  

    rng = jax.random.PRNGKey(0)
    learning_rate = 0.003
    num_epochs = 10
    batch_size = 4
    hidden_size = 256

    state = create_train_state(rng, learning_rate, hidden_size, vocab)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, metrics = training.train_epoch(state,
                                              train_x_padded, train_y_padded, train_lengths,
                                              batch_size, epoch, input_rng,
                                              functools.partial(cross_entropy_loss, token_cnt=vocab.token_cnt),
                                              functools.partial(compute_metrics, token_cnt=vocab.token_cnt))
        train_losses.append(metrics['loss'])
        train_accuracies.append(metrics['accuracy'])
        
        test_loss, test_accuracy = training.eval_model(state,
                                                       test_x_padded, test_y_padded, test_lengths,
                                                       functools.partial(compute_metrics, token_cnt=vocab.token_cnt))
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'\ttest epoch: {epoch}, loss: {test_loss}, accuracy: {test_accuracy*100:.2f}')

    training.visualize(train_losses, test_losses, 'Loss', 'upper right')
    training.visualize(train_accuracies, test_accuracies, 'Accuracy', 'lower right')

    training.export('translation_model_', state.params)