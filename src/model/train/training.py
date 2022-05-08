import functools
import os
from datetime import datetime

import jax
import numpy as np
from flax import serialization
from jax import numpy as jnp
from matplotlib import pyplot as plt

CUR_DIR = os.path.dirname(__file__)

def export(name_prefix, params):
    if not os.path.exists(f"{CUR_DIR}/runs"):
        os.mkdir(f"{CUR_DIR}/runs")
    fname = f"{CUR_DIR}/runs/{name_prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}.params"
    with open(fname, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Saved model as {fname}")

def progress_bar(progress, total):
    bar = '█'
    bar_cnt = int(40 * (progress / float(total)))
    percent = 100 * (progress / float(total))
    dot_cnt = 40 - bar_cnt
    print(f"\r|{bar*bar_cnt}{'.'*dot_cnt}| {percent:.2f}%", end='\r')


@functools.partial(jax.jit, static_argnames=['error_fn', 'metric_fn'])
def train_step(state, batch_x, batch_y, lengths, error_fn, metric_fn):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_x)
        logits = jnp.squeeze(logits)
        loss = jnp.sum(error_fn(logits=logits, labels=batch_y, lengths=lengths))
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = metric_fn(logits=logits, labels=batch_y, lengths=lengths)
    return state, metrics

@functools.partial(jax.jit, static_argnames=['metric_fn'])
def eval_step(state, batch_x, batch_y, lengths, metric_fn):
    logits = state.apply_fn({'params': state.params}, batch_x)
    logits = jnp.squeeze(logits)
    return metric_fn(logits=logits, labels=batch_y, lengths=lengths)

def train_epoch(state, train_x, train_y, train_lengths, batch_size, epoch, rng, error_fn, metric_fn):
    train_ds_size = train_x.shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    progress_bar(0, train_ds_size)
    for i, perm in enumerate(perms):
        batch_x = train_x[perm]
        batch_y = train_y[perm]
        lengths = train_lengths[perm]

        state, metrics = train_step(state, batch_x, batch_y, lengths, error_fn, metric_fn)
        batch_metrics.append(metrics)
        progress_bar((i+1) * batch_size, train_ds_size)


    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    print(f'train epoch: {epoch}, loss: {epoch_metrics_np["loss"]}, accuracy: {epoch_metrics_np["accuracy"]*100:.2f}')
    return state, epoch_metrics_np

def eval_model(state, test_x, test_y, test_lengths, metric_fn):
    metrics = eval_step(state, test_x, test_y, test_lengths, metric_fn)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']



def visualize(train, test, ylabel, legend_loc):
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.show()
