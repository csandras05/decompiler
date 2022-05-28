import functools
import os
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax import serialization
from flax.training import train_state
from jax import numpy as jnp
from matplotlib import pyplot as plt  # type: ignore

CUR_DIR = os.path.dirname(__file__)

OPTIMIZERS = {
    'adabelief': optax.adabelief,
    'adafactor': optax.adafactor,
    'adagrad': optax.adagrad,
    'adam': optax.adam,
    'adamw': optax.adamw,
    'fromage': optax.fromage,
    'lamb': optax.lamb,
    'lars': optax.lars,
    'noisy_sgd': optax.noisy_sgd,
    'dpsgd': optax.dpsgd,
    'radam': optax.radam,
    'rmsporp': optax.rmsprop,
    'sgd': optax.sgd,
    'sm3': optax.sm3,
    'yogi': optax.yogi
}

def export(name_prefix: str, params) -> None:
    if not os.path.exists(f"{CUR_DIR}/runs"):
        os.mkdir(f"{CUR_DIR}/runs")
    fname = f"{CUR_DIR}/runs/{name_prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}.params"
    with open(fname, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Saved model as {fname}")

def print_progress_bar(progress: int, total: int) -> None:
    bar = '█'
    bar_cnt = int(40 * (progress / float(total)))
    percent = 100 * (progress / float(total))
    dot_cnt = 40 - bar_cnt
    print(f"\r|{bar*bar_cnt}{'.'*dot_cnt}| {percent:.2f}%", end='\r')


@functools.partial(jax.jit, static_argnames=['error_fn', 'metric_fn'])
def train_step(state: train_state.TrainState,
               batch_x: jnp.ndarray,
               batch_y: jnp.ndarray,
               lengths: jnp.ndarray,
               error_fn: Callable[..., float],
               metric_fn: Callable[..., Dict[str, float]]) -> Tuple[train_state.TrainState, Dict[str, float]]:
    
    def loss_fn(params) -> Tuple[float, jnp.ndarray]:
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
def eval_step(state: train_state.TrainState,
              batch_x: jnp.ndarray,
              batch_y: jnp.ndarray,
              lengths: jnp.ndarray,
              metric_fn: Callable[..., Dict[str, float]]) -> Dict[str, float]:
    
    logits = state.apply_fn({'params': state.params}, batch_x)
    logits = jnp.squeeze(logits)
    return metric_fn(logits=logits, labels=batch_y, lengths=lengths)


def train_epoch(state: train_state.TrainState,
                train_x: jnp.ndarray,
                train_y: jnp.ndarray,
                train_lengths: jnp.ndarray,
                batch_size: int,
                epoch: int,
                rng: jax.random.KeyArray,
                error_fn: Callable[..., float],
                metric_fn: Callable[..., Dict[str, float]]) -> Tuple[train_state.TrainState, Dict[str, float]]:
    
    train_ds_size = train_x.shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    
    print_progress_bar(0, train_ds_size)
    for i, perm in enumerate(perms):
        batch_x = train_x[perm]
        batch_y = train_y[perm]
        lengths = train_lengths[perm]

        state, metrics = train_step(state, batch_x, batch_y, lengths, error_fn, metric_fn)
        batch_metrics.append(metrics)
        print_progress_bar((i+1) * batch_size, train_ds_size)

    print(f'\r{" " * 80}', end='\r')
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    print(f'train epoch: {epoch}, loss: {epoch_metrics_np["loss"]:.5f}, accuracy: {epoch_metrics_np["accuracy"]*100:.2f}')
    return state, epoch_metrics_np


def eval_model(state: train_state.TrainState,
               test_x: jnp.ndarray,
               test_y: jnp.ndarray,
               test_lengths: jnp.ndarray,
               metric_fn: Callable[..., Dict[str, float]]) -> Tuple[float, float]:
    metrics = eval_step(state, test_x, test_y, test_lengths, metric_fn)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']

def plot(train: List[float], test: List[float], ylabel: str, legend_loc: str) -> None:
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)


def visualize(*,
              train_losses: List[float],
              test_losses: List[float],
              train_accuracies: List[float],
              test_accuracies: List[float]) -> None:
    
    plt.figure(figsize=[11, 9]).canvas.set_window_title('Change of loss & accuracy')

    plt.subplot(2, 1, 1)
    plot(train_losses, test_losses, 'Loss', 'upper right')

    plt.subplot(2, 1, 2)
    plot(train_accuracies, test_accuracies, 'Accuracy', 'lower right')
    # plt.savefig(f'{CUR_DIR}/runs/fig.pdf')
    plt.show()
