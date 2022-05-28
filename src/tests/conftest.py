import os
import sys

import jax
import optax
import pytest
from model import utils
from model.train import segmentation_train, translation_train

CUR_DIR = os.path.dirname(__file__)


@pytest.fixture
def capture_stdout(monkeypatch):
    buffer = {"stdout": "", "write_calls": 0}

    def fake_write(s):
        buffer["stdout"] += s
        buffer["write_calls"] += 1

    monkeypatch.setattr(sys.stdout, 'write', fake_write)
    return buffer

@pytest.fixture
def segmentation_train_state():
    state = segmentation_train.create_train_state(rng=jax.random.PRNGKey(0),
                                                  optimizer=optax.adam,
                                                  learning_rate=0.003,
                                                  hidden_size=256,
                                                  max_len=80,
                                                  embedding_size=128)
    return state

@pytest.fixture
def translation_train_state():
    vocab = utils.load_json(f'{CUR_DIR}/vocab.json')
    state = translation_train.create_train_state(rng=jax.random.PRNGKey(0),
                                                 optimizer=optax.adam,
                                                 learning_rate=0.003,
                                                 hidden_size=256,
                                                 vocab=translation_train.Vocab([]),
                                                 max_input_len=40,
                                                 max_output_len=40,
                                                 embedding_size=128)
    return state
