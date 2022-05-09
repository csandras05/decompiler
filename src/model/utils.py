import importlib.util
import json
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Union

import yaml
from jax import numpy as jnp

CUR_DIR = os.path.dirname(__file__)

JsonType = Union[None, int, str, bool, List['JsonType'], Dict[str, 'JsonType']]

def block_printing(func):
    def func_wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        value = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return value

    return func_wrapper

@block_printing
def load_palmtree():
    palmtree_path = f"{CUR_DIR}/third_party/PalmTreeTrained"
    sys.path.insert(0, palmtree_path)
    spec = importlib.util.spec_from_file_location("eval_utils", f"{palmtree_path}/eval_utils.py")
    palmtree_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(palmtree_lib)

    palmtree = palmtree_lib.UsableTransformer(model_path=f"{palmtree_path}/palmtree/transformer.ep19",
                                              vocab_path=f"{palmtree_path}/palmtree/vocab")
    return palmtree


def encode(palmtree, input: List[str]):
    return jnp.array(palmtree.encode(input))

def pad_input(input_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0]), (0, 0)]) for x in input_data])

def pad_label(label_data, max_len):
    return jnp.array([jnp.pad(x, [(0, max_len - x.shape[0])]) for x in label_data])


def compile(*, c_file: str | None, flags: str, input=None, output=None):
    return subprocess.run([f'gcc {flags} {c_file}'], shell=True, input=input, stdout=output)

def objdump(*, binary: str | None, flags: str, input=None, output=None) -> str:
    return subprocess.run([f'objdump {flags} {binary}'], shell=True, input=input, stdout=output)

def extract_fun(*, objdumped: str, fun_name: str) -> str:
    p = re.compile(f'<{fun_name}>:\n(.*?)ret', re.MULTILINE|re.DOTALL)
    m = p.search(objdumped)
    if m is None:
        raise AttributeError
    return m.group(1) + 'ret'

def extract_labels(fun: str) -> str:
    fun_splitted = fun.splitlines()
    for prev_line, line in zip(fun_splitted, fun_splitted[1:]):
        if line.startswith('    '):
            yield 1 if prev_line.startswith('/') else 0

def extract_asm(fun: str) -> Iterable[str]:
    for line in fun.splitlines():
        if line.startswith('    '):
            ins = re.sub(',', ' ', line)
            symbols = re.split('([0-9a-zA-Z]+)', ins)
            symbols = ' '.join(symbols).split()
            filtered_symbols = filter(lambda sym: sym not in ['DWORD', 'PTR', "WORD"], symbols[2:])
            yield ' '.join(filtered_symbols)

def save_as_json(path: str, data: JsonType):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> JsonType:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_yaml(path: str):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data
