import functools
import os
import re
import subprocess
from typing import Dict, Iterable, List

from model import utils

CUR_DIR = os.path.dirname(__file__)

def mask_variables(c_code: str) -> str:
    return re.sub(r'\bx\d+\b', 'VAR', c_code)

def mask_numbers(c_code: str) -> str:
    return re.sub(r'(?<!\S)-?\d+\b', 'NUM', c_code)

def mask(c_code: str) -> str:
    return mask_variables(mask_numbers(c_code))

def raw_files(path_to_files: str = f'{CUR_DIR}/raw_data') -> Iterable[str]:
    return map(lambda f: f'{CUR_DIR}/raw_data/{f}',
               os.listdir(path_to_files))

def extract_features(c_file: str, palmtree) -> Dict:
    with open(c_file, 'r') as f:
        c_code = f.read()

    binary = utils.compile(c_file='',
                           flags='-g -O0 -xc -',
                           input=c_code.encode(),
                           output=subprocess.PIPE).stdout
    objdumped = utils.objdump(binary='',
                              flags='-d -l --no-show-raw-insn -M intel',
                              input=binary,
                              output=subprocess.PIPE).stdout.decode()

    main = utils.extract_fun(objdumped=objdumped, fun_name='main')

    asm = list(utils.extract_asm(main))
    labels = list(utils.extract_labels(main))
    embedding = palmtree.encode(asm)
    masked_c = mask(c_code)
    return {'embedding': embedding.tolist(),
            'labels': labels,
            'masked_c': masked_c}


if __name__ == '__main__':    

    files = raw_files()
    palmtree = utils.load_palmtree()

    data = list(map(functools.partial(extract_features,
                                      palmtree=palmtree),
                    files))

    utils.save_as_json(f'{CUR_DIR}/data.json', {k: [d[k] for d in data] for k in data[0]})

