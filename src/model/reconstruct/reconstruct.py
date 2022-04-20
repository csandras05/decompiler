import copy
import json
import os
import random
import re
import subprocess
from itertools import permutations
from typing import Dict, Iterable, List, Tuple

from model import utils


def s16(hex_str: str) -> str:
    value = int(hex_str, 16)
    return str(-(value & 0x8000) | (value & 0x7fff))

def compile_and_get_blocks(c_code: str) -> List[List[str]]:
    binary = utils.compile(c_file='',
                           flags='-g -O0 -xc -',
                           input=c_code.encode(),
                           output=subprocess.PIPE).stdout
    objdumped = utils.objdump(binary='',
                              flags='-S -l --no-show-raw-insn -M intel',
                              input=binary,
                              output=subprocess.PIPE).stdout.decode()

    main = utils.extract_fun(objdumped=objdumped, fun_name='main')

    asm_sample = list(utils.extract_asm(main))
    labels_sample = list(utils.extract_labels(main))
    
    indices = [i for i, x in enumerate(labels_sample) if x == 1]
    indices.append(len(labels_sample))

    blocks = ['\n'.join(asm_sample[cur:nxt]) for (cur, nxt) in zip(indices, indices[1:])]
    return blocks[1:-1]

def load_magic() -> Tuple[Dict[str, List[int]]]:
    dir = os.path.dirname(__file__)
    with open(f'{dir}/div_magic.json', 'r') as f:
        div_magic = json.load(f)
    with open(f'{dir}/mod_magic.json', 'r') as f:
        mod_magic = json.load(f)
    with open(f'{dir}/mul_magic.json', 'r') as f:
        mul_magic = json.load(f)
    return div_magic, mod_magic, mul_magic

def extract_nums(asm_block: str) -> List[str]:
    tmp = re.sub('\[ rbp - 0x[0-9a-f]+ \]', 'REG', asm_block)
    p_num = re.compile('0x[0-9a-f]+')
    return list(map(s16, p_num.findall(tmp)))

def extract_vars(asm_block: str, d: Dict[str, str]) -> List[str]:
    p_reg = re.compile('\[ rbp - 0x[0-9a-f]+ \]')
    ret = []
    cnt = len(d)
    for r in p_reg.findall(asm_block):
        if r not in d:
            d[r] = cnt
            cnt += 1
        ret.append(f'x{d[r]}')
    return ret

def extra_nums(nums: List[str], magic: Dict[str, List[str]]) -> Iterable[str]:
    for (k, v) in magic.items():
        if all(map(lambda x: x in nums, v)):
            yield str(k)

def compare(retrieved_blocks: List[List[str]], orig_blocks: List[List[str]]) -> bool:
    orig_blocks = '\n'.join(orig_blocks)
    retrieved_blocks = '\n'.join(retrieved_blocks)
    
    retrieved_vars = extract_vars(retrieved_blocks, {})
    orig_vars = extract_vars(orig_blocks, {})
    
    retrieved_nums = extract_nums(retrieved_blocks)
    orig_nums = extract_nums(orig_blocks)
    
    return len(orig_blocks.split()) ==  len(retrieved_blocks.split()) and\
           retrieved_vars == orig_vars and\
           retrieved_nums == orig_nums
           

def retrieve(masked_c, asm_blocks):
    div_magic, mod_magic, mul_magic = load_magic()
        
    retrieved = 'int main(){\n' + "volatile int " + " , ".join("x" + str(i) for i in range(20)) + " ;\n"
    d = {}
    for i, (masked_c_line, asm_block) in enumerate(zip(masked_c, asm_blocks)):
        nums = extract_nums(asm_block)
        
        tmp = copy.deepcopy((nums))
        
        if '/' in masked_c_line.split():
            for n in extra_nums(tmp, div_magic):
                nums.append(n)
        if '%' in masked_c_line.split():
            for n in extra_nums(tmp, mod_magic):
                nums.append(n)
        if '*' in masked_c_line.split():
            for n in extra_nums(tmp, mul_magic):
                nums.append(n)
        
        vars = extract_vars(asm_block, d)
        n = len(re.findall('NUM', masked_c_line))
        masked_c_line = re.sub('VAR', vars[-1], masked_c_line, count=1)
        found = False
        for var_perm in permutations(vars[:-1]):
            for num_perm in permutations(nums, n):
                if found:
                    break
                sample_line = []
                num_cnt = 0
                var_cnt = 0
                for token in masked_c_line.split():
                    if token == 'NUM':
                        token = num_perm[num_cnt]
                        num_cnt += 1
                    if token == 'VAR':
                        token = var_perm[var_cnt]
                        var_cnt += 1
                    sample_line.append(token)
                sample_line = ' '.join(sample_line)
                tmp = retrieved + sample_line + '\nreturn 0;}'
                retrieved_asm_blocks = compile_and_get_blocks(tmp)
                if compare(retrieved_asm_blocks, asm_blocks[:(i+1)]):
                    retrieved = retrieved + sample_line + '\n'
                    found = True
                    
        if not found:
            print("Hmm...")
            return

    retrieved = retrieved + 'return 0;}'
    return retrieved
