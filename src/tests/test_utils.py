import os

from model import utils

CUR_DIR = os.path.dirname(__file__)

def test_extract_fun():
    with open(f'{CUR_DIR}/objdumped.txt', 'r') as f:
        objdumped = f.read()
    with open(f'{CUR_DIR}/main.txt', 'r') as f:
        main = f.read()
    print(utils.extract_fun(objdumped=objdumped, fun_name='main'))
    assert utils.extract_fun(objdumped=objdumped, fun_name='main').split() == main.split()

def test_extract_asm():
    with open(f'{CUR_DIR}/main.txt', 'r') as f:
        main = f.read()
    with open(f'{CUR_DIR}/asm.txt', 'r') as f:
        asm = f.read()
    assert '\n'.join(utils.extract_asm(main)).split() == asm.split()

def test_extract_labels():
    with open(f'{CUR_DIR}/main.txt', 'r') as f:
        main = f.read()
    labels = [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert list(utils.extract_labels(main)) == labels

def test_save_load_json():
    d = {'foo': 42, 'bar': {'baz': "Hello", 'poo': [124.2, 42.2]}}
    utils.save_as_json('tmp.json', d)
    d2 = utils.load_json('tmp.json')
    os.remove('tmp.json')
    assert d == d2
