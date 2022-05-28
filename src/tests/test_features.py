import os

import pytest
from model import utils
from model.train import features

CUR_DIR = os.path.dirname(__file__)


@pytest.mark.parametrize("test_input,expected", [
    ('foo bar', 'foo bar'),
    ('x0', 'VAR'),
    ('foo x19 bar 1', 'foo VAR bar 1'),
    ('x0 x1 x2 x3', 'VAR VAR VAR VAR')
])
def test_mask_variables(test_input, expected):
    assert features.mask_variables(test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
  ('foo bar', 'foo bar'),
  ('0', 'NUM'),
  ('foo -192 bar 1', 'foo NUM bar NUM'),
  ('x0 -1 42 x3 asd35asd', 'x0 NUM NUM x3 asd35asd')  
])
def test_mask_numbers(test_input, expected):
    assert features.mask_numbers(test_input) == expected
    
@pytest.mark.parametrize("test_input,expected", [
    ('foo bar', 'foo bar'),
    ('0', 'NUM'),
    ('x0', 'VAR'),
    ('foo -192 bar x1', 'foo NUM bar VAR'),
    ('x0 -1 42 x3 asd35asd', 'VAR NUM NUM VAR asd35asd')
]) 
def test_mask(test_input, expected):
    assert features.mask(test_input) == expected


def test_extract_features():
    palmtree = utils.load_palmtree()
    data = features.extract_features(f'{CUR_DIR}/test.c', palmtree)
    assert data['masked_c'] == 'int main(){\nvolatile int VAR ;\nVAR = NUM ;\nreturn NUM;}'
    assert data['labels'] == [1, 0, 0, 1, 1, 0, 0]
    assert all(map(lambda x: len(x) == 128, data['embedding']))
