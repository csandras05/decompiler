import pytest
from model.reconstruct import reconstruct


@pytest.mark.parametrize("test_input,expected", [
    ('0x0', '0'),
    ('0x2a', '42'),
    ('0xffed', '-19'),
    ('0x1', '1'),
    ('0xffff', '-1')
])
def test_s16(test_input, expected):
    assert reconstruct.s16(test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
    (('foo bar', {}), []),
    (('[ rbp - 0x0 ]', {}), ['x0']),
    (('asd ret [ rbp - 0x0 ] 0x17', {'[ rbp - 0x0 ]': 0}), ['x0']),
    (('[ rbp - 0x1 ]', {'[ rbp - 0x0 ]': 0}), ['x1']),
    (('[ rbp - 0x1 ] mov [ rbp - 0x0 ]', {'[ rbp - 0x0 ]': 0}), ['x1', 'x0'])
])
def test_extract_vars(test_input, expected):
    assert reconstruct.extract_vars(*test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
  ('foo bar', []),
  ('0x0', ['0']),
  ('foo 0xff40 bar 0x1', ['-192', '1']),
  ('mov 0xffff 0x2a add ret', ['-1', '42'])  
])
def test_extract_nums(test_input, expected):
    assert reconstruct.extract_nums(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [
    ((['mov [ rbp - 0x0 ] 0x2a'], ['mov [ rbp - 0x0 ] 0x2a']), True),
    ((['mov'], ['mov', 'ret']), False),
    ((['mov [ rbp - 0x0 ] 0x2a', '[ rbp - 0x0 ]'], ['mov [ rbp - 0x2 ] 0x2a', '[ rbp - 0x0 ]']), False),
    ((['mov [ rbp - 0x0 ] 0x2a'], ['mov [ rbp - 0x0 ] 0xa2']), False)
])
def test_compare(test_input, expected):
    assert reconstruct.compare(*test_input) == expected
    
def test_compile_and_get_blocks():
    c_code = 'int main(){\n' +\
             'volatile int ' + ' , '.join('x' + str(i) for i in range(20)) + ' ;\n' +\
             'x0 = 42 ;\n' +\
             'return 0;}'  
    expected = ['mov [ rbp - 0x0 ] 0x2a']
    assert reconstruct.compare(reconstruct.compile_and_get_blocks(c_code), expected)
    
def test_retrieve():
    masked_c = ['VAR = NUM ;']
    asm_blocks = ['mov [ rbp - 0x0 ] 0x2a']
    expected = 'int main(){\n' +\
               'volatile int ' + ' , '.join('x' + str(i) for i in range(20)) + ' ;\n' +\
               'x0 = 42 ;\n' +\
               'return 0;}'
    assert reconstruct.retrieve(masked_c, asm_blocks) == expected
