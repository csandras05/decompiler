from model.model import Model


def test_get_masked_c():
    model = Model()
    model.masked_c = ['1', '2']
    assert model.get_masked_c() == '1\n2'
