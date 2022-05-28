import pytest
from model.train import training


@pytest.mark.usefixtures
@pytest.mark.parametrize('test_input,expected', [
    ((0, 100), '\r|........................................| 0.00%\r'),
    ((40, 100), '\r|████████████████........................| 40.00%\r'),
    ((100, 100), '\r|████████████████████████████████████████| 100.00%\r'),
])
def test_print_progress_bar(capture_stdout, test_input, expected):
    training.print_progress_bar(*test_input)
    assert capture_stdout["stdout"] == expected
