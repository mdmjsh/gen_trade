import pytest
from mock import patch, Mock
import load_strategy

RSI = {
    "name": "RSI",
    "absolute": True,
    "op": ">=",
    "abs_value": 70,
    "rel_value": None
}
EMA_1 = {
    "name": "EMA",
    "absolute": False,
    "op": ">=",
    "abs_value": None,
    "rel_value": "x"
}
EMA_2 = {
    "name": "EMA",
    "absolute": False,
    "op": ">=",
    "abs_value": None,
    "rel_value": "y"
}


@pytest.mark.parametrize("strategy,callback_count,expected",
                         [(dict(indicators=[RSI], conjunctions=[]), 0, 'RSI >= 70'), (dict(indicators=[EMA_1], conjunctions=[]), 1, "EMA >= Z"), (dict(indicators=[RSI, EMA_1, EMA_2], conjunctions=["AND", "OR"]), 2,
                             "RSI >= 70 AND EMA >= Z OR EMA >= Z")]
                         )
@patch("load_strategy.make_callback")
def test_load_strategy(make_callback, strategy, callback_count, expected):
    make_callback.return_value = 'Z'
    strategy = load_strategy.load_from_object(strategy)
    assert make_callback.call_count == callback_count
    assert strategy == expected

@patch("load_strategy.get_callback")
def test_make_callback(get_callback):
    callback = Mock()

    # returns callbackm args, kwargs
    get_callback.return_value = [callback, [1], dict(a=1)]
    load_strategy.make_callback('x')
    callback.assert_called_once_with(1, a=1)
