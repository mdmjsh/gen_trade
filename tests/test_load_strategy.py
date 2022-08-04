import pytest
from mock import patch, Mock
import load_strategy

RSI = {
    "indicator": "RSI",
    "absolute": True,
    "op": ">=",
    "abs_value": 70,
    "rel_value": None
}
EMA_1 = {
    "indicator": "EMA",
    "absolute": False,
    "op": ">=",
    "abs_value": None,
    "rel_value": "x"
}
EMA_2 = {
    "indicator": "EMA",
    "absolute": False,
    "op": ">=",
    "abs_value": None,
    "rel_value": "y"
}


@pytest.mark.parametrize("strategy,callback_count,expected",
                         [(dict(indicators=[RSI], conjunctions=[]), 0, 'RSI >= 70'), (dict(indicators=[EMA_1], conjunctions=[]), 1, "EMA >= Z"), (dict(indicators=[RSI, EMA_1, EMA_2], conjunctions=["AND", "OR"]), 2,
                             "RSI >= 70 AND EMA >= Z OR EMA >= Z")]
                         )
@patch("load_strategy.get_callback")
def test_load_strategy(get_callback, strategy, callback_count, expected):
    get_callback.return_value = 'Z'
    strategy = load_strategy.load_from_object(strategy)
    assert get_callback.call_count == callback_count
    assert strategy == expected


@pytest.mark.parametrize("strategy,expected",
                         [
                             (dict(indicators=[RSI], conjunctions=[]), 'RSI >= 70'),
                             (dict(indicators=[RSI, EMA_1, EMA_2], conjunctions=["AND", "OR"]),
                             "RSI >= 70 AND (EMA >= Z OR EMA >= Z)"),
                            (dict(indicators=[RSI, EMA_1, EMA_2], conjunctions=["AND", "NOT OR"]),
                            "RSI >= 70 AND (EMA >= Z NOT OR EMA >= Z)"),
                            (dict(indicators=[RSI, EMA_1, EMA_2, RSI], conjunctions=["AND",  "OR", "AND"]),
                            "RSI >= 70 AND (EMA >= Z OR EMA >= Z AND RSI >= 70)"),
                             (dict(indicators=[RSI, EMA_1, EMA_2, RSI], conjunctions=["AND",  "OR", "OR"]),
                             "RSI >= 70 AND (EMA >= Z OR (EMA >= Z OR RSI >= 70))")
                             ]
                         )
@patch("load_strategy.get_callback")
def test_load_strategy_parenthesised(get_callback, strategy, expected):
    get_callback.return_value = 'EMA >= Z'
    strategy = load_strategy.load_from_object_parenthesised(strategy)
    assert strategy == expected

@patch("load_strategy.get_callback")
def test_get_callback(get_callback):
    callback = Mock()

    # returns callback args, kwargs
    get_callback.return_value = [callback, [1], dict(a=1)]
    load_strategy.get_callback('x')
    callback.assert_called_once_with(1, a=1)
