from unittest.mock import patch, Mock
import generate_strategy


class FakeIndicator():
    def __init__(self, name='foo') -> None:
        self.name = name


def test_choose_indicator():
    # use two datatypes to test the recursion
    mockInd = Mock()
    fakeInd = FakeIndicator

    indicators = [mockInd]
    same_class_indicators = dict()
    assert generate_strategy.choose_indicator(
        indicators, 1, same_class_indicators) == mockInd

    same_class_indicators[str(Mock)] = 1
    indicators.extend([Mock(), fakeInd])
    # We already got 1 Mock indicator so this should always return the FakeIndicator
    assert generate_strategy.choose_indicator(
        indicators, 1, same_class_indicators) == fakeInd


@patch("generate_strategy.choose_num_indicators")
@patch("generate_strategy.choose_indicator")
def test_generate(choose_indicator, choose_num_indicators):
    choose_num_indicators.return_value = 3
    indicators = ['ble', 'foo', 'bar', Mock(), Mock()]
    conjunctions = ['AND']
    # base_indicator gets popped off, hence slice from 1
    choose_indicator.side_effect = indicators[1:]

    strategy = generate_strategy.generate('ble', indicators, conjunctions, 3)
    assert strategy == ['ble', 'AND', 'foo', 'AND', 'bar']
