
import random
from typing import Optional, Sequence

"""
Using these building blocks I can create a set of all  indicators in the study.
This can then be used to randomly generate a population.

indicators = [i1, i2..in]

"""

CONJUNTIONS = ['AND', 'OR', 'NOT']

class Indicator():
    def __init__(self, name, symbol, threshold) -> None:
        self.name = name
        self.symbol = symbol
        self.threshold = threshold

    def __repr__(self):
        return " ".join([self.name, self.symbol, str(self.threshold)])


class Strategy():
    def __init__(self, indicator: Indicator, conjunction: Optional[str]=None, co_strategy: 'Strategy'=None):
        self.indicator = indicator

        if conjunction and not co_strategy:
            raise ValueError(f"You cannot create a conjunctive strategy without co_strategy")

        if co_strategy and not conjunction:
            raise ValueError(f"You cannot create a co_strategy strategy without conjunction")

        self.conjunction = conjunction or ''
        self.co_strategy = co_strategy or ''

    def __repr__(self):
        return " ".join([str(self.indicator), self.conjunction, str(self.co_strategy)])


def generate_strategy(indicators: Sequence[Indicator], max: int):
    n_indicators = random.choice(1, max+1) # plus 1 as range is not inclusive
    _indicators = [indicators[random.choice(range(len(indicators)))] for _ in n_indicators]

    conjunctions = [CONJUNTIONS[random.choice(range(len(CONJUNTIONS)))] for _ in (n_indicators -1)]

    # TODO sensibly generate a strategy from all of this.
    return Strategy(_indicators)