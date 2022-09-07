import logging
from copy import copy
from df_adapter import DFAdapter
from typing import Dict

from logger import get_logger

dfa = DFAdapter()

def get_callback(indicator: Dict) -> str:
    """Returns the relative string for the indicator passed in.
    """
    # {'ADX_NEG',
    # 'AROON_DOWN',
    # 'AROON_UP',
    # 'MA',
    # 'PPO_SIGNAL',
    # 'PREVIOUS_PERIOD',
    # 'STOCH_RSI_D',
    # 'STOCH_RSI_K'}

    if indicator['rel_value'] == 'MA':
        # trend_sma_slow / trend_sma_fast available
        return f"{indicator['indicator']} {indicator['op']} trend_sma_slow"
    elif indicator['rel_value'] == 'PREVIOUS_PERIOD':
        return f"{indicator['indicator']} {indicator['op']} {indicator['indicator']}_previous"
    else:
        return f"{indicator['indicator']} {indicator['op']} {indicator['rel_value'].lower()}"


def load_from_object(strategy: Dict) -> str:
    """Original un-parenthesised version"""

    parsed = []

    conjunctions = strategy.get('conjunctions', [])

    if len(conjunctions) != len(strategy['indicators']) - 1:
        raise RuntimeError(
            f"Strategy does not have correct number of conjunctions! {strategy}"
        )

    for indicator in strategy['indicators']:
        if indicator["absolute"]:
            _parsed = f"{indicator['indicator']} {indicator['op']} {indicator['abs_value']}"
        else:
            # NB Need to pass DF in here somehow...
            _parsed = get_callback(indicator)
        parsed.append(_parsed)

    if conjunctions:
        result = ""
        for ix, strat in enumerate(parsed):
            # try:
            result += strat
            try:
                result += f" {conjunctions[ix]} "
            except IndexError:
                continue
        return result
    return parsed[0]


def load_from_object_parenthesised(strategy: Dict) -> str:
    """parenthesis disjunctions to ensure strategy quality"""
    parsed = []

    conjunctions = strategy.get('conjunctions', [])

    if len(conjunctions) != len(strategy['indicators']) - 1:
        raise RuntimeError(
            f"Strategy does not have correct number of conjunctions! {strategy}"
        )

    for indicator in strategy['indicators']:
        if indicator["absolute"]:
            _parsed = f"{indicator['indicator']} {indicator['op']} {indicator['abs_value']}"
        else:
            # NB Need to pass DF in here somehow...
            _parsed = get_callback(indicator)
        parsed.append(_parsed)

    if conjunctions:
        result = ""
        open_paren_count = 0
        for ix, strat in enumerate(parsed):
            _strat = copy(strat)
            try:
                conj = conjunctions[ix]
                if 'or' in conj.lower():
                    _strat = f"({_strat} {conj}"
                else:
                    _strat = f"{_strat} {conj}"
                open_paren_count += int(_strat[0] == '(' and _strat[-1] != ')')
                result = f"{result} {_strat}"
            except IndexError:
                result = f"{result} {_strat}"
                if open_paren_count:
                    result += ')' * open_paren_count
        # remove additional whitespace
        res = result[1:]
        logger = get_logger(__name__)
        logger.info(f"Generated strategy: {res}")
        return res
    return parsed[0]


def query_strategy(df: dfa.DataFrame, strategy: Dict = None, query: str = None) -> dfa.DataFrame:
    if query:
        return df.query(query)
    query = load_from_object_parenthesised(strategy)
    try:
        return df.query(query).compute()
    except AttributeError:
        return df.query(query)
