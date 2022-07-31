from copy import copy
import pandas as pd
from typing import Dict


# def make_callback(df: pd.DataFrame, indicator: Dict):
#     import ipdb;ipdb.set_trace()
#     callback, args, kwargs = get_callback(indicator)
#     return df.loc[df.apply(callback, axis=1)]


def get_callback(indicator : Dict) -> str:
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
    print(f"STRATEGY: {strategy}")

    conjunctions = strategy.get('conjunctions', [])

    if len(conjunctions) != len(strategy['indicators']) -1:
        raise RuntimeError(f"Strategy does not have correct number of conjunctions! {strategy}")


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
    """parenthesis disjunctives to ensure strategy quality"""
    parsed = []
    print(f"STRATEGY: {strategy}")

    conjunctions = strategy.get('conjunctions', [])

    if len(conjunctions) != len(strategy['indicators']) -1:
        raise RuntimeError(f"Strategy does not have correct number of conjunctions! {strategy}")


    for indicator in strategy['indicators']:
        if indicator["absolute"]:
            _parsed = f"{indicator['indicator']} {indicator['op']} {indicator['abs_value']}"
        else:
            # NB Need to pass DF in here somehow...
            _parsed = get_callback(indicator)
        parsed.append(_parsed)

    if conjunctions:
        result = ""
        requires_closing = False
        for ix, strat in enumerate(parsed):

            _strat = copy(strat)
            try:
                conj = conjunctions[ix]
                if requires_closing:
                    _strat = f"{_strat})"
                if 'or' in conj.lower():
                    _strat = f"({_strat} {conj}"
                else:
                    _strat = f"{_strat} {conj}"
                requires_closing = _strat[0] == '(' and _strat[-1] != ')'
                result = f"{result} {_strat}"
            except IndexError:
                if requires_closing:
                    result = f"{result} {_strat})"
                else:
                    result =  f"{result} {_strat}"
        return result[1:] # remove additional whitespace
    return parsed[0]


def query_strategy(df: pd.DataFrame, strategy: Dict):

    query = load_from_object_parenthesised(strategy)
    if ' or ' in query:
        import ipdb;ipdb.set_trace()

    return df.query(query)
