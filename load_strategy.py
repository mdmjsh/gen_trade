import pandas as pd
from typing import Callable, Dict, Sequence


# def make_callback(df: pd.DataFrame, indicator: Dict):
#     import ipdb;ipdb.set_trace()
#     callback, args, kwargs = get_callback(indicator)
#     return df.loc[df.apply(callback, axis=1)]


def get_callback(indicator : Dict) -> str:
    """Returns the relative string for the indicator passed in.
    """
    {'ADX_NEG',
    'AROON_DOWN',
    'AROON_UP',
    'MA',
    'PPO_SIGNAL',
    'PREVIOUS_PERIOD',
    'STOCH_RSI_D',
    'STOCH_RSI_K'}

    if indicator['rel_value'] == 'MA':
        # trend_sma_slow / trend_sma_fast available
        return f"{indicator['indicator']} {indicator['op']} trend_sma_slow"
    elif indicator['rel_value'] == 'PREVIOUS_PERIOD':
        return f"{indicator['indicator']} {indicator['op']} {indicator['indicator']}_previous"
    else:
        return f"{indicator['indicator']} {indicator['op']} {indicator['indicator']}_previous"


def load_from_object(df: pd.DataFrame, strategy: Dict):

    parsed = []
    print(f"STRATEGY: {strategy}")
    conjunctions = strategy['conjunctions']

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
            try:
                result += strat
            except TypeError:
                import ipdb;ipdb.set_trace()
            try:
                result += f" {conjunctions[ix]} "
            except IndexError:
                continue
        return result
    return parsed[0]


def query_strategy(df: pd.DataFrame, strategy: Dict):
    query = load_from_object(df, strategy)
    return df.query(query)
