import pandas as pd
from typing import Callable, Dict, Sequence


def get_callback(callback_value: str, indicator):
    CALLBACKS = dict(PRICE_DIVERGENCE_UP=[lambda x: x[indicator] > x['close'], None, None])
    return CALLBACKS[callback_value]

def make_callback(df: pd.DataFrame, callback_value: str, indicator:str):
    callback, args, kwargs = get_callback(callback_value, indicator)
    return df.loc[df.apply(callback, axis=1)]


def load_from_object(strategy: Dict):
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
            _parsed = f"{indicator['indicator']} {indicator['op']} {make_callback(indicator['rel_value'])}"
        parsed.append(_parsed)

    if conjunctions:
        result = ""
        for ix, strat in enumerate(parsed):
            result += strat
            try:
                result += f" {conjunctions[ix]} "
            except IndexError:
                continue
        return result
    return parsed[0]


def query_strategy(df: pd.DataFrame, strategy: Dict):
    query = load_from_object(strategy)
    return df.query(query)
