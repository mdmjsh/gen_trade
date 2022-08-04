import pandas as pd
import json
"""
Write blank signals relative to signals.json -
absolute signals have to be created by hand.

"""
DF = pd.read_csv('BTCUSDC_indicators.csv')

CATS = ['momentum', 'trend', 'volatility', 'volume']

INDICATORS = (x for x in DF.columns if any([x.startswith(y) for y in CATS]))
RELATIVES = ["PREVIOUS_PERIOD", "MA"]


def gte_signal(ind, _type, rel_value):
    return gen_signal(ind, _type, rel_value, ">=")


def lte_signal(ind, _type, rel_value):
    return gen_signal(ind, _type, rel_value, "<=")


def gen_signal(ind, _type, rel_value, op):
    return {
        "indicator": ind,
        "type": _type,
        "name": f"{ind}_gt",
        "absolute": False,
        "op": op,
        "abs_value": None,
        "rel_value": rel_value
    }


def main():
    signals = []

    for ind in INDICATORS:
        _type = ind.split("_")[0]
        for rel in RELATIVES:
            signals.extend(
                [gte_signal(ind, _type, rel),
                 lte_signal(ind, _type, rel)])
    return signals


if __name__ == "__main__":
    signals = main()

    with open('relative_signals.json', 'w+') as fi:
        json.dump(signals, fi)
