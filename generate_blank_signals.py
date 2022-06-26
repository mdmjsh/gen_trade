import pandas as pd
import json

"""
Write blank signals to signals.json so that the values can be manually completed.

"""
df = pd.read_csv('BTCUSDC_indicators.csv')

cats = ['momentum', 'trend', 'volatility_bbh', 'volume']

indicators = (x for x in df.columns if any([x.startswith(y) for y in cats]))

with open('signals.json', 'r') as fi:
    signals = json.load(fi)

for ind in indicators:
    _type = ind.split("_")[0]
    signals.extend([
        {
            "indicator": ind,
            "type": _type,
            "name": f"{ind}_gt",
            "absolute": None,
            "op": ">=",
            "abs_value": 0,
            "rel_value": None
        },
        {
            "indicator": ind,
            "type": _type,
            "name": f"{ind}_lt",
            "absolute": None,
            "op": "<=",
            "abs_value": 0,
            "rel_value": None
        }
    ])

with open('signals.json', 'w+') as fi:
    json.dump(signals, fi)