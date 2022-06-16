import pandas as pd
from generate_strategy import main as generate
from load_strategy import query_strategy

def main():
    strategies = generate()
    df = pd.read_csv('BTCUSDC_indicators.csv')
    for strat in strategies:
        import ipdb;ipdb.set_trace()
        subset = query_strategy(df, strat)

if __name__ == "__main__":
    main()
