import pandas as pd
from generate_strategy import main as generate
from load_strategy import query_strategy

def main():
    """ Logic:

    1. Subset the dataframe by the strategy (for each strategy)
    2. Set the stop loss to 90% buy price for each trade
    3. Find the point where the stop loss is breached for each trade
    4. For each of these windows find the highest point (profit)
    5. calculate the average of these profits points
    """

    strategies = generate()
    df = pd.read_csv('BTCUSDC_indicators.csv')
    df.index = df['open_ts']
    for strat in strategies:
        import ipdb;ipdb.set_trace()
        subset = query_strategy(df, strat)
        # Add stop loss and profit targets to trade
        subset['stop_loss'] = subset.apply(lambda x: x['open'] * 0.98, axis=1)
        # Find the point where the stop loss is hit.
        find_profit_in_window(df, subset)


def find_profit_in_window(df:pd.DataFrame, subset: pd.DataFrame):
    """ Points 3,4 from `main`
    3. Find the point where the stop loss is breached for each trade
    4. For each of these windows find the highest point (profit)
    """
    for row in subset:
        # stop = subset.query(lambda x: x['close'] <= row['stop_low'])
        # Get a day's worth of data since the trade opened - nb not working yet!
        window = df[(df['open_ts'] <= row['open_ts'] + (60 *24)) & (df['open_ts'] >= row['open_ts'])]
        # find the highest point in that window
        high = window['close'].max()
        window[window['close'] == high]
        # delta between window open and high point
        row['open_ts'] - window[window['close'] == high].index
        profit = high - row['open']


if __name__ == "__main__":
    main()
