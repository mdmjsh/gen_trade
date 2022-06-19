import uuid
import sys
import pandas as pd
import logging
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
    df['converted_open_ts'] = pd.to_datetime(df['open_ts'], unit='ms')
    df.index = df['open_ts']
    results = []
    logger = logging.getLogger()
    for strategy in strategies:
        logger.info(f"Querying strategy {strategy}")
        subset = query_strategy(df, strategy)
        res = find_profit_in_window(df, subset, strategy)

        logger.info(f"Strategy got results {res}")
        results.append(res)
    import ipdb
    ipdb.set_trace()
    return res


def find_profit_in_window(df: pd.DataFrame, subset: pd.DataFrame, strategy: dict):
    """ Points 3,4 from `main`
    3. Find the point where the stop loss is breached for each trade
    4. For each of these windows find the highest point (profit)
    """
    results = []
    for _, row in subset.iterrows():
        # stop = subset.query(lambda x: x['close'] <= row['stop_low'])
        try:
            end = row.converted_open_ts.replace(day=row.converted_open_ts.day + 1)
        except ValueError: # day is out of range for month
            try:
                end = row.converted_open_ts.replace(day=1, month=row.converted_open_ts.month + 1)
            except ValueError: # month must be in 1..12
                end = row.converted_open_ts.replace(year=row.converted_open_ts.year + 1, day=1, month=1)

        # Get a day's worth of data since the trade opened
        mask = (df['converted_open_ts'] > row.converted_open_ts) & (
            df['converted_open_ts'] <= end)
        window = df.loc[mask]

        if not window.empty:
            high = window['close'].max()
            low = window['close'].min()
            high_point = window[window['close'] == high]
            low_point = window[window['close'] == low]
            # delta between window open and high point
            max_profit = high - row['open']
            max_loss = low - row['open']

            high_timestamp = pd.Timestamp(
                high_point['converted_open_ts'].values[0])
            high_delta = (high_timestamp - row['converted_open_ts'])
            profit_steps = high_delta.components.hours * 15 + high_delta.components.minutes

            low_timestamp = pd.Timestamp(low_point['converted_open_ts'].values[0])
            low_delta = (high_timestamp - row['converted_open_ts'])
            loss_steps = low_delta.components.hours * 15 + low_delta.components.minutes

            res = dict(id=str(uuid.uuid4()), strategy=strategy,
                                profit=dict(
                                    max=max_profit, timestamp=high_timestamp, n_steps=profit_steps),
                                loss=dict(
                                    max=max_loss, timestamp=low_timestamp, n_steps=loss_steps),
                                open_timestamp=row['converted_open_ts'])

            logger = logging.getLogger()
            logger.info(f"Strategy got results {res}")
            results.append(res)
    return results


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s%(levelname)s:%(message)s',
                        stream=sys.stderr, level=logging.INFO)

    main()
