from typing import List
import uuid
from numpy import datetime64
import arrow
import sys
import pandas as pd
import logging
from generate_strategy import main as generate
from load_strategy import query_strategy
from memory_profiler import profile
from async_caller import process_future_caller, threaded_future_caller

df = pd.read_csv('BTCUSDC_indicators.csv')

@profile
def main():
    """ Logic:

    1. Subset the dataframe by the strategy (for each strategy)
    2. For each strategy:
        3. Set the trade period to one day.
        4. For each of these windows find the highest point (profit)
    5. calculate the average of these profits points
    """

    strategies = generate()
    df['converted_open_ts'] = pd.to_datetime(df['open_ts'], unit='ms')
    df.index = df['open_ts']
    # import ipdb;ipdb.set_trace()
    res = threaded_future_caller(run_strategy, strategies[:2])

    return res

def run_strategy(strategy):
    logger = logging.getLogger()
    logger.info(f"Querying strategy {strategy}")
    subset = query_strategy(df, strategy)
    res = find_profit_in_window(df, subset, strategy)
    logger.info(f"Strategy got results {res}")
    return pd.DataFrame(res)

def find_profit_in_window(df: pd.DataFrame, subset: pd.DataFrame, strategy: dict) -> List[dict]:
    """ Points 3,4 from `main`
    3. Set the trade period to one day.
    4. For each of these windows find the highest point (profit)

    NB that the subset passed in is all of trades with would execute for the entire dataset (i.e. 4 years worth of trades).
    """
    logger = logging.getLogger()
    results = []
    strat_id = str(uuid.uuid4())
    for _, row in subset.iterrows():
        # Get a period of data since the trade opened
        end = shift_period(row['converted_open_ts'])
        mask = (df['converted_open_ts'] > row.converted_open_ts) & (
            df['converted_open_ts'] <= end)
        window = df.loc[mask]

        if not window.empty:
            high = window['close'].max()
            low = window['close'].min()

            # edge case: multiple rows can close with the same value, hence locate the first (earliest) occurence
            high_point = window.loc[window['open_ts'] ==
                                    window[window['close'] == high].index.min()]
            low_point = window.loc[window['open_ts'] ==
                                   window[window['close'] == low].index.min()]
            # delta between window open and high point
            max_profit = high - row['open']
            max_loss = low - row['open']

            high_timestamp = pd.Timestamp(
                high_point['converted_open_ts'].values[0])
            high_delta = high_timestamp - row['converted_open_ts']
            profit_steps = high_delta.components.hours * 15 + high_delta.components.minutes

            low_timestamp = pd.Timestamp(
                low_point['converted_open_ts'].values[0])
            low_delta = (low_timestamp - row['converted_open_ts'])
            loss_steps = low_delta.components.hours * 15 + low_delta.components.minutes

            res = dict(id=strat_id, strategy=strategy, trend=row.trend_direction,
                       profit=dict(
                           max=max_profit, timestamp=high_timestamp, n_steps=profit_steps),
                       loss=dict(
                           max=max_loss, timestamp=low_timestamp, n_steps=loss_steps),
                       open_timestamp=row['converted_open_ts'], profitable=is_profitable(
                           max_profit, max_loss, high_timestamp, low_timestamp),
                       loss_making=is_loss_making(max_profit, max_loss, high_timestamp, low_timestamp))
            logger.info(f"Strategy {strat_id} got results {res}")
            results.append(res)
    return results


def shift_period(timestamp: pd.Timestamp) -> datetime64:
    """Shift the timestamp by 1 day - 1 ms"""
    ts = arrow.get(timestamp.isoformat())
    return datetime64(ts.shift(days=1).shift(microseconds=-1).isoformat())

def is_profitable(max_profit: float, max_loss: float, high_timestamp: pd.Timestamp, low_timestamp: pd.Timestamp) -> bool:
    """Is the profitable high point before the max loss occurred?"""
    if max_loss < 0:
        return (max_profit > 0) and (high_timestamp < low_timestamp)
    else:
        return max_profit > 0


def is_loss_making(max_profit: float, max_loss: float, high_timestamp: pd.Timestamp, low_timestamp: pd.Timestamp) -> bool:
    """Does the max loss occur before the max profit was made?"""
    if max_profit <= 0:
        return (max_loss < 0) and (low_timestamp < high_timestamp)
    else:
        return max_loss < 0


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s%(levelname)s:%(message)s',
                        stream=sys.stderr, level=logging.INFO)

    results = main()
