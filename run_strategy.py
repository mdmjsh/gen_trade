import pandas as pd
from typing import Dict, List
import logging
from load_strategy import query_strategy
from numpy import datetime64
import arrow

# risk:reward ratio 2:1
TARGET = 0.015
STOP_LOSS = TARGET / 2

HIT = 'TARGET_HIT'
STOPPED = 'STOPPED_OUT'
NA = 'NO_CLOSE_IN_WINDOW'
NO_TRADES = 'NO_ENTRIES_FOR_STRATEGY'


def shift_period(timestamp: pd.Timestamp) -> datetime64:
    """Shift the timestamp by 1 day - 1 ms"""
    ts = arrow.get(timestamp.isoformat())
    return datetime64(ts.shift(days=1).shift(microseconds=-1).isoformat())

def run_strategy(df: pd.DataFrame, strategy: Dict):
    """ For each strategy:
    - Set the trade period to one day.
    - find all entry points
    - find the profit/loss of each entry point
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Querying strategy {strategy['id']}")
    entry_points = query_strategy(df, strategy)
    res = find_profit_in_window(df, entry_points, strategy)
    # logger.info(f"Strategy got results {res}")
    return pd.DataFrame(res)


def find_profit_in_window(df: pd.DataFrame, subset: pd.DataFrame, strategy: dict) -> List[dict]:
    """ Points 2 from `main`
    2.a Set the trade period to one day.
    2.b For each of these windows find the highest point (profit)

    NB that the subset passed in is all of trades with would execute for the entire dataset (i.e. 4 years worth of trades).
    """
    # logger = logging.getLogger(__name__)
    results = []

    try:
        if not len(subset):
            res = dict(strategy=strategy, trend='na',
                open='na',
                close='na',
                high='na',
                low='na',
                result=NO_TRADES,
                profit=dict(
                    max=NO_TRADES, timestamp='na', n_steps=0),
                loss=dict(
                    max=NO_TRADES, timestamp='na', n_steps=0),
                open_timestamp='na',
                performance=0
                )
            results.append(res)


        for ix in range(len(subset)):
            row = subset.iloc[ix]
            # Get a period of data since the trade opened
            end = shift_period(row['converted_open_ts'])
            mask = (df['converted_open_ts'] > row.converted_open_ts) & (
                df['converted_open_ts'] <= end)
            window = df.loc[mask]

            if not window.empty:

                target = row.open * (1 + TARGET)
                stop_loss = row.open * (1 - STOP_LOSS)

                trade_result = get_trade_result(window, target, stop_loss)

                res = dict(strategy=strategy, trend=row.trend_direction,
                        open=row.open,
                        close=row.close,
                        high=row.high,
                        low=row.low,
                        result=trade_result,
                        open_timestamp=row['converted_open_ts'])
                # performance used for fitness function
                if res['result'] == HIT:
                    res['performance'] = (target - row.open) / row.open
                elif res['result'] == STOPPED:
                    res['performance'] = (stop_loss - row.open) / row.open
                else:
                    res['performance'] = 0
                # logger.info(f"Strategy {strategy['id']} got results {res}")
                results.append(res)
    except Exception as err:
        import ipdb;ipdb.set_trace()
    return results

def get_trade_result(window: pd.Series, target: float, stop_loss: float) -> str:
    try:
        target_hit_at = window.loc[window[window['close']
                                          >= target].index.min()]
    except KeyError:
        target_hit_at = -1
    else:
        target_hit_at = target_hit_at.open_ts

    try:
        stopped_out_at = window.loc[window[window['close']
                                           <= stop_loss].index.min()]
    except:
        stopped_out_at = -1
    else:
        stopped_out_at = stopped_out_at.open_ts

    if target_hit_at > stopped_out_at:
        result = HIT
    elif stopped_out_at > target_hit_at:
        result = STOPPED
    else:
        result = NA
    return result


def get_window_performance(window: pd.Series, row:pd.Series) -> Dict:
    """Finds the high and lowest points of a trade window and time deltas between them.
    """
    high = window['close'].max()
    low = window['close'].min()

    # edge case: multiple rows can close with the same value, hence locate the first occurence
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

    return dict(max_profit=max_profit,
    high_timestamp=high_timestamp,
    profit_steps=profit_steps,
    max_loss=max_loss,
    low_timestamp=low_timestamp,
    loss_steps=loss_steps)