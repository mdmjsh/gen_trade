from asyncio.log import logger
import pandas as pd
from typing import Dict, List
from async_caller import process_future_caller, threaded_future_caller
from logger import get_logger
from helpers import make_pandas_df
from load_strategy import load_from_object_parenthesised, query_strategy
from numpy import datetime64
import arrow

# example multiple disjunct strategy
# volatility_dcm < volatility_dcm_previous and (trend_psar_up > trend_sma_slow or (volatility_kchi < volatility_kchi_previous or trend_ema_slow < trend_sma_slow))

# risk:reward ratio 2:1
TARGET = 0.015
STOP_LOSS = TARGET / 2

HIT = "TARGET_HIT"
STOPPED = "STOPPED_OUT"
NA = "NO_CLOSE_IN_WINDOW"
NO_TRADES = "NO_ENTRIES_FOR_STRATEGY"
CUTOFF_PERCENT = 1 / 3


def shift_period(timestamp: pd.Timestamp) -> datetime64:
    """Shift the timestamp by 1 day - 1 ms"""
    ts = arrow.get(timestamp.isoformat())
    return datetime64(ts.shift(days=1).shift(microseconds=-1).isoformat())


def run_strategy(df: pd.DataFrame, strategy: Dict):
    """For each strategy:
    - Set the trade period to one day.
    - find all entry points
    - find the profit/loss of each entry point
    """
    logger = get_logger(__name__)
    logger.info(f"Querying strategy {strategy['id']}")

    # NB not a pure function as we update the strategy here, but hey-ho
    strategy["parsed"] = load_from_object_parenthesised(strategy)
    entry_points = query_strategy(df, query=strategy["parsed"])

    df = make_pandas_df(df)
    entry_points = make_pandas_df(entry_points)
    logger.info(f"Found {len(entry_points)} potential trades...")

    res = find_profit_in_window(df, entry_points, strategy)

    return pd.DataFrame(res)


def find_profit_in_window(
    df: pd.DataFrame, subset: pd.DataFrame, strategy: dict
) -> List[dict]:
    """Points 2 from `main`
    2.a Set the trade period to one day.
    2.b For each of these windows find the highest point (profit)

    NB that the subset passed in is all of trades with would execute for the
    entire dataset (i.e. 4 years worth of trades).
    """
    logger = get_logger(__name__)
    results = []

    # Don't evaluate trades which look like high false positive %
    trade_percent = len(subset) / len(df)
    if trade_percent >= CUTOFF_PERCENT:
        logger.info(
            f"Strategy {strategy['id']} generated too many potential "
            f"trades ({trade_percent}%) - assuming it's rubbish."
        )
        return [no_trade_results(strategy, trade_percent)]

    if not len(subset):
        results.append(no_trade_results(strategy, trade_percent))

    results = process_future_caller(
        assess_strategy_window,
        range(len(subset)), df, subset, strategy, trade_percent,
    )
    return results


def assess_strategy_window(df, subset, strategy, trade_percent, ix):
    # Get a period of data since the trade opened
    row = subset.iloc[ix]

    end = shift_period(row["converted_open_ts"])
    mask = (df["converted_open_ts"] > row.converted_open_ts) & (
        df["converted_open_ts"] <= end
    )
    window = df.loc[mask]

    if not window.empty:
        target = row.open * (1 + TARGET)
        stop_loss = row.open * (1 - STOP_LOSS)

        trade_result = get_trade_result(window, target, stop_loss)

        res = dict(
            target=target,
            stop_loss=stop_loss,
            strategy=strategy,
            trend=row.trend_direction,
            open=row.open,
            close=row.close,
            high=row.high,
            low=row.low,
            result=trade_result,
            open_timestamp=row["converted_open_ts"],
            trade_percent=trade_percent,
        )
        # performance used for fitness function
        if res["result"] == HIT:
            res["performance"] = (target - row.open) / row.open
        elif res["result"] == STOPPED:
            res["performance"] = (stop_loss - row.open) / row.open
        else:
            res["performance"] = 0
            # logger.info(f"Strategy {strategy['id']} got results {res}")
        res


def no_trade_results(strategy, trade_percent):
    res = dict(
        strategy=strategy,
        trend="na",
        open="na",
        close="na",
        high="na",
        low="na",
        result=NO_TRADES,
        profit=dict(max=NO_TRADES, timestamp="na", n_steps=0),
        loss=dict(max=NO_TRADES, timestamp="na", n_steps=0),
        open_timestamp="na",
        performance=0,
        trade_percent=trade_percent,
    )
    return res


def get_trade_result(window: pd.Series, target: float, stop_loss: float) -> str:
    try:
        target_hit_at = window.loc[window[window["close"] >= target].index.min()]
    except KeyError:
        target_hit_at = -1
    else:
        target_hit_at = target_hit_at.open_ts

    try:
        stopped_out_at = window.loc[window[window["close"] <= stop_loss].index.min()]
    except Exception:
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


def get_window_performance(window: pd.Series, row: pd.Series) -> Dict:
    """Finds the high and lowest points of a trade window and time deltas between them."""
    high = window["close"].max()
    low = window["close"].min()

    # edge case: multiple rows can close with the same value,
    # hence locate the first occurence
    high_point = window.loc[
        window["open_ts"] == window[window["close"] == high].index.min()
    ]
    low_point = window.loc[
        window["open_ts"] == window[window["close"] == low].index.min()
    ]
    # delta between window open and high point
    max_profit = high - row["open"]
    max_loss = low - row["open"]

    high_timestamp = pd.Timestamp(high_point["converted_open_ts"].values[0])
    high_delta = high_timestamp - row["converted_open_ts"]
    profit_steps = high_delta.components.hours * 15 + high_delta.components.minutes

    low_timestamp = pd.Timestamp(low_point["converted_open_ts"].values[0])
    low_delta = low_timestamp - row["converted_open_ts"]
    loss_steps = low_delta.components.hours * 15 + low_delta.components.minutes

    return dict(
        max_profit=max_profit,
        high_timestamp=high_timestamp,
        profit_steps=profit_steps,
        max_loss=max_loss,
        low_timestamp=low_timestamp,
        loss_steps=loss_steps,
    )
