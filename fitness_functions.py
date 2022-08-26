from math import log
from typing import Callable, Iterable, Tuple
import pandas as pd
from df_adapter import dfa
from env import HIT
from profit_calculator import simple_profit_calculator
from async_caller import process_future_caller


def fitness_metadata(serial_debug: bool | None, strategy: pd.DataFrame) -> Tuple:
    try:
        n_trades = len(strategy)
        win_percent = (len(strategy.loc[strategy.result == HIT]) / n_trades) * 100
        avg_gain = strategy.performance.mean()

    except (AttributeError, ZeroDivisionError):
        n_trades = 0
        win_percent = 0
        avg_gain = 0
    return n_trades, win_percent, avg_gain


def fitness_simple_profit(serial_debug: bool, strategy: Iterable) -> pd.DataFrame:
    if serial_debug:
        import ipdb;ipdb.set_trace()
        df = pd.concat([fitness_function_ha_and_moon(x, serial_debug) for x in strategy])
    else:
        print(f"STRATEGY: {strategy}")
        df = pd.concat(process_future_caller(fitness_function_ha_and_moon, strategy, serial_debug))
    fitness = simple_profit_calculator(df=df)
    df["fitness"] = fitness.values()
    fitness = [df]
    return fitness


def fitness_function_ha_and_moon(
    serial_debug: bool | None, strategy: dfa.DataFrame
) -> pd.DataFrame:
    """Fitness function based on the Ha & Moon study.

    g(r)i,j = log((pc(i, j) + k) / pc(i, j))
    """
    # handle recursive calls already made which already have fitness data
    print(f"STRATEGY: {strategy}")
    if "fitness" in strategy.keys():
        return strategy

    n_trades, win_percent, avg_gain = fitness_metadata(strategy)
    if not n_trades:
        fitness = -1000
    else:
        strategy["fitness"] = strategy.apply(
            lambda x: log_increase(x, n_trades), axis=1
        )
        fitness = strategy.fitness.mean()
    return transform_fitness_results(strategy, n_trades, win_percent, avg_gain, fitness)


def fitness_function_original(
    serial_debug: bool | None, strategy: dfa.DataFrame
) -> pd.DataFrame:
    """Win percentage * number of trades gives a performance coefficient.
    That is the higher the WP / NT, the bigger the coeff.
    Returns gain on account * performance coeff.
    """
    # handle recursive calls already made which already have fitness data
    if "fitness" in strategy.keys():
        return strategy

    n_trades, win_percent, avg_gain = fitness_metadata(strategy)
    if n_trades <= 1:
        fitness = -1000
    else:
        fitness = avg_gain * (win_percent * n_trades)
    return transform_fitness_results(strategy, n_trades, win_percent, avg_gain, fitness)


def transform_fitness_results(strategy, n_trades, win_percent, avg_gain, fitness):
    return pd.DataFrame(
        [
            {
                "id": strategy.iloc[0].strategy["id"],
                "avg_gain": avg_gain,
                "fitness": fitness,
                "n_trades": n_trades,
                "win_percent": win_percent,
                "target": strategy.target.to_dict(),
                "stop_loss": strategy.stop_loss.to_dict(),
                "result": strategy.result.to_dict(),
                "strategy": strategy.iloc[0].strategy,
                "performance": strategy.performance.to_dict(),
                "open_ts": strategy.open_timestamp.to_dict(),
                "trend": strategy.trend.to_dict(),
            }
        ]
    )


def log_increase(x, n_trades):
    price = x.close or 0
    try:
        return log((price + n_trades) / price)
    except ZeroDivisionError:
        return -1000
