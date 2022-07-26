from copy import deepcopy
from functools import partial
from itertools import repeat
import random
from typing import Callable, Dict, List, Tuple

from generate_strategy import LOADED_INDICATORS, make_strategy_from_indicators
from numpy import datetime64
import arrow
import sys
import pandas as pd
import logging
from generate_strategy import main as generate_main
from generate_strategy import generate, POPULATION_SIZE
from load_strategy import query_strategy
from memory_profiler import profile
from generate_blank_signals import INDICATORS

# risk:reward ratio 2:1
TARGET = 0.015
STOP_LOSS = TARGET / 2

HIT = 'TARGET_HIT'
STOPPED = 'STOPPED_OUT'
NA = 'NO_CLOSE_IN_WINDOW'
NO_TRADES = 'NO_ENTRIES_FOR_STRATEGY'

MAX_GENERATIONS = 100


def load_df():
    df = pd.read_csv('BTCUSDC_indicators.csv')
    return add_previous_window_values(df)


def add_previous_window_values(df):
    # shift the whole df back one place
    shifted = df.shift(1)
    for ind in INDICATORS:
        df[f'{ind}_previous'] = shifted[ind]
    return df

def run_strategy(df: pd.DataFrame, strategy: Dict):
    """ For each strategy:
    - Set the trade period to one day.
    - find all entry points
    - find the profit/loss of each entry point
    """
    logger = logging.getLogger(__name__)
    # logger.info(f"Querying strategy {strategy}")
    subset = query_strategy(df, strategy)
    # res = find_profit_in_window(df, entry_points, strategy)

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

                target = row.open * (1 + TARGET)
                stop_loss = row.open * (1 - STOP_LOSS)

                trade_result = get_trade_result(window, target, stop_loss)

                res = dict(strategy=strategy, trend=row.trend_direction,
                        open=row.open,
                        close=row.close,
                        high=row.high,
                        low=row.low,
                        result=trade_result,
                        profit=dict(
                            max=max_profit, timestamp=high_timestamp, n_steps=profit_steps),
                        loss=dict(
                            max=max_loss, timestamp=low_timestamp, n_steps=loss_steps),
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

    # logger.info(f"Strategy got results {res}")
    return pd.DataFrame(results)

# @profile
def main(trading_data: pd.DataFrame, strategies: List[Dict] = None, fitness_function: Callable = None, generation: int = 1, max_generations: int = MAX_GENERATIONS, ranked_results: List = None):
    """ Main Genetic Algorthim. Logic:

    1. Subset the dataframe by the strategy (for each strategy)
    2. For each strategy:
        2.a Set the trade period to one day.
        2.b For each of these windows find the highest point (profit)
    3. calculate the average of these profits points
    4. calculate the fitness function
    5. selection pair from initial population
    6.a apply crossover operation on pair with cross over probability
    6.b apply mutation on offspring using mutation probability
    7. replace old population and repeat for MAX_POP steps
    8. rank the solutions and return the best
    """
    ranked_results = ranked_results or []
    logger = logging.getLogger(__name__)
    # from async_caller import process_future_caller, threaded_future_caller
    from concurrent.futures import ProcessPoolExecutor

    if generation <= max_generations:
        logger = logging.getLogger(__name__)
        logger.info(f"Running generation {generation}")
        import ipdb;ipdb.set_trace()

        with ProcessPoolExecutor() as executor:
            fun = partial(run_strategy, trading_data)
            results = tuple(executor.map(fun, strategies))

        # results = process_future_caller(
        #     run_strategy, strategies, trading_data)

        # fitness = process_future_caller(fitness_function, results)

        ranking, weights = apply_ranking(fitness)
        ranked_results.append(ranking)

        logger.info(f"RECURSSING!")
        population = generate_population(ranking, weights)
        main(trading_data, population, fitness_function,
             generation=generation+1, max_generations=max_generations, ranked_results=ranked_results)
    return ranked_results


def fitness_function(strategy: pd.DataFrame) -> pd.DataFrame:
    """ Win percentage * number of trades gives a performance coeffient. That is the higher
    the WP / NT, the bigger the coeff. Returns gain on account * performance coeff.
    """
    try:
        n_trades = len(strategy.loc[strategy.result != NO_TRADES])
        win_percent = len(
            strategy.loc[strategy.result == 'TARGET_HIT']) / n_trades
        avg_percent_gain = (strategy.performance.sum() / n_trades) * 100
        fitness = avg_percent_gain * (win_percent * n_trades)
    except (AttributeError, ZeroDivisionError):
        n_trades=0
        win_percent=0
        avg_percent_gain=0
        fitness=0
    return pd.DataFrame([{'id': strategy.iloc[0].strategy['id'], 'avg_percent_gain': avg_percent_gain, 'fitness': fitness,
                        'n_trades':n_trades, 'win_percent': win_percent, 'strategy': strategy.iloc[0].strategy}])


def apply_ranking(results: Tuple) -> Tuple[pd.DataFrame, Dict]:
    df = pd.concat(results)
    df.sort_values(by='fitness', ascending=False, inplace=True)
    n_items = len(df)
    # nb +1 to avoid divide by zero
    weights = {df.iloc[i].strategy['id']: n_items + 1 / (i+1) for i in range(n_items)}
    return df, weights


def generate_population(ranked: pd.DataFrame, weights: Dict) -> List[Dict]:
    """Applies ranking, cross over and mutation to create a new population"""
    # Ellitism - keep the two best solutions from the previous population
    ranked = ranked.reset_index()
    population = [ranked.iloc[0].strategy, ranked.iloc[1].strategy]
    logger = logging.getLogger(__name__)

    logger.info(f"generating new population of length {POPULATION_SIZE}")

    _weights = deepcopy(weights)
    while len(population) < POPULATION_SIZE:
        try:
            sampled = ranked.sample(2, weights=_weights.values())
            ranked = ranked.drop(sampled.index)
            for i in range(len(sampled)):
                strat_id = sampled.iloc[i].strategy['id']
                if strat_id in _weights:
                    del _weights[strat_id]
            # x,y will be used to created offspring, but also included in next (i.e. ellitism).
            x, y = sampled.iloc[0], sampled.iloc[1]
            offspring = cross_over_ppx(x.strategy, y.strategy)
            population.append(offspring)
        except ValueError:
            import ipdb;ipdb.set_trace()
    logger.info("Population created")
    return population



def cross_over_ppx(strat_x: Dict, strat_y: Dict) -> Dict:
    """Precedence preservative crossover"""
    x_ind = strat_x['indicators']
    y_ind = strat_y['indicators']

    child_len = max(len(x_ind), len(y_ind))
    mapping = [random.choice([0, 1]) for _ in range(child_len)]
    mapped_inds = {0: (x_ind, y_ind), 1: (y_ind, x_ind)}

    offspring = []

    for ix, chromo in enumerate(mapping):
        primary, secondary = mapped_inds[chromo]
        try:
            p_ind = primary[ix]
        except IndexError:
            p_ind = secondary[ix]
        finally:
            if p_ind not in offspring:
                offspring.append(p_ind)
    ble = make_strategy_from_indicators(offspring)
    return ble

def cross_over_pmx(strat_x: Dict, strat_y: Dict) -> Tuple[Dict, Dict]:
    """Partially matched cross over (PMX). """
    logger = logging.getLogger(__name__)

    x_ind = strat_x['indicators']
    y_ind = strat_y['indicators']

    pmx_inheritence_range = min(len(x_ind), len(y_ind))
    pmx_inheritence_ix = random.randint(0, pmx_inheritence_range)

    base_indicator = random.choice(LOADED_INDICATORS)

    logger.info(f"Creating child using {base_indicator}")

    child_x = generate(base_indicator, LOADED_INDICATORS)
    child_y = generate(base_indicator, LOADED_INDICATORS)

    # Select the genes to carry through to the next generation
    x_inherited_gene = x_ind[pmx_inheritence_ix]
    y_inherited_gene = y_ind[pmx_inheritence_ix]

    child_x_inds = child_x['indicators']
    child_y_inds = child_y['indicators']

    # Select the position in which to place the inherited genes
    pmx_cross_over_range = min(max(len(child_x_inds)-1, 0), max(len(child_y_inds)-1, 0))
    pmx_cross_over_ix = random.randint(0, pmx_cross_over_range)

    logger.info(
        f"child_x to inherit: {x_inherited_gene} in position {pmx_cross_over_ix}")
    logger.info(
        f"child_y to inherit: {y_inherited_gene} in position {pmx_cross_over_ix}")

    # Mutate the offspring
    child_x_inds[pmx_cross_over_ix] = x_inherited_gene
    child_y_inds[pmx_cross_over_ix] = y_inherited_gene

    return child_x, child_y


def find_profit_in_window(df: pd.DataFrame, subset: pd.DataFrame, strategy: dict) -> List[dict]:
    """ Points 2 from `main`
    2.a Set the trade period to one day.
    2.b For each of these windows find the highest point (profit)

    NB that the subset passed in is all of trades with would execute for the entire dataset (i.e. 4 years worth of trades).
    """
    logger = logging.getLogger(__name__)
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

                target = row.open * (1 + TARGET)
                stop_loss = row.open * (1 - STOP_LOSS)

                trade_result = get_trade_result(window, target, stop_loss)

                res = dict(strategy=strategy, trend=row.trend_direction,
                        open=row.open,
                        close=row.close,
                        high=row.high,
                        low=row.low,
                        result=trade_result,
                        profit=dict(
                            max=max_profit, timestamp=high_timestamp, n_steps=profit_steps),
                        loss=dict(
                            max=max_loss, timestamp=low_timestamp, n_steps=loss_steps),
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


def load_trading_data():
    df = load_df()
    df['converted_open_ts'] = pd.to_datetime(df['open_ts'], unit='ms')
    df.index = df['open_ts']
    return df


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s%(levelname)s:%(message)s',
                        stream=sys.stderr, level=logging.INFO)

    strategies = generate_main()
    start = arrow.utcnow()
    results = main(load_trading_data(), strategies,
                   fitness_function, max_generations=10)
    stop = arrow.utcnow()
    import ipdb
    ipdb.set_trace()
