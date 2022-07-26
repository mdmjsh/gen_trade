from copy import deepcopy
from functools import partial
from itertools import repeat
import random
from typing import Callable, Dict, List, Tuple

from generate_strategy import LOADED_INDICATORS, make_strategy_from_indicators
import arrow
import sys
import pandas as pd
import logging
from generate_strategy import main as generate_main
from generate_strategy import generate, POPULATION_SIZE
from load_strategy import query_strategy
from memory_profiler import profile
from generate_blank_signals import INDICATORS
from run_strategy import run_strategy

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
    from async_caller import process_future_caller, threaded_future_caller
    from concurrent.futures import ProcessPoolExecutor

    if generation <= max_generations:
        logger = logging.getLogger(__name__)
        logger.info(f"Running generation {generation}")

        # with ProcessPoolExecutor() as executor:
        #     fun = partial(run_strategy, trading_data)
        #     results = tuple(executor.map(fun, strategies))
        #     import ipdb;ipdb.set_trace()
        results = process_future_caller(
            run_strategy, strategies, trading_data)

        fitness = process_future_caller(fitness_function, results)

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
