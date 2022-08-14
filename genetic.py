from copy import deepcopy
from math import log
import os
import random
from typing import Callable, Dict, List, Tuple

from generate_strategy import LOADED_INDICATORS, make_strategy_from_indicators
import arrow
import sys
import pandas as pd
from df_adapter import dfa
import logging
from generate_strategy import main as generate_main
from generate_strategy import generate
from memory_profiler import profile
from generate_blank_signals import INDICATORS
from run_strategy import run_strategy
from async_caller import process_future_caller
from helpers import make_pandas_df, write_df_to_s3, base_arg_parser, POPULATION_SIZE

# Test run:
# population size 10,
# 100 generations
#
# runtime: 21037 seconds

# risk:reward ratio 2:1
TARGET = 0.015
STOP_LOSS = TARGET / 2

HIT = "TARGET_HIT"
STOPPED = "STOPPED_OUT"
NA = "NO_CLOSE_IN_WINDOW"
NO_TRADES = "NO_ENTRIES_FOR_STRATEGY"

MAX_GENERATIONS = 100

def load_df():
    df = dfa.read_csv("BTCUSDC_indicators.csv")
    return add_previous_window_values(df)


def add_previous_window_values(df: dfa.DataFrame) -> dfa.DataFrame:
    # shift the whole df back one place
    shifted = df.shift(1)
    for ind in INDICATORS:
        df[f"{ind}_previous"] = shifted[ind]
    return df


# @profile
def main(
    trading_data: dfa.DataFrame,
    strategies: List[Dict] = None,
    fitness_function: Callable = None,
    generation: int = 1,
    max_generations: int = MAX_GENERATIONS,
    ranked_results: List = None,
    serial_debug: bool = False
):
    """Main Genetic Algorthim. Logic:

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

    if generation <= max_generations:
        logger = logging.getLogger(__name__)
        logger.info(f"Running generation {generation}")

        # Serial invocation - for debugging
        if serial_debug:
            results = [run_strategy(trading_data, strat) for strat in strategies]
            fitness = [fitness_function(x) for x in results]
        else:
            results = process_future_caller(run_strategy, strategies, trading_data)
            fitness = process_future_caller(fitness_function, results)

        ranking, weights = apply_ranking(fitness)
        ranked_results.append(ranking)

        logger.info(f"RECURSSING!")

        population = generate_population(ranking, weights)
        main(
            trading_data,
            population,
            fitness_function,
            generation=generation + 1,
            max_generations=max_generations,
            ranked_results=ranked_results,
        )
    return ranked_results


def fitness_function_ha_and_moon(strategy: dfa.DataFrame) -> pd.DataFrame:
    """Fitness function based on the Ha & Moon study.

    g(r)i,j = log (pc(i, j + k) / pc(i, j))
    """

    n_trades, win_percent, avg_gain = fitness_metadata(strategy)
    if n_trades == 0:
        fitness = -1000
        max_fitness = -1000
    else:
        strategy["fitness"] = strategy.apply(
            lambda x: log((x.close + n_trades) / x.close), axis=1
        )
        fitness = strategy.fitness.mean()
        max_fitness = strategy.fitness.max()
    return pd.DataFrame(
        [
            {
                "id": strategy.iloc[0].strategy["id"],
                "avg_gain": avg_gain,
                "fitness": fitness,
                "max_fitness": max_fitness,
                "n_trades": n_trades,
                "win_percent": win_percent,
                "strategy": strategy.iloc[0].strategy,
            }
        ]
    )


def fitness_function(strategy: dfa.DataFrame) -> pd.DataFrame:
    """Win percentage * number of trades gives a performance coefficient.
    That is the higher the WP / NT, the bigger the coeff.
    Returns gain on account * performance coeff.
    """
    n_trades, win_percent, avg_gain = fitness_metadata(strategy)
    if n_trades == 0:
        fitness = -1000
    else:
        fitness = avg_gain * (win_percent * n_trades)
    return pd.DataFrame(
        [
            {
                "id": strategy.iloc[0].strategy["id"],
                "avg_gain": avg_gain,
                "fitness": fitness,
                "n_trades": n_trades,
                "win_percent": win_percent,
                "strategy": strategy.iloc[0].strategy,
            }
        ]
    )


def fitness_metadata(strategy):
    try:
        # Deletion NB: no longer using this method as no trades should still
        # be counted as the trade was entered.
        # n_trades = len(strategy.loc[strategy.result != NO_TRADES])
        # win_percent = (
        #     len(strategy.loc[strategy.result == HIT]) / n_trades
        # ) * 100
        # avg_gain = (strategy.performance.mean() / n_trades) * 100

        n_trades = len(strategy)
        win_percent = (
            len(strategy.loc[strategy.result == HIT]) / n_trades
        ) * 100
        avg_gain = strategy.performance.mean()

    except (AttributeError, ZeroDivisionError):
        n_trades = 0
        win_percent = 0
        avg_gain = 0
    return n_trades, win_percent, avg_gain


def apply_ranking(results: Tuple) -> Tuple[pd.DataFrame, Dict]:
    df = pd.concat(results)
    df.sort_values(by="fitness", ascending=False, inplace=True)
    n_items = len(df)
    # nb +1 to avoid divide by zero
    weights = {df.iloc[i].strategy["id"]: n_items + 1 / (i + 1) for i in range(n_items)}
    return df, weights


def generate_population(ranked: dfa.DataFrame, weights: Dict) -> List[Dict]:
    """Applies ranking, cross over and mutation to create a new population"""
    # Elitism - keep the two best solutions from the previous population

    ranked = ranked.reset_index()
    # elitism
    population = [ranked.iloc[0].strategy, ranked.iloc[1].strategy]
    logger = logging.getLogger(__name__)
    logger.info(f"generating new population of length {POPULATION_SIZE}")

    _weights = deepcopy(weights)

    while len(population) < POPULATION_SIZE:
        x, y = select_parents(ranked, _weights)
        offspring = cross_over_ppx(x.strategy, y.strategy)
        population.append(mutate(offspring))
    logger.info("Population created")
    return population


def select_parents(
    ranked: dfa.DataFrame, _weights: Dict, parents: Dict = {}
) -> Tuple[dfa.DataFrame, dfa.DataFrame]:
    """Randomly sample parents to use for offspring generation"""
    sampled = ranked.sample(2, weights=_weights.values())

    x, y = sampled.iloc[0], sampled.iloc[1]
    if (x.id, y.id) in parents:
        return select_parents(ranked, _weights, parents)
    parents[(x.id, y.id)] = 1

    return x, y


def cross_over_ppx(strat_x: Dict, strat_y: Dict) -> Dict:
    """Precedence preservative crossover"""
    x_ind = strat_x["indicators"]
    y_ind = strat_y["indicators"]

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
    """Partially matched cross over (PMX)."""
    logger = logging.getLogger(__name__)

    x_ind = strat_x["indicators"]
    y_ind = strat_y["indicators"]

    pmx_inheritence_range = min(len(x_ind), len(y_ind))
    pmx_inheritence_ix = random.randint(0, pmx_inheritence_range)

    base_indicator = random.choice(LOADED_INDICATORS)

    logger.info(f"Creating child using {base_indicator}")

    child_x = generate(base_indicator, LOADED_INDICATORS)
    child_y = generate(base_indicator, LOADED_INDICATORS)

    # Select the genes to carry through to the next generation
    x_inherited_gene = x_ind[pmx_inheritence_ix]
    y_inherited_gene = y_ind[pmx_inheritence_ix]

    child_x_inds = child_x["indicators"]
    child_y_inds = child_y["indicators"]

    # Select the position in which to place the inherited genes
    pmx_cross_over_range = min(
        max(len(child_x_inds) - 1, 0), max(len(child_y_inds) - 1, 0)
    )
    pmx_cross_over_ix = random.randint(0, pmx_cross_over_range)

    logger.info(
        f"child_x to inherit: {x_inherited_gene} in position {pmx_cross_over_ix}"
    )
    logger.info(
        f"child_y to inherit: {y_inherited_gene} in position {pmx_cross_over_ix}"
    )

    # Mutate the offspring
    child_x_inds[pmx_cross_over_ix] = x_inherited_gene
    child_y_inds[pmx_cross_over_ix] = y_inherited_gene

    return child_x, child_y


def mutate(strategy: List):
    _strat = deepcopy(strategy)
    abs_strats = [x for x in _strat["indicators"] if x["absolute"]]

    if abs_strats:
        mutate_ix = random.choice(range(len(abs_strats)))
        mutate_ind = abs_strats[mutate_ix]
        mutate_percent = random.choice(range(-20, 20)) / 100
        mutate_ind["abs_value"] = mutate_ind["abs_value"] * (1 + mutate_percent)

    return _strat


def is_profitable(
    max_profit: float,
    max_loss: float,
    high_timestamp: pd.Timestamp,
    low_timestamp: pd.Timestamp,
) -> bool:
    """Is the profitable high point before the max loss occurred?"""
    if max_loss < 0:
        return (max_profit > 0) and (high_timestamp < low_timestamp)
    else:
        return max_profit > 0


def is_loss_making(
    max_profit: float,
    max_loss: float,
    high_timestamp: pd.Timestamp,
    low_timestamp: pd.Timestamp,
) -> bool:
    """Does the max loss occur before the max profit was made?"""
    if max_profit <= 0:
        return (max_loss < 0) and (low_timestamp < high_timestamp)
    else:
        return max_loss < 0


def load_trading_data():
    df = load_df()
    df["converted_open_ts"] = dfa.to_datetime(df["open_ts"], unit="ms")
    df.index = df["open_ts"]
    return df


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s%(levelname)s:%(message)s",
        stream=sys.stderr,
        level=logging.INFO,
    )

    parser = base_arg_parser("Main genetic algorithm.")
    parser.add_argument("--write_s3", type=bool, help="exports data to s3.")
    parser.add_argument("--write_local", type=bool, help="exports data to local FS.")
    parser.add_argument(
        "--s3_bucket", type=str, help="bucket name to which data is written."
    )
    parser.add_argument("--generations", type=int, help="N generations to run.")
    parser.add_argument("--serial_debug", type=bool, help='run without async for debugging')

    parser.set_defaults(
        write_s3=False,
        write_local=True,
        generations=MAX_GENERATIONS,
        s3_bucket=os.environ['BUCKET'],
        serial_debug=False
    )
    args = parser.parse_args()

    strategies = generate_main(
        max_indicators=args.max_indicators,
        max_same_class=args.max_same_class,
        population_size=args.population_size,
    )
    start = arrow.utcnow()
    results = main(
        load_trading_data(),
        strategies,
        fitness_function,
        max_generations=args.generations,
        serial_debug=args.serial_debug
    )
    stop = arrow.utcnow()

    df = dfa.concat(results)

    fname = f"{start.isoformat()}_results.csv"

    df = make_pandas_df(df)

    if args.write_local:
        df.to_csv(fname)
    if args.write_s3:
        write_df_to_s3(df, args.s3_bucket, fname)
