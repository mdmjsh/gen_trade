from ast import Call
from copy import deepcopy
from decimal import DivisionByZero
import json

import os
import random
from typing import Callable, Dict, Iterable, List, Tuple

from generate_strategy import LOADED_INDICATORS, make_strategy_from_indicators
import arrow
import sys
import pandas as pd
from df_adapter import dfa
from generate_strategy import main as generate_main
from generate_strategy import generate
from generate_blank_signals import INDICATORS
from run_strategy import run_strategy

from async_caller import process_future_caller
from helpers import make_pandas_df, write_df_to_s3, base_arg_parser
from logger import get_logger
from env import POPULATION_SIZE, MAX_GENERATIONS
from fitness_functions import (
    fitness_function_ha_and_moon,
    fitness_function_original,
    fitness_simple_profit,
)

"""
# Test run:
# population size 10,
# 100 generations
#
# runtime: 21037 seconds

# risk:reward ratio 2:1

# Second test run
# 2022-08-21 15:19:05,962
# INFO:Finished in 13753 seconds
# 100 generations

# third run
# 2022-08-21T18:33:01.133223+00:00_results.csv
# Finished in 63173 seconds
# 500 generations


run 4: after removing negation conjunctions
original FF
2022-08-23T13:36:00.298698+00:00_results.csv
INFO:Finished in 34164 seconds
"""


def load_df():
    df = dfa.read_csv("BTCUSDC_indicators.csv")
    return add_previous_window_values(df)


def add_previous_window_values(df: dfa.DataFrame) -> dfa.DataFrame:
    # shift the whole df back one place
    shifted = df.shift(1)
    for ind in INDICATORS:
        df[f"{ind}_previous"] = shifted[ind]
    return df


def main(
    trading_data: dfa.DataFrame,
    strategies: List[Dict] = None,
    fitness_function: Callable = None,
    generation: int = 1,
    max_generations: int = MAX_GENERATIONS,
    ranked_results: List = None,
    serial_debug: bool = False,
):
    """Main Genetic Algorithm. Logic:

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
    logger = get_logger(__name__)

    if generation <= max_generations:
        logger.info(f"Running generation {generation}")

        # Serial invocation - for debugging
        if serial_debug:
            results = [
                run_strategy(trading_data, ranked_results, strat)
                for strat in strategies
            ]
            fitness = [fitness_function(serial_debug, x) for x in results]

        else:

            results = process_future_caller(
                run_strategy, strategies, trading_data, ranked_results
            )
            fitness = process_future_caller(fitness_function, results, serial_debug)

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
            serial_debug=serial_debug,
        )
    return ranked_results


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
    logger = get_logger(__name__)
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
    logger = get_logger(__name__)

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


def load_strategies(
    path: str = None,
    max_indicators: int = None,
    max_same_class: int = None,
    population_size: int = None,
) -> List[Dict]:
    if path:
        with open(path, "r"):
            return json.load(path)

    return generate_main(
        max_indicators=max_indicators,
        max_same_class=max_same_class,
        population_size=population_size,
    )


FITNESS_MAP = dict(
    o=fitness_function_original,
    h=fitness_function_ha_and_moon,
    p=fitness_simple_profit,
)


if __name__ == "__main__":
    parser = base_arg_parser("Main genetic algorithm.")
    parser.add_argument("--write_s3", type=bool, help="exports data to s3.")
    parser.add_argument("--write_local", type=bool, help="exports data to local FS.")
    parser.add_argument(
        "--s3_bucket", type=str, help="bucket name to which data is written."
    )
    parser.add_argument("--generations", type=int, help="N generations to run.")
    parser.add_argument(
        "--serial_debug", type=bool, help="run without async for debugging"
    )
    parser.add_argument(
        "--strategies_path",
        type=str,
        help="load strategies from this path rather than generating on the fly",
    )
    parser.add_argument(
        "--fitness_function",
        type=str,
        help="fitness function use (h=ha_and_moon, o=original, p=profit)",
    )

    parser.set_defaults(
        write_s3=False,
        write_local=True,
        generations=os.getenv("GENERATIONS", MAX_GENERATIONS),
        s3_bucket=os.getenv("BUCKET"),
        serial_debug=False,
        strategies_path=None,
        fitness_function="o",
    )
    args = parser.parse_args()

    strategies = load_strategies(
        args.strategies_path,
        args.max_indicators,
        args.max_same_class,
        args.population_size,
    )
    start = arrow.utcnow()
    results = main(
        load_trading_data(),
        strategies,
        FITNESS_MAP[args.fitness_function],
        max_generations=args.generations,
        serial_debug=args.serial_debug,
    )
    stop = arrow.utcnow()

    df = dfa.concat(results)

    fname = (
        f"{start.isoformat()}_results_{args.fitness_function}_{args.generations}.csv"
    )

    df = make_pandas_df(df)

    if args.write_local:
        df.to_csv(fname)
    if args.write_s3:
        write_df_to_s3(df, args.s3_bucket, fname)

    logger = get_logger(__name__)
    logger.info(f"Finished in {(stop - start).seconds} seconds")
