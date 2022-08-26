# See Algorithn 1: Strategy Generation
import random
import logging
import sys
import argparse
import json
from string import ascii_lowercase
from typing import Sequence
import uuid
from generate_blank_signals import INDICATORS
from helpers import base_arg_parser

from env import (
    MIN_INDICATORS,
    MAX_SAME_CLASS_INDICATORS,
    MAX_STRATEGY_INDICATORS,
    CONJUNCTIONS,
    POPULATION_SIZE,
)

CONJUNCTIONS = CONJUNCTIONS[:2]  # exclude negations.

with open("signals/absolute_signals.json", "r") as fi:
    LOADED_INDICATORS = json.load(fi)
with open("signals/relative_signals.json", "r") as fi:
    relative = json.load(fi)
    LOADED_INDICATORS.extend(relative)


def choose_indicator(indicators, max_same_class, same_class_indicators):
    indicator = random.choice(indicators)
    if same_class_indicators.get(indicator["type"], -1) + 1 > max_same_class:
        # Need to subset indicators to avoid hitting max recursion depth.
        _indicators = [x for x in indicators if x["type"] != indicator["type"]]
        return choose_indicator(_indicators, max_same_class, same_class_indicators)
    return indicator


def pop_indicator(to_pop, indicators):
    # Python does have an inbuilt .pop but that is o(N) for arbitary items.
    # this leads to less boiler plate when the to_pop doesn't exist.
    return [x for x in indicators if x != to_pop]


def choose_num_indicators(max_indicators):
    # helper function to aid testability of `generate`
    # nb +1 as range is exclusive
    return random.choice(range(MIN_INDICATORS, max_indicators + 1))


def make_strategy_from_indicators(indicators: Sequence, conjunctions=CONJUNCTIONS):
    """Given an iterable of indicators, turn it into a well form strategy."""
    return dict(
        id=str(uuid.uuid4()),
        indicators=indicators,
        conjunctions=[random.choice(conjunctions) for _ in range(len(indicators) - 1)],
    )


def generate(
    base_indicator,
    indicators,
    conjunctions=CONJUNCTIONS,
    max_indicators=MAX_STRATEGY_INDICATORS,
    max_same_class=MAX_SAME_CLASS_INDICATORS,
):
    """Generate a strategy with the base_indicator as the first indicator."""
    # nb range is 0 indexed and exclusive
    strategy = dict(id=str(uuid.uuid4()), indicators=[base_indicator], conjunctions=[])
    same_class_indicators = {base_indicator["type"]: 1}
    indicators = pop_indicator(base_indicator, indicators)

    num_indicators = choose_num_indicators(max_indicators)

    for i in range(num_indicators - 1):
        if i <= num_indicators - 1:
            strategy["conjunctions"].append(random.choice(conjunctions))
        # Enforce the first indicator to be an 'and'
        strategy["conjunctions"][0] = "and"
        ind = choose_indicator(indicators, max_same_class, same_class_indicators)
        count = same_class_indicators.get(ind["type"], 0)
        same_class_indicators[ind["type"]] = count + 1
        strategy["indicators"].append(ind)
        indicators = pop_indicator(ind, indicators)
    # logger = logging.getLogger()
    # logger.info(f"Generated strategy {strategy}")
    return strategy


def main(
    indicators=LOADED_INDICATORS,
    max_indicators=MAX_STRATEGY_INDICATORS,
    max_same_class=MAX_SAME_CLASS_INDICATORS,
    population_size=POPULATION_SIZE,
):
    strategies = []
    for _ in range(population_size):
        base_indicator = random.choice(indicators)
        strategies.append(
            generate(
                base_indicator,
                indicators,
                CONJUNCTIONS,
                max_indicators,
                max_same_class,
            )
        )
    return strategies


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s%(levelname)s:%(message)s",
        stream=sys.stderr,
        level=logging.INFO,
    )
    parser = base_arg_parser("Generate an initial population of random strategies")
    parser.add_argument("--test", type=bool, help="whether to run in testmode")

    args = parser.parse_args()
    if args.test:
        indicators = ascii_lowercase
    else:
        indicators = LOADED_INDICATORS
    main(indicators, args.max_indicators,
         args.max_same_class, args.population_size)
