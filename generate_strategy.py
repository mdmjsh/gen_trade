# See Algorithn 1: Strategy Generation
import random
import logging
import sys
import argparse
from string import ascii_lowercase

MIN_INDICATORS = 2
MAX_SAME_CLASS_INDICATORS = 2
MAX_STRATEGY_INDICATORS = 4
POPULATION_SIZE = 10
CONJUNCTIONS = ['AND', 'OR', 'AND NOT', 'OR NOT']


def choose_indicator(indicators, max_same_class, same_class_indicators):
    indicator = random.choice(indicators)
    if same_class_indicators.get(str(indicator.__class__), -1) + 1 > max_same_class:
        # Need to subset indicators to avoid hitting max recursion depth.
        _indicators = [
            x for x in indicators if x.__class__ != indicator.__class__]
        return choose_indicator(_indicators, max_same_class, same_class_indicators)
    return indicator


def pop_indicator(to_pop, indicators):
    # Python does have an inbuilt .pop but that is o(N) for arbitary items any way
    # this leads to less boiler plate when the to_pop doesn't exist.
    return [x for x in indicators if x != to_pop]


def choose_num_indicators(max_indicators):
    # helper function to aid testability of `generate`
    return random.choice(range(MIN_INDICATORS, max_indicators))


def generate(base_indicator, indicators, conjunctions, max_indicators=MAX_STRATEGY_INDICATORS, max_same_class=MAX_SAME_CLASS_INDICATORS):
    # nb range is 0 indexed and exclusive
    strategy = [base_indicator]
    same_class_indicators = {str(base_indicator.__class__): 1}
    indicators = pop_indicator(base_indicator, indicators)

    num_indicators = choose_num_indicators(max_indicators)
    for i in range(num_indicators - 1):
        if i <= num_indicators - 1:
            strategy.append(random.choice(conjunctions))
        ind = choose_indicator(
            indicators, max_same_class, same_class_indicators)
        count = same_class_indicators.get(str(ind.__class__), 0)
        same_class_indicators[str(ind.__class__)] = count + 1
        strategy.append(ind)
        indicators = pop_indicator(ind, indicators)

    logger = logging.getLogger()
    logger.info(f"Generated strategy {strategy}")
    return strategy


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s%(levelname)s:%(message)s',
                        stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser(
        "Generate an initial population of random strategies")
    parser.add_argument('--population_size', type=int,
                        help="number of strategies to generate", required=False)
    parser.add_argument('--max_indicators', type=int,
                        help="max number of indicators in a strategy", required=False)
    parser.add_argument('--max_same_class', type=int,
                        help="max number of same class indicators in a strategy", required=False)

    parser.set_defaults(population_size=POPULATION_SIZE,
                        max_indicators=MAX_STRATEGY_INDICATORS, max_same_class=MAX_SAME_CLASS_INDICATORS)
    args = parser.parse_args()
    indicators = ascii_lowercase

    strategies = []
    for i in range(args.population_size):
        base_indicator = random.choice(indicators)
        strategies.append(generate(base_indicator, indicators,
                          CONJUNCTIONS, args.max_indicators, args.max_same_class))
