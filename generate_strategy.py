# See Algorithn 1: Strategy Generation
import random

MAX_INDICATORS=4

def choose_indicator(indicators, max_same_class, same_class_indicators):
    indicator = random.sample(indicators, k=1)[0]
    if same_class_indicators.get(str(indicator.__class__), -1) +1 > max_same_class:
        return choose_indicator(indicators, max_same_class, same_class_indicators)
    return indicator

def pop_indicator(to_pop, indicators):
    # Python does have an inbuilt .pop but that is o(N) for arbitary items any way
    # this leads to less boiler plate when the to_pop doesn't exist.
    return [x for x in indicators if x != to_pop]

def generate(base_indicator, indicators, conjunctions, max_indicators):
    # nb range is 0 indexed and exclusive
    strategy = [base_indicator]
    same_class_indicators = {str(base_indicator.__class__): 1}
    # import ipdb;ipdb.set_trace()
    indicators = pop_indicator(base_indicator, indicators)

    for i in range(max_indicators -1):
        if i <= max_indicators -1:
            strategy.append(random.sample(conjunctions, k=1)[0])
        ind = choose_indicator(indicators, MAX_INDICATORS, same_class_indicators)
        count = same_class_indicators.get(str(ind.__class__), 0)
        same_class_indicators[str(ind.__class__)] = count +1
        strategy.append(ind)
        indicators = pop_indicator(ind, indicators)

    return strategy

