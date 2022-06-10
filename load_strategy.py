from typing import Callable, Dict, Sequence

def get_callback(callback_value: str):
    pass

def make_callback(callback_value: str):
    callback, args, kwargs = get_callback(callback_value)
    return callback(*args, **kwargs)


def load_from_object(strategy: Dict):
    parsed = []
    conjunctions = strategy['conjunctions']

    if len(conjunctions) != len(strategy['indicators']) -1:
        raise RuntimeError(f"Strategy does not have correct number of conjunctions! {strategy}")

    for indicator in strategy['indicators']:
        if indicator["absolute"]:
            _parsed = f"{indicator['name']} {indicator['op']} {indicator['abs_value']}"
        else:
            _parsed = f"{indicator['name']} {indicator['op']} {make_callback(indicator['rel_value'])}"
        parsed.append(_parsed)

    if conjunctions:
        result = ""
        for ix, strat in enumerate(parsed):
            result += strat
            try:
                result += f" {conjunctions[ix]} "
            except IndexError:
                continue
        return result
    return parsed[0]