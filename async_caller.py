from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import repeat
from typing import Iterable


"""Wrapper library for async calls using Futures. Note that when implementing
the function being called the iterable as the last argument, but when calling
the future caller it needs to be the first
(due to the partial function creation).

e.g.

a function which adds items to a list could be parallelised to work on multiple
lists as follows:

def func(thing1, thing2, myList):
    return myList.extend([thing1, thing2])

which would be called as:

process_future_caller([[], [], []], 1, 2)

"""


def process_future_caller(func: callable, iterable: Iterable, *constants):
    return future_caller(func, iterable, ProcessPoolExecutor, *constants)


def threaded_future_caller(func: callable, iterable: Iterable, *constants):
    return future_caller(func, iterable, ThreadPoolExecutor, *constants)


def future_caller(func: callable, iterable, Executor=ThreadPoolExecutor, *constants):
    """Wrapper to map the callable to the iteratable using the `constants` as partial args for each call."""

    with Executor() as executor:
        fun = partial(func, *constants)
        results = executor.map(fun, iterable)

    return tuple(results)
