from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures._base import Future
from functools import partial
from typing import Iterable

def process_future_caller(func: callable, iterable: Iterable, *constants):
    return future_caller(func, iterable, ProcessPoolExecutor, *constants)

def threaded_future_caller(func: callable, iterable: Iterable, *constants):
    return future_caller(func, iterable, ThreadPoolExecutor, *constants)

def future_caller(func: callable, iterable, Executor=ThreadPoolExecutor, *constants):
    """Wrapper to map the callable to the iteratable using the `constants` as partial args for each call.
    """
    with Executor() as executor:
        fun = partial(func, *constants)
        print(fun.args)
        results = executor.map(fun, iterable)

    return tuple(results)


