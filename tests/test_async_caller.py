
from async_caller import future_caller, ThreadPoolExecutor

def sum_four(a, b, c, d):
    return a + b + c + d


def test_future_caller():
    a, b, c = 1, 2, 3

    all_d_values = [1, 2, 3, 4]

    res = future_caller(sum_four, all_d_values, ThreadPoolExecutor, *[a, b, c])
    assert res == (7, 8, 9, 10)