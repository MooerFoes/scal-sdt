from typing import Callable, Any


def try_then_default(f: Callable[[], Any], default=None):
    try:
        return f()
    except:
        return default


def timeit(f: Callable[[], Any]):
    import time
    start = time.perf_counter()
    result = f()
    t = time.perf_counter() - start
    return result, t
