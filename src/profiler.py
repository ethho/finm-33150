import os
from typing import Optional, Callable, List
from functools import wraps
import logging
from datetime import datetime
import json
import time
import tracemalloc


def profiler(flavor: Optional[str] = 'wall_clock', log_dir: Optional[str] = './', 
             log_stub: Optional[str] = None, show_prof: bool = False,
             cumulative: bool = False) -> Callable:
    """Decorates `func` with Dask memory and thread profiling. This function
    returns a decorator, so use like:
    @profiler()
    def my_func():
        pass

    Author: Ethan Ho @ethho
    Source: https://gist.github.com/ethho/e63f9ac68b7a362c454a40233da5ddff
    """

    assert os.path.isdir(log_dir)
    if log_stub is None:
        log_stub = _get_timestamp()

    def base_fp(name, ext):
        fp = os.path.join(log_dir, f"{log_stub}_{name}.{ext}")
        print(f"Saving profiling report to '{fp}'...")
        return fp

    def mem_prof(func):
        """Decorator
        """
        @wraps(func)
        def with_prof(*args, **kwargs):
            start = time.time()
            tracemalloc.start()
            result = func(*args, **kwargs)

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('filename', cumulative=cumulative)
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

            elapsed = time.time() - start
            print(f"'{func.__name__}' took {elapsed:0.2f} seconds")
            return result
        return with_prof

    def wall_clock(func):
        """Decorator
        """
        @wraps(func)
        def with_prof(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"'{func.__name__}' took {elapsed:0.2f} seconds")
            return result
        return with_prof

    # choose decorator
    allowed_flavors = ('mem', 'wall_clock')
    if flavor in ('mem', 'ram'):
        decorator = mem_prof
    elif flavor in ('timer', 'wall_clock'):
        decorator = wall_clock
    else:
        raise ValueError(f"please select a `flavor` from: {allowed_flavors}")

    return decorator


def _get_timestamp():
    return datetime.today().strftime('%Y%m%d_%H%M%S')
