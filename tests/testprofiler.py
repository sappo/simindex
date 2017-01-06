# -*- coding: utf-8 -*-
import pytest
from line_profiler import LineProfiler


def profile(follow=[]):
    if not pytest.config.getoption("--profile"):
        return pytest.mark.skip(reason="need --profile option to run")
    else:
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner
