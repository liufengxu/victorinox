# coding=utf-8
"""
date time and logging utils
"""

import time
import datetime as dt
import warnings
from functools import wraps

from log import LOGGER

_FORMAT = "%Y-%m-%d %H:%M:%S"


def log_info(fmt, *args):
    """deprecated"""
    warnings.warn("log_info is deprecated, use LOGGER.info instead", DeprecationWarning,
                  stacklevel=2)
    if args and len(args) > 0:
        LOGGER.info(fmt.format(*args))
    else:
        LOGGER.info(fmt)


def fn_timer(fn):
    @wraps(fn)
    def function_timer(*args, **kwargs):
        LOGGER.info("Start running {0} ...".format(fn.__name__))
        t0 = time.time()
        result = fn(*args, **kwargs)
        t1 = time.time()
        LOGGER.info("Total time running {0}: {1} seconds".format(fn.__name__, round(t1 - t0, 3)))
        return result

    return function_timer


def dt_to_ts(d):
    return int(time.mktime(d.timetuple()))


def ts_to_dt(ts):
    return dt.datetime.fromtimestamp(ts)


def str_to_ts(str_, fmt=_FORMAT):
    t = time.strptime(str_, fmt)
    return int(time.mktime(t))


def str_to_dt(str_, fmt=_FORMAT):
    d = dt.datetime.strptime(str_, fmt)
    return d


def dt_to_str(dt_, fmt=_FORMAT):
    return dt_.strftime(fmt)


def now_ts():
    return int(time.time())
