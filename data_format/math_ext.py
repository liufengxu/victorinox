# coding=utf-8
from hashlib import md5
from struct import unpack


def diversity(list_):
    n = len(list_)
    if n < 2:
        return 0
    val = sum([v * (v - 1) / 2 for v in value_counts(list_).values() if v > 1])
    return 1 - val * 2.0 / (n * (n - 1))


def value_counts(list_, normalize=False):
    res = {}
    delta = 1.0 / len(list_) if normalize else 1.0
    for e in list_:
        if e in res:
            res[e] += delta
        else:
            res[e] = delta
    return res


def pdf(list_):
    return value_counts(list_, normalize=True)


def median(list_):
    list_ = sorted(list_)
    n = len(list_)
    return (list_[n >> 1] + list_[(n - 1) >> 1]) / 2.0


def argtop(x, k):
    """
    :param x: numpy array
    :param k: integer, typically k << len(x)
    :return: the indices of the top k largest elements
    """
    idx = x.argpartition(-k)[-k:]
    idx = idx[x[idx].argsort()]
    return idx[::-1]


def jhash(str_):
    """
    same as java hashCode
    In Python 3.x, the result of hash() will only be consistent within a process,
    not across python invocations
    """
    if not str_:
        return 0

    h = 0
    for c in str_:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def lhash(str_):
    """
    hash unicode string to Long, ~4us per call
    """
    if not str_:
        return 0

    o = md5()
    o.update(str_.encode('utf-8'))
    bytes_ = o.digest()  # 16 bytes
    tup = unpack("<qq", bytes_)  # little-endian long
    return tup[0] ^ tup[1]


def chunks(a, chunk_size):
    return [a[x: x + chunk_size] for x in range(0, len(a), chunk_size)]


def zscore(x, y):
    import numpy as np
    import scipy.stats as sts
    x = np.array(x)
    y = np.array(y)
    val = (np.mean(x) - np.mean(y)) / np.sqrt(np.var(x) / x.shape[0] + np.var(y) / y.shape[0])
    return val, sts.norm.cdf(-np.abs(val)) * 2

