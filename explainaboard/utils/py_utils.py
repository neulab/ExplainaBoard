from __future__ import annotations

import functools
import itertools
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def zip_dict(*dicts):
    """Iterate over items of dictionaries grouped by their keys."""
    for key in set(itertools.chain(*dicts)):  # set merge all keys
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


def sort_dict(dict_obj, flag="key"):
    sorted_dict_obj = []
    if flag == "key":
        sorted_dict_obj = sorted(dict_obj.items(), key=lambda item: item[0])
    elif flag == "value":
        # dict_bucket2span_
        sorted_dict_obj = sorted(
            dict_obj.items(), key=lambda item: len(item[1]), reverse=True
        )
    return dict(sorted_dict_obj)


def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapped
