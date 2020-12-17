import functools
import hashlib
import numbers
import operator
from typing import Tuple
import numpy as np


def _create_ctx(cache):
    ctx = dict()
    if "data" in cache.keys():
        ctx["data"] = cache.get("data")
    if "shape" in cache.keys():
        ctx["shape"] = cache.get("shape")
    if "targets" in cache.keys():
        ctx["targets"] = cache.get("targets")
    return ctx


def persistent_hash(to_hash, algorithm=hashlib.md5):
    def add_str(a, b):
        return operator.add(str(persistent_hash(str(a), algorithm)), str(persistent_hash(str(b), algorithm)))

    if isinstance(to_hash, Tuple):
        to_hash = functools.reduce(add_str, to_hash)
    return int(algorithm(to_hash.encode("utf-8")).hexdigest(), 16)


def get_by_tag(tag=None, value=None, experiment_id=None):
    from pypads.app.pypads import get_current_pads
    pads = get_current_pads()
    if not experiment_id:
        experiment_id = pads.api.active_run().info.experiment_id

    runs = pads.mlf.list_run_infos(experiment_id)
    selection = []
    for run in runs:
        run = pads.mlf.get_run(run.run_id)
        if tag:
            tags = run.data.tags
            if value and tag in tags:
                if tags[tag] == value:
                    selection.append(run)
            else:
                selection.append(run)
        else:
            selection.append(run)
    return selection


def _len(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
        else:
            return len(x)
    else:
        return len(x)


def _tolist(x):
    """Return a list representation of any array-like/Iterable x """
    if x is None:
        return None
    elif isinstance(x, list):
        return x
    elif hasattr(x, 'tolist'):
        return x.tolist()
    elif hasattr(x, 'values'):
        return x.values.tolist()
    else:
        try:
            list(x)
        except Exception as e:
            raise TypeError("%s cannot be converted to a list due to error: %s" % (type(x), str(e)))


def validate_type(value):
    if "int" in str(type(value)):
        return int(value)
    elif "float" in str(type(value)):
        return float(value)
    elif "str" in str(type(value)):
        return str(value)
    elif "bool" in str(type(value)):
        return bool(value)
    elif isinstance(value, tuple):
        value_ = []
        for v in value:
            value_.append(validate_type(v))
        return tuple(value_)
    elif "array" in str(type(value)) or isinstance(value, list):
        value_ = []
        for v in value:
            value_.append(validate_type(v))
        return value_
    return value