# -*- coding: utf-8 -*-
import numpy as np

from tests import conf


def get_random_int(min_value=0, max_value=100):
    return np.random.randint(min_value, max_value, dtype=int)


def get_random_nx(min_value=conf.nx_min, max_value=conf.nx_max):
    return get_random_int(min_value, max_value)


def get_random_nb(min_value=conf.nb_min, max_value=conf.nb_max):
    return get_random_int(min_value, max_value)


def get_random_nz(min_value=conf.nz_min, max_value=conf.nz_max):
    return get_random_int(min_value, max_value)


def get_random_float(min_value=conf.field_min, max_value=conf.field_max):
    return min_value + (max_value - min_value) * np.random.rand(1).item()


def get_random_positive_float(max_value=conf.field_max):
    out = get_random_float(min_value=-max_value, max_value=max_value)
    return out if out > 0.0 else -out


def get_random_array_1d(n, min_value=conf.field_min, max_value=conf.field_max, sort=False):
    out = min_value + (max_value - min_value) * np.random.rand(n)
    return sorted(out) if sort else out


def get_random_array_2d(ni, nk, min_value=conf.field_min, max_value=conf.field_max):
    return min_value + (max_value - min_value) * np.random.rand(ni, nk)


def compare_arrays(a, b):
    assert np.allclose(a, b, equal_nan=True)
