# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nmwc_model.boundary import periodic, relax

from tests import utils


def test_periodic_colocated():
    # load reference data
    ds = np.load("baseline_datasets/test_boundary/test_periodic_colocated.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    phi = ds["phi"]

    # run user code
    phi = periodic(phi, nx, nb)

    # validation
    for i in range(nb):
        utils.compare_arrays(phi[i, :], phi[nx + i, :])
        utils.compare_arrays(phi[-i - 1, :], phi[-nx - i - 1, :])


def test_periodic_staggered():
    # load reference data
    ds = np.load("baseline_datasets/test_boundary/test_periodic_staggered.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    phi = ds["phi"]

    # run user code
    phi = periodic(phi, nx, nb)

    # validation
    for i in range(nb):
        utils.compare_arrays(phi[i, :], phi[nx + i, :])
        utils.compare_arrays(phi[-i - 1, :], phi[-nx - i - 1, :])


def test_relax_1d():
    # load reference data
    ds = np.load("baseline_datasets/test_boundary/test_relax_1d.npz")

    # run user code
    phi = relax(ds["phi"], ds["nx"], ds["nb"], ds["phi1"], ds["phi2"])

    # validation
    utils.compare_arrays(phi, ds["phi_val"])


def test_relax_2d():
    # load reference data
    ds = np.load("baseline_datasets/test_boundary/test_relax_2d.npz")

    # run user code
    phi = relax(ds["phi"], ds["nx"], ds["nb"], ds["phi1"], ds["phi2"])

    # validation
    utils.compare_arrays(phi, ds["phi_val"])


if __name__ == "__main__":
    pytest.main([__file__])
