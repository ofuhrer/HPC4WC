# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nmwc_model import diagnostics

from tests import utils


def test_pressure():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_pressure.npz")

    # prepare input data
    prs = np.zeros_like(ds["prs_val"])

    # hack
    diagnostics.__dict__["dth"] = ds["dth"]
    diagnostics.__dict__["nz"] = ds["nz"]

    # run user code
    prs = diagnostics.diag_pressure(ds["prs0"], prs, ds["snew"])

    # validation
    utils.compare_arrays(prs, ds["prs_val"])


def test_montgomery():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_montgomery.npz")

    # prepare input data
    mtg = np.zeros_like(ds["mtg_val"])

    # hack
    diagnostics.__dict__["dth"] = ds["dth"]
    diagnostics.__dict__["nz"] = ds["nz"]

    # run user code
    exn, mtg = diagnostics.diag_montgomery(
        ds["prs"], mtg, ds["th0"], ds["topo"], ds["topofact"]
    )

    # validation
    utils.compare_arrays(exn, ds["exn_val"])
    utils.compare_arrays(mtg, ds["mtg_val"])


def test_height():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_height.npz")

    # prepare input data
    zht = np.zeros_like(ds["zht_val"])

    # hack
    diagnostics.__dict__["nz"] = ds["nz"]

    # run user code
    zht = diagnostics.diag_height(
        ds["prs"], ds["exn"], zht, ds["th0"], ds["topo"], ds["topofact"]
    )

    # validation
    utils.compare_arrays(zht, ds["zht_val"])


def test_density_and_temperature():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_density_and_temperature.npz")

    # hack
    diagnostics.__dict__["nz"] = ds["nz"]

    # run user code
    rho, temp = diagnostics.diag_density_and_temperature(
        ds["s"], ds["exn"], ds["zht"], ds["th0"]
    )

    # validation
    utils.compare_arrays(rho, ds["rho_val"])
    utils.compare_arrays(temp, ds["temp_val"])


if __name__ == "__main__":
    pytest.main([__file__])
