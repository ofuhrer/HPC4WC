# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nmwc_model import prognostics

from tests import utils


def test_prog_isendens():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_isendens.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["idthdt"] = 0
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    snew = prognostics.prog_isendens(ds["sold"], ds["snow"], ds["unow"], dtdx)

    # validation
    utils.compare_arrays(snew, ds["snew_val"])


def test_prog_isendens_idthdt():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_isendens_idthdt.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["dt"] = ds["dt"]
    prognostics.__dict__["dth"] = ds["dth"]
    prognostics.__dict__["idthdt"] = 1
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    snew = prognostics.prog_isendens(
        ds["sold"], ds["snow"], ds["unow"], dtdx, dthetadt=ds["dthetadt"]
    )

    # validation
    utils.compare_arrays(snew, ds["snew_val"])


def test_prog_velocity():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_velocity.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["idthdt"] = 0
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    unew = prognostics.prog_velocity(ds["uold"], ds["unow"], ds["mtg"], dtdx)

    # validation
    utils.compare_arrays(unew, ds["unew_val"])


def test_prog_velocity_idthdt():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_velocity_idthdt.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["dt"] = ds["dt"]
    prognostics.__dict__["dth"] = ds["dth"]
    prognostics.__dict__["idthdt"] = 1
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    unew = prognostics.prog_velocity(
        ds["uold"], ds["unow"], ds["mtg"], dtdx, dthetadt=ds["dthetadt"]
    )

    # validation
    utils.compare_arrays(unew, ds["unew_val"])


def test_prog_moisture():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_moisture.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["idthdt"] = 0
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    qvnew, qcnew, qrnew = prognostics.prog_moisture(
        ds["unow"],
        ds["qvold"],
        ds["qcold"],
        ds["qrold"],
        ds["qvnow"],
        ds["qcnow"],
        ds["qrnow"],
        dtdx,
    )

    # validation
    utils.compare_arrays(qvnew, ds["qvnew_val"])
    utils.compare_arrays(qcnew, ds["qcnew_val"])
    utils.compare_arrays(qrnew, ds["qrnew_val"])


def test_prog_moisture_idthdt():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_moisture_idthdt.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["idthdt"] = 1
    prognostics.__dict__["dt"] = ds["dt"]
    prognostics.__dict__["dth"] = ds["dth"]
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    qvnew, qcnew, qrnew = prognostics.prog_moisture(
        ds["unow"],
        ds["qvold"],
        ds["qcold"],
        ds["qrold"],
        ds["qvnow"],
        ds["qcnow"],
        ds["qrnow"],
        dtdx,
        dthetadt=ds["dthetadt"],
    )

    # validation
    utils.compare_arrays(qvnew, ds["qvnew_val"])
    utils.compare_arrays(qcnew, ds["qcnew_val"])
    utils.compare_arrays(qrnew, ds["qrnew_val"])


def test_prog_numdens():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_numdens.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["idthdt"] = 0
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    ncnew, nrnew = prognostics.prog_numdens(
        ds["unow"], ds["ncold"], ds["nrold"], ds["ncnow"], ds["nrnow"], dtdx
    )

    # validation
    utils.compare_arrays(ncnew, ds["ncnew_val"])
    utils.compare_arrays(nrnew, ds["nrnew_val"])


def test_prog_numdens_idthdt():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_numdens_idthdt.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    # hack
    prognostics.__dict__["idthdt"] = 1
    prognostics.__dict__["dt"] = ds["dt"]
    prognostics.__dict__["dth"] = ds["dth"]
    prognostics.__dict__["nb"] = ds["nb"]
    prognostics.__dict__["nx"] = ds["nx"]
    prognostics.__dict__["nz"] = ds["nz"]

    # run user code
    ncnew, nrnew = prognostics.prog_numdens(
        ds["unow"],
        ds["ncold"],
        ds["nrold"],
        ds["ncnow"],
        ds["nrnow"],
        dtdx,
        dthetadt=ds["dthetadt"],
    )

    # validation
    utils.compare_arrays(ncnew, ds["ncnew_val"])
    utils.compare_arrays(nrnew, ds["nrnew_val"])


if __name__ == "__main__":
    pytest.main([__file__])
