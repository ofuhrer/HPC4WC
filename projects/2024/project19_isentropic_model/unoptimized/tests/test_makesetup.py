# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nmwc_model import makesetup

from tests import utils


def test_maketopo():
    # load reference data
    ds = np.load("baseline_datasets/test_makesetup/test_maketopo.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    dx = ds["dx"]
    topomx = ds["topomx"]
    topowd = ds["topowd"]
    topo_val = ds["topo_val"]

    # prepare input data
    topo = np.zeros((nx + 2 * nb, 1))

    # hack
    makesetup.__dict__["dx"] = dx
    makesetup.__dict__["topomx"] = topomx
    makesetup.__dict__["topowd"] = topowd

    # run user code
    topo = makesetup.maketopo(topo, nx + 2 * nb)

    # validation
    utils.compare_arrays(topo, topo_val)


def test_makeprofile_dry():
    # load reference data
    ds = np.load("baseline_datasets/test_makesetup/test_makeprofile_dry.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    nz = ds["nz"]
    bv00 = ds["bv00"]
    dth = ds["dth"]
    dx = ds["dx"]
    th00 = ds["th00"]
    u00 = ds["u00"]
    z00 = ds["z00"]
    th0_val = ds["th0_val"]
    exn0_val = ds["exn0_val"]
    prs0_val = ds["prs0_val"]
    z0_val = ds["z0_val"]
    mtg0_val = ds["mtg0_val"]
    s0_val = ds["s0_val"]
    u0_val = ds["u0_val"]

    # prepare input data
    sold = np.zeros((nx + 2 * nb, nz))
    uold = np.zeros((nx + 1 + 2 * nb, nz))

    # hack
    makesetup.__dict__["bv00"] = bv00
    makesetup.__dict__["dth"] = dth
    makesetup.__dict__["dx"] = dx
    makesetup.__dict__["imoist"] = 0
    makesetup.__dict__["ishear"] = 0
    makesetup.__dict__["nz"] = nz
    makesetup.__dict__["th00"] = th00
    makesetup.__dict__["u00"] = u00
    makesetup.__dict__["z00"] = z00

    # run user code
    out_list = makesetup.makeprofile(sold, uold)

    # validation
    assert len(out_list) == 13
    utils.compare_arrays(out_list[0], th0_val)
    utils.compare_arrays(out_list[1], exn0_val)
    utils.compare_arrays(out_list[2], prs0_val)
    utils.compare_arrays(out_list[3], z0_val)
    utils.compare_arrays(out_list[4], mtg0_val)
    utils.compare_arrays(out_list[5], s0_val)
    utils.compare_arrays(out_list[6], u0_val)
    assert out_list[7].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[7], s0_val[np.newaxis, :])
    assert out_list[8].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[8], s0_val[np.newaxis, :])
    assert out_list[9].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[9], u0_val[np.newaxis, :])
    assert out_list[10].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[10], u0_val[np.newaxis, :])
    assert out_list[11].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[11], mtg0_val[np.newaxis, :])
    assert out_list[12].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[12], mtg0_val[np.newaxis, :])


def test_makeprofile_dry_shear():
    # load reference data
    ds = np.load("baseline_datasets/test_makesetup/test_makeprofile_dry_shear.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    nz = ds["nz"]
    bv00 = ds["bv00"]
    dth = ds["dth"]
    dx = ds["dx"]
    k_shl = ds["k_shl"]
    k_sht = ds["k_sht"]
    th00 = ds["th00"]
    u00 = ds["u00"]
    u00_sh = ds["u00_sh"]
    z00 = ds["z00"]
    th0_val = ds["th0_val"]
    exn0_val = ds["exn0_val"]
    prs0_val = ds["prs0_val"]
    z0_val = ds["z0_val"]
    mtg0_val = ds["mtg0_val"]
    s0_val = ds["s0_val"]
    u0_val = ds["u0_val"]

    # prepare input data
    sold = np.zeros((nx + 2 * nb, nz))
    uold = np.zeros((nx + 1 + 2 * nb, nz))

    # hack
    makesetup.__dict__["bv00"] = bv00
    makesetup.__dict__["dth"] = dth
    makesetup.__dict__["dx"] = dx
    makesetup.__dict__["imoist"] = 0
    makesetup.__dict__["ishear"] = 1
    makesetup.__dict__["k_shl"] = k_shl
    makesetup.__dict__["k_sht"] = k_sht
    makesetup.__dict__["nz"] = nz
    makesetup.__dict__["th00"] = th00
    makesetup.__dict__["u00"] = u00
    makesetup.__dict__["u00_sh"] = u00_sh
    makesetup.__dict__["z00"] = z00

    # run user code
    out_list = makesetup.makeprofile(sold, uold)

    # validation
    assert len(out_list) == 13
    utils.compare_arrays(out_list[0], th0_val)
    utils.compare_arrays(out_list[1], exn0_val)
    utils.compare_arrays(out_list[2], prs0_val)
    utils.compare_arrays(out_list[3], z0_val)
    utils.compare_arrays(out_list[4], mtg0_val)
    utils.compare_arrays(out_list[5], s0_val)
    utils.compare_arrays(out_list[6], u0_val)
    assert out_list[7].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[7], s0_val[np.newaxis, :])
    assert out_list[8].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[8], s0_val[np.newaxis, :])
    assert out_list[9].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[9], u0_val[np.newaxis, :])
    assert out_list[10].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[10], u0_val[np.newaxis, :])
    assert out_list[11].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[11], mtg0_val[np.newaxis, :])
    assert out_list[12].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[12], mtg0_val[np.newaxis, :])


def test_makeprofile_moist():
    # load reference data
    ds = np.load("baseline_datasets/test_makesetup/test_makeprofile_moist.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    nz = ds["nz"]
    bv00 = ds["bv00"]
    dth = ds["dth"]
    dx = ds["dx"]
    th00 = ds["th00"]
    u00 = ds["u00"]
    z00 = ds["z00"]
    th0_val = ds["th0_val"]
    exn0_val = ds["exn0_val"]
    prs0_val = ds["prs0_val"]
    z0_val = ds["z0_val"]
    mtg0_val = ds["mtg0_val"]
    s0_val = ds["s0_val"]
    u0_val = ds["u0_val"]
    qv0_val = ds["qv0_val"]

    # prepare input data
    sold = np.zeros((nx + 2 * nb, nz))
    uold = np.zeros((nx + 1 + 2 * nb, nz))
    qvold = np.zeros((nx + 2 * nb, nz))
    qvnow = np.zeros((nx + 2 * nb, nz))
    qcold = np.zeros((nx + 2 * nb, nz))
    qcnow = np.zeros((nx + 2 * nb, nz))
    qrold = np.zeros((nx + 2 * nb, nz))
    qrnow = np.zeros((nx + 2 * nb, nz))
    ncold = np.zeros((nx + 2 * nb, nz))
    ncnow = np.zeros((nx + 2 * nb, nz))
    nrold = np.zeros((nx + 2 * nb, nz))
    nrnow = np.zeros((nx + 2 * nb, nz))

    #
    # imicrophys = 0,1
    #
    # hack
    makesetup.__dict__["bv00"] = bv00
    makesetup.__dict__["dth"] = dth
    makesetup.__dict__["dx"] = dx
    makesetup.__dict__["imicrophys"] = 0
    makesetup.__dict__["imoist"] = 1
    makesetup.__dict__["ishear"] = 0
    makesetup.__dict__["nz"] = nz
    makesetup.__dict__["th00"] = th00
    makesetup.__dict__["u00"] = u00
    makesetup.__dict__["z00"] = z00

    # run user code
    out_list = makesetup.makeprofile(
        sold, uold, qvold, qvnow, qcold, qcnow, qrold, qrnow
    )

    # validation
    assert len(out_list) == 22
    utils.compare_arrays(out_list[0], th0_val)
    utils.compare_arrays(out_list[1], exn0_val)
    utils.compare_arrays(out_list[2], prs0_val)
    utils.compare_arrays(out_list[3], z0_val)
    utils.compare_arrays(out_list[4], mtg0_val)
    utils.compare_arrays(out_list[5], s0_val)
    utils.compare_arrays(out_list[6], u0_val)
    assert out_list[7].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[7], s0_val[np.newaxis, :])
    assert out_list[8].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[8], s0_val[np.newaxis, :])
    assert out_list[9].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[9], u0_val[np.newaxis, :])
    assert out_list[10].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[10], u0_val[np.newaxis, :])
    assert out_list[11].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[11], mtg0_val[np.newaxis, :])
    assert out_list[12].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[12], mtg0_val[np.newaxis, :])
    utils.compare_arrays(out_list[13], qv0_val)
    assert out_list[14].shape == (nz,)
    utils.compare_arrays(out_list[14], 0.0)
    assert out_list[15].shape == (nz,)
    utils.compare_arrays(out_list[15], 0.0)
    assert out_list[16].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[16], qv0_val[np.newaxis, :])
    assert out_list[17].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[17], qv0_val[np.newaxis, :])
    assert out_list[18].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[18], 0.0)
    assert out_list[19].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[19], 0.0)
    assert out_list[20].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[20], 0.0)
    assert out_list[21].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[21], 0.0)

    #
    # imicrophys = 2
    #
    # hack
    makesetup.__dict__["imicrophys"] = 2

    out_list = makesetup.makeprofile(
        sold, uold, qvold, qvnow, qcold, qcnow, qrold, qrnow, ncold, ncnow, nrold, nrnow
    )

    assert len(out_list) == 26
    utils.compare_arrays(out_list[0], th0_val)
    utils.compare_arrays(out_list[1], exn0_val)
    utils.compare_arrays(out_list[2], prs0_val)
    utils.compare_arrays(out_list[3], z0_val)
    utils.compare_arrays(out_list[4], mtg0_val)
    utils.compare_arrays(out_list[5], s0_val)
    utils.compare_arrays(out_list[6], u0_val)
    assert out_list[7].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[7], s0_val[np.newaxis, :])
    assert out_list[8].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[8], s0_val[np.newaxis, :])
    assert out_list[9].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[9], u0_val[np.newaxis, :])
    assert out_list[10].shape == (nx + 1 + 2 * nb, nz)
    utils.compare_arrays(out_list[10], u0_val[np.newaxis, :])
    assert out_list[11].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[11], mtg0_val[np.newaxis, :])
    assert out_list[12].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[12], mtg0_val[np.newaxis, :])
    utils.compare_arrays(out_list[13], qv0_val)
    assert out_list[14].shape == (nz,)
    utils.compare_arrays(out_list[14], 0.0)
    assert out_list[15].shape == (nz,)
    utils.compare_arrays(out_list[15], 0.0)
    assert out_list[16].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[16], qv0_val[np.newaxis, :])
    assert out_list[17].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[17], qv0_val[np.newaxis, :])
    assert out_list[18].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[18], 0.0)
    assert out_list[19].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[19], 0.0)
    assert out_list[20].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[20], 0.0)
    assert out_list[21].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[21], 0.0)
    assert out_list[22].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[22], 0.0)
    assert out_list[23].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[23], 0.0)
    assert out_list[24].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[24], 0.0)
    assert out_list[25].shape == (nx + 2 * nb, nz)
    utils.compare_arrays(out_list[25], 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
