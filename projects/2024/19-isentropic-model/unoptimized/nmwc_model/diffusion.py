import numpy as np

from nmwc_model.namelist import (
    idbg,
    nb,
    nx,
    imoist_diff,
    imoist,
    imicrophys,
    irelax,
)  # import global variables
from nmwc_model.boundary import periodic


def horizontal_diffusion(
    tau, unew, snew, qvnew=None, qcnew=None, qrnew=None, ncnew=None, nrnew=None
):
    """ Horizontal diffusion.

    Apply horizontal diffusion to all prognostic variables.
    The diffusivity may be increased towards the top of the domain to mimic
    the action of a gravity wave absorber.

    Parameters
    ----------
    tau : np.ndarray
        1-d array collecting the values of the diffusivity on each vertical level.
    unew : np.ndarray
        Staggered horizontal velocity in [m s^-1].
    snew : np.ndarray
        Isentropic density in [kg m^-2 K^-1].
    qvnew : ``np.ndarray``, optional
        Mass fraction of water vapor in [g g^-1].
    qcnew : ``np.ndarray``, optional
        Mass fraction of cloud liquid water in [g g^-1].
    qrnew : ``np.ndarray``, optional
        Mass fraction of precipitation water in [g g^-1].
    ncnew : ``np.ndarrya``, optional
        Number density of cloud liquid water in [g g^-1].
    nrnew : ``np.ndarrya``, optional
        Number density of precipitation water in [g g^-1].

    Returns
    -------
    unew : np.ndarray
        The input array ``unew`` after that diffusion has been applied.
    snew : np.ndarray
        The input array ``snew`` after that diffusion has been applied.
    qvnew : ``np.ndarray``, optional
        The input array ``qvnew`` (if given) after that diffusion has been applied.
    qcnew : ``np.ndarray``, optional
        The input array ``qcnew`` (if given) after that diffusion has been applied.
    qrnew : ``np.ndarray``, optional
        The input array ``qrnew`` (if given) after that diffusion has been applied.
    ncnew : ``np.ndarray``, optional
        The input array ``ncnew`` (if given) after that diffusion has been applied.
    nrnew : ``np.ndarray``, optional
        The input array ``nrnew`` (if given) after that diffusion has been applied.
    """
    ind = tau > 0

    if idbg == 1 and np.size(ind) > 0:
        print("Apply diffusion and gravity wave absorber ...\n")

    taumat = np.ones_like(unew) * tau

    if np.all(tau <= 0):
        raise ValueError("All entries of the diffusivity matrix are negative.")
    else:
        sel = taumat > 0

        i = np.arange(nb, nx + 1 + nb)
        unew[i, :] = (
            unew[i, :]
            + taumat[i, :] * (unew[i - 1, :] - 2.0 * unew[i, :] + unew[i + 1, :]) / 4.0
        ) * sel[i, :] + unew[i, :] * ~sel[i, :]

        i = np.arange(nb, nx + nb)
        snew[i, :] = (
            snew[i, :]
            + taumat[i, :] * (snew[i - 1, :] - 2.0 * snew[i, :] + snew[i + 1, :]) / 4.0
        ) * sel[i, :] + snew[i, :] * ~sel[i, :]

        if imoist == 1 and imoist_diff == 1:
            qvnew[i, :] = (
                qvnew[i, :]
                + taumat[i, :]
                * (qvnew[i - 1, :] - 2.0 * qvnew[i, :] + qvnew[i + 1, :])
                / 4.0
            ) * sel[i, :] + qvnew[i, :] * ~sel[i, :]

            qcnew[i, :] = (
                qcnew[i, :]
                + taumat[i, :]
                * (qcnew[i - 1, :] - 2.0 * qcnew[i, :] + qcnew[i + 1, :])
                / 4.0
            ) * sel[i, :] + qcnew[i, :] * ~sel[i, :]

            qrnew[i, :] = (
                qrnew[i, :]
                + taumat[i, :]
                * (qrnew[i - 1, :] - 2.0 * qrnew[i, :] + qrnew[i + 1, :])
                / 4.0
            ) * sel[i, :] + qrnew[i, :] * ~sel[i, :]

            if imicrophys == 2:
                nrnew[i, :] = (
                    nrnew[i, :]
                    + taumat[i, :]
                    * (nrnew[i - 1, :] - 2.0 * nrnew[i, :] + nrnew[i + 1, :])
                    / 4.0
                ) * sel[i, :] + nrnew[i, :] * ~sel[i, :]

                ncnew[i, :] = (
                    ncnew[i, :]
                    + taumat[i, :]
                    * (ncnew[i - 1, :] - 2.0 * ncnew[i, :] + ncnew[i + 1, :])
                    / 4.0
                ) * sel[i, :] + ncnew[i, :] * ~sel[i, :]

    # exchange periodic boundaries
    if irelax == 0:
        unew = periodic(unew, nx, nb)
        snew = periodic(snew, nx, nb)

        if imoist == 1 and imoist_diff == 1:
            qvnew = periodic(qvnew, nx, nb)
            qcnew = periodic(qcnew, nx, nb)
            qrnew = periodic(qrnew, nx, nb)

            if imicrophys == 2:
                ncnew = periodic(ncnew, nx, nb)
                nrnew = periodic(nrnew, nx, nb)

    if imoist == 0:
        return unew, snew
    elif imicrophys == 0 or imicrophys == 1:
        return unew, snew, qvnew, qcnew, qrnew
    elif imicrophys == 2:
        return unew, snew, qvnew, qcnew, qrnew, ncnew, nrnew


# END OF DIFFUSION.PY
