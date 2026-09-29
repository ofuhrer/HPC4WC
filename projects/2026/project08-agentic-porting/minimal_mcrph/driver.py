"""Wires satad + the 5 processes + housekeeping into one timestep, matching
mo_nwp_gscp_interface.f90::nwp_microphysics's satad -> two_moment_mcrph ->
satad structure and mo_2mom_mcrph_main.f90::clouds_twomoment's process order.
See porting_plan.md's "Process Call Graph" section for the exact ordering
this mirrors.
"""

import dataclasses

import gt4py.next as gtx
import numpy as np
from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta

from minimal_mcrph import ccn_activation as ccn
from minimal_mcrph import config as cfg
from minimal_mcrph import particles as p
from minimal_mcrph.stencils import housekeeping as hk
from minimal_mcrph.stencils import ice_nucleation as icenuc
from minimal_mcrph.stencils import post_housekeeping as post_hk
from minimal_mcrph.stencils import processes as proc
from minimal_mcrph.stencils import satad as sa
from minimal_mcrph.stencils import unit_conversion as uc
from minimal_mcrph.stencils import vapor_deposition as vd

# CSV field list that gets density-converted (mo_2mom_prepare.f90; see
# porting_plan.md's "Negative-mixing-ratio clip, and the field list for
# prepare/post" section -- ninagi/ssat/qgl/qhl are NOT in this list for this
# config, their Fortran guards are all False).
_DENSITY_FIELDS = (
    "qv", "qc", "qnc", "qr", "qnr", "qi", "qni",
    "qs", "qns", "qg", "qng", "qh", "qnh", "ninact", "nccn", "ninpot",
)  # fmt: skip


@dataclasses.dataclass
class ColumnState:
    """One column, mixing-ratio space (as read from a CSV row). All fields
    are 1D numpy arrays of length nlev, except `hhl` (half-level heights,
    length nlev+1, from hhl.csv -- needed only for the post-housekeeping
    background-profile relaxation, mo_2mom_mcrph_driver.f90:489-522)."""

    hhl: np.ndarray
    rho: np.ndarray
    pres: np.ndarray
    w: np.ndarray
    tk: np.ndarray
    qv: np.ndarray
    qc: np.ndarray
    qnc: np.ndarray
    qr: np.ndarray
    qnr: np.ndarray
    qi: np.ndarray
    qni: np.ndarray
    qs: np.ndarray
    qns: np.ndarray
    qg: np.ndarray
    qng: np.ndarray
    qh: np.ndarray
    qnh: np.ndarray
    nccn: np.ndarray
    ninpot: np.ndarray
    ninagi: np.ndarray
    ninact: np.ndarray


@dataclasses.dataclass(frozen=True)
class _ParticleCoeffs:
    x_min: float
    x_max: float
    a_geo: float
    b_geo: float
    a_vel: float
    b_vel: float
    a_ven: float
    a_f: float
    b_f: float
    c_i: float
    c_z: float


def _particle_coeffs(particle: p.ParticleConfig) -> _ParticleCoeffs:
    c = p.setup_particle_coeffs(particle)
    return _ParticleCoeffs(
        x_min=particle.x_min, x_max=particle.x_max,
        a_geo=particle.a_geo, b_geo=particle.b_geo,
        a_vel=particle.a_vel, b_vel=particle.b_vel, a_ven=particle.a_ven,
        a_f=c.a_f, b_f=c.b_f, c_i=c.c_i, c_z=c.c_z,
    )  # fmt: skip


class Driver:
    """Precomputes coefficients once (matching Fortran's one-time
    init_2mom_scheme_once), then runs timesteps."""

    def __init__(self, *, backend=None):
        self._backend = backend
        self.cloud = _particle_coeffs(p.CLOUD)
        self.ice = _particle_coeffs(p.ICE)
        self.snow = _particle_coeffs(p.SNOW)
        self.graupel = _particle_coeffs(p.GRAUPEL)
        self.hail = _particle_coeffs(p.HAIL)
        self.ccn_table = ccn.build_ccn_table()

    def _field(self, values: np.ndarray) -> fa.CellKField[ta.wpfloat]:
        return gtx.as_field(
            [dims.CellDim, dims.KDim],
            np.asarray(values, dtype=np.float64).reshape(1, -1),
            allocator=self._backend,
        )

    def _zeros(self, nlev: int) -> fa.CellKField[ta.wpfloat]:
        return self._field(np.zeros(nlev))

    def _prog(self, program):
        return program.with_backend(self._backend) if self._backend is not None else program

    # Column names written by the Fortran's mo_stage_dump.f90, in its header order.
    # Recording under the same names is what lets a stage be compared without any
    # per-stage translation table on the Python side.
    _STAGE_FIELDS = (
        "rho", "pres", "w", "tk", "qv", "qc", "qnc", "qr", "qnr", "qi", "qni",
        "qs", "qns", "qg", "qng", "qh", "qnh", "nccn", "ninpot", "ninact",
    )  # fmt: skip

    def run_timestep(
        self, state: ColumnState, dt: float, stages: dict | None = None
    ) -> ColumnState:
        """Run one timestep.

        If ``stages`` is given, a snapshot of the column is stored into it at each of the
        process boundaries the Fortran's ``dump_stage`` writes, under the same names
        (``satad_pre``, ``prepare``, ``ccn``, ``default_n``, ``ice_nuc``, ``cloud_freeze``,
        ``vapor_dep``, ``ice_melt``, ``post``). Whole-column agreement alone cannot say
        which process a discrepancy came from; these snapshots are what make that
        answerable. Recording is off by default and costs nothing when unused.
        """
        nlev = state.rho.shape[0]
        dom = (0, 1, 0, nlev)

        def record(stage, **fields):
            """Snapshot the named boundary. Anything not passed is taken from `state`
            (fields the pipeline has not touched yet, or does not touch at all)."""
            if stages is None:
                return
            snap = {}
            for name in self._STAGE_FIELDS:
                value = fields.get(name, getattr(state, name, None))
                if value is None:
                    value = np.zeros(nlev)
                elif hasattr(value, "asnumpy"):
                    value = value.asnumpy().flatten()
                snap[name] = np.asarray(value, dtype=np.float64).copy()
            stages[stage] = snap

        rho = self._field(state.rho)
        pres = self._field(state.pres)
        w = self._field(state.w)
        tk = self._field(state.tk)
        qv = self._field(state.qv)
        qc = self._field(state.qc)

        # --- satad (before) ---
        out_tk, out_qv, out_qc = self._zeros(nlev), self._zeros(nlev), self._zeros(nlev)
        self._prog(sa.satad_program)(tk, qv, qc, rho, out_tk, out_qv, out_qc, *dom, offset_provider={})
        tk, qv, qc = out_tk, out_qv, out_qc
        record("satad_pre", tk=tk, qv=qv, qc=qc)

        # --- two_moment_mcrph ---
        # negative clip in mixing-ratio space (mo_2mom_mcrph_driver.f90:312-318)
        qr = self._field(np.maximum(state.qr, 0.0))
        qi = self._field(np.maximum(state.qi, 0.0))
        qs = self._field(np.maximum(state.qs, 0.0))
        qg = self._field(np.maximum(state.qg, 0.0))
        qh = self._field(np.maximum(state.qh, 0.0))
        qnc, qnr, qni, qns, qng, qnh = (
            self._field(x) for x in (state.qnc, state.qnr, state.qni, state.qns, state.qng, state.qnh)
        )  # fmt: skip
        nccn, ninpot, ninagi, ninact = (
            self._field(x) for x in (state.nccn, state.ninpot, state.ninagi, state.ninact)
        )  # fmt: skip

        # density corrections (mo_2mom_mcrph_driver.f90:355-365)
        rhocorr, rhocld = self._zeros(nlev), self._zeros(nlev)
        self._prog(uc.compute_density_corrections)(rho, rhocorr, rhocld, *dom, offset_provider={})
        rho_r_np = 1.0 / state.rho
        rho_r = self._field(rho_r_np)

        # prepare_twomoment: mixing ratio -> density
        density_fields = {
            "qv": qv, "qc": qc, "qnc": qnc, "qr": qr, "qnr": qnr, "qi": qi, "qni": qni,
            "qs": qs, "qns": qns, "qg": qg, "qng": qng, "qh": qh, "qnh": qnh,
            "ninact": ninact, "nccn": nccn, "ninpot": ninpot,
        }  # fmt: skip
        uc.convert_fields(
            list(density_fields.values()), rho,
            horizontal_start=0, horizontal_end=1, vertical_start=0, vertical_end=nlev,
            backend=self._backend,
        )  # fmt: skip
        qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, ninact, nccn, ninpot = (
            density_fields[name] for name in _DENSITY_FIELDS
        )

        # prepare's size-clip + zero-n-where-q-tiny housekeeping (mo_2mom_prepare.f90:140-162)
        for q, n, x_min, x_max in (
            (qr, qnr, p.RAIN.x_min, p.RAIN.x_max),
            (qi, qni, self.ice.x_min, self.ice.x_max),
            (qs, qns, self.snow.x_min, self.snow.x_max),
            (qg, qng, self.graupel.x_min, self.graupel.x_max),
            (qh, qnh, self.hail.x_min, self.hail.x_max),
        ):
            self._prog(hk.clip_number_concentration)(q, n, x_min, x_max, *dom, offset_provider={})

        for q, n in ((qc, qnc), (qr, qnr), (qi, qni), (qs, qns), (qg, qng), (qh, qnh)):
            self._prog(hk.zero_n_where_q_tiny)(q, n, *dom, offset_provider={})

        _live = dict(
            tk=tk, qv=qv, qc=qc, qnc=qnc, qr=qr, qnr=qnr, qi=qi, qni=qni,
            qs=qs, qns=qns, qg=qg, qng=qng, qh=qh, qnh=qnh,
            nccn=nccn, ninpot=ninpot, ninact=ninact,
        )  # fmt: skip
        record("prepare", **_live)

        # save q_vap_old, q_liq_old (mo_2mom_mcrph_driver.f90:392-405, no lprogmelt)
        q_vap_old = qv.asnumpy().flatten().copy()
        q_liq_old = (qc.asnumpy() + qr.asnumpy()).flatten().copy()

        # --- clouds_twomoment ---
        # 1. CCN activation (NumPy)
        cq_np, cn_np, qv_np, ncn_np = ccn.ccn_activation_sk_4d(
            rho=state.rho, w=state.w, cloud_q=qc.asnumpy().flatten(), cloud_n=qnc.asnumpy().flatten(),
            qv=qv.asnumpy().flatten(), n_cn=nccn.asnumpy().flatten(),
            ccn_ncn0=cfg.CCN_COEFFS.ncn0, ccn_wcb_min=cfg.CCN_COEFFS.wcb_min,
            ccn_r2=cfg.CCN_COEFFS.r2, ccn_lsigs=cfg.CCN_COEFFS.lsigs, ccn_etas=cfg.CCN_COEFFS.etas,
            cloud_x_min=p.CLOUD.x_min, table=self.ccn_table,
        )  # fmt: skip
        qc, qnc, qv, nccn = self._field(cq_np), self._field(cn_np), self._field(qv_np), self._field(ncn_np)
        _live.update(qc=qc, qnc=qnc, qv=qv, nccn=nccn)
        record("ccn", **_live)

        # 2. set_default_n (all 6 species)
        self._prog(hk.set_default_n)(
            qc, qnc, qi, qni, qr, qnr, qs, qns, qg, qng, qh, qnh, *dom, offset_provider={}
        )  # fmt: skip

        # 3. clip cloud.n (nuc_c_typ=8 != 0, always applies for this config)
        self._prog(hk.clip_number_concentration)(qc, qnc, p.CLOUD.x_min, p.CLOUD.x_max, *dom, offset_provider={})
        record("default_n", **_live)

        # 4. IN nucleation
        out = tuple(self._zeros(nlev) for _ in range(5))
        self._prog(icenuc.ice_nucleation_homhet_program)(
            self.ice.x_min, self.ice.x_max, tk, pres, w, qv, qc, qi, qni, ninact, ninpot,
            *out, *dom, offset_provider={},
        )  # fmt: skip
        qi, qni, qv, ninact, ninpot = out
        _live.update(qi=qi, qni=qni, qv=qv, ninact=ninact, ninpot=ninpot)
        record("ice_nuc", **_live)

        # 5. Cloud freezing
        out = tuple(self._zeros(nlev) for _ in range(4))
        self._prog(proc.cloud_freeze_program)(
            dt, self.cloud.c_z, p.CLOUD.x_max, p.CLOUD.x_min, tk, qc, qnc, qi, qni,
            *out, *dom, offset_provider={},
        )  # fmt: skip
        qc, qnc, qi, qni = out

        # 6. clip ice.n
        self._prog(hk.clip_number_concentration)(qi, qni, self.ice.x_min, self.ice.x_max, *dom, offset_provider={})
        _live.update(qc=qc, qnc=qnc, qi=qi, qni=qni)
        record("cloud_freeze", **_live)

        # 7. Vapor deposition (ice, snow, graupel, hail; rho_v = rhocorr for all four)
        out = tuple(self._zeros(nlev) for _ in range(9))
        self._prog(vd.vapor_dep_relaxation_program)(
            dt,
            self.ice.x_min, self.ice.x_max, self.ice.a_geo, self.ice.b_geo,
            self.ice.a_vel, self.ice.b_vel, self.ice.a_ven, self.ice.a_f, self.ice.b_f, self.ice.c_i,
            self.snow.x_min, self.snow.x_max, self.snow.a_geo, self.snow.b_geo,
            self.snow.a_vel, self.snow.b_vel, self.snow.a_ven, self.snow.a_f, self.snow.b_f, self.snow.c_i,
            self.graupel.x_min, self.graupel.x_max, self.graupel.a_geo, self.graupel.b_geo,
            self.graupel.a_vel, self.graupel.b_vel, self.graupel.a_ven,
            self.graupel.a_f, self.graupel.b_f, self.graupel.c_i,
            self.hail.x_min, self.hail.x_max, self.hail.a_geo, self.hail.b_geo,
            self.hail.a_vel, self.hail.b_vel, self.hail.a_ven, self.hail.a_f, self.hail.b_f, self.hail.c_i,
            tk, pres, qv,
            qi, qni, rhocorr,
            qs, qns, rhocorr,
            qg, qng, rhocorr,
            qh, qnh, rhocorr,
            *out, *dom, offset_provider={},
        )  # fmt: skip
        qi, qni, qs, qns, qg, qng, qh, qnh, qv = out
        _live.update(qi=qi, qni=qni, qs=qs, qns=qns, qg=qg, qng=qng, qh=qh, qnh=qnh, qv=qv)
        record("vapor_dep", **_live)

        # 8. Ice melting
        out = tuple(self._zeros(nlev) for _ in range(6))
        self._prog(proc.ice_melting_program)(
            p.CLOUD.x_max, self.ice.x_min, self.ice.x_max, tk, qi, qni, qc, qnc, qr, qnr,
            *out, *dom, offset_provider={},
        )  # fmt: skip
        qi, qni, qc, qnc, qr, qnr = out
        _live.update(qi=qi, qni=qni, qc=qc, qnc=qnc, qr=qr, qnr=qnr)
        record("ice_melt", **_live)

        # 9. final clipping, all six species + cloud hard cap
        self._prog(hk.clip_number_concentration)(qc, qnc, p.CLOUD.x_min, p.CLOUD.x_max, *dom, offset_provider={})
        self._prog(hk.clip_cloud_hard_cap)(qnc, *dom, offset_provider={})
        for q, n, x_min, x_max in (
            (qr, qnr, p.RAIN.x_min, p.RAIN.x_max),
            (qi, qni, self.ice.x_min, self.ice.x_max),
            (qs, qns, self.snow.x_min, self.snow.x_max),
            (qg, qng, self.graupel.x_min, self.graupel.x_max),
            (qh, qnh, self.hail.x_min, self.hail.x_max),
        ):
            self._prog(hk.clip_number_concentration)(q, n, x_min, x_max, *dom, offset_provider={})

        # --- latent heat update (writes back into tk in place) ---
        q_liq_new = (qc.asnumpy() + qr.asnumpy()).flatten()
        self._prog(uc.update_temperature)(
            tk, rho_r, self._field(q_vap_old), qv, self._field(q_liq_old), self._field(q_liq_new),
            *dom, offset_provider={},
        )  # fmt: skip

        # --- post_twomoment: density -> mixing ratio ---
        density_fields = {
            "qv": qv, "qc": qc, "qnc": qnc, "qr": qr, "qnr": qnr, "qi": qi, "qni": qni,
            "qs": qs, "qns": qns, "qg": qg, "qng": qng, "qh": qh, "qnh": qnh,
            "ninact": ninact, "nccn": nccn, "ninpot": ninpot,
        }  # fmt: skip
        uc.convert_fields(
            list(density_fields.values()), rho_r,
            horizontal_start=0, horizontal_end=1, vertical_start=0, vertical_end=nlev,
            backend=self._backend,
        )  # fmt: skip
        qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, ninact, nccn, ninpot = (
            density_fields[name] for name in _DENSITY_FIELDS
        )
        _live.update(
            tk=tk, qv=qv, qc=qc, qnc=qnc, qr=qr, qnr=qnr, qi=qi, qni=qni,
            qs=qs, qns=qns, qg=qg, qng=qng, qh=qh, qnh=qnh,
            nccn=nccn, ninpot=ninpot, ninact=ninact,
        )  # fmt: skip
        record("post", **_live)

        # --- post-post_twomoment driver housekeeping (mixing-ratio space),
        # mo_2mom_mcrph_driver.f90:467-524 -- NOT part of clouds_twomoment,
        # easy to miss for exactly that reason (see port_log.md, 2026-07-23
        # Phase 4 entry) ---
        for q in (qr, qi, qs, qg, qh):
            self._prog(uc.clip_negative)(q, *dom, offset_provider={})
        for n in (qnr, qni, qns, qng, qnh):
            self._prog(uc.clip_negative)(n, *dom, offset_provider={})

        zf = self._field(0.5 * (state.hhl[:-1] + state.hhl[1:]))
        out_nccn, out_ninact, out_ninpot = self._zeros(nlev), self._zeros(nlev), self._zeros(nlev)
        self._prog(post_hk.post_housekeeping_program)(
            cfg.CCN_COEFFS.ncn0, cfg.CCN_COEFFS.z0, cfg.CCN_COEFFS.z1e,
            cfg.IN_COEFFS.n0, cfg.IN_COEFFS.z0, cfg.IN_COEFFS.z1e, dt,
            zf, qc, qi, nccn, ninact, ninpot,
            out_nccn, out_ninact, out_ninpot, *dom, offset_provider={},
        )  # fmt: skip
        nccn, ninact, ninpot = out_nccn, out_ninact, out_ninpot

        self._prog(hk.zero_n_where_q_tiny)(qc, qnc, *dom, offset_provider={})

        # --- satad (after) ---
        out_tk, out_qv, out_qc = self._zeros(nlev), self._zeros(nlev), self._zeros(nlev)
        self._prog(sa.satad_program)(tk, qv, qc, rho, out_tk, out_qv, out_qc, *dom, offset_provider={})
        tk, qv, qc = out_tk, out_qv, out_qc

        return ColumnState(
            hhl=state.hhl, rho=state.rho, pres=state.pres, w=state.w,
            tk=tk.asnumpy().flatten(), qv=qv.asnumpy().flatten(), qc=qc.asnumpy().flatten(),
            qnc=qnc.asnumpy().flatten(), qr=qr.asnumpy().flatten(), qnr=qnr.asnumpy().flatten(),
            qi=qi.asnumpy().flatten(), qni=qni.asnumpy().flatten(),
            qs=qs.asnumpy().flatten(), qns=qns.asnumpy().flatten(),
            qg=qg.asnumpy().flatten(), qng=qng.asnumpy().flatten(),
            qh=qh.asnumpy().flatten(), qnh=qnh.asnumpy().flatten(),
            nccn=nccn.asnumpy().flatten(), ninpot=ninpot.asnumpy().flatten(),
            ninagi=state.ninagi, ninact=ninact.asnumpy().flatten(),
        )  # fmt: skip
