scidoc:
Outputs:
 - z_exner_ex_pr :
    $$
    \exnerprime{\ntilde}{\c}{\k} = (1 + \WtimeExner) \exnerprime{\n}{\c}{\k} - \WtimeExner \exnerprime{\n-1}{\c}{\k}, \k \in [0, \nlev) \\
    \exnerprime{\ntilde}{\c}{\nlev} = 0
    $$
    Compute the temporal extrapolation of perturbed exner function
    using the time backward scheme (see the |ICONTutorial| page 74).
    This variable has nlev+1 levels even though it is defined on full levels.
 - exner_pr :
    $$
    \exnerprime{\n-1}{\c}{\k} = \exnerprime{\ntilde}{\c}{\k}
    $$
    Store the perturbed exner function from the previous time step.

Inputs:
 - $\WtimeExner$ : exner_exfac
 - $\exnerprime{\n}{\c}{\k}$ : exner - exner_ref_mc
 - $\exnerprime{\n-1}{\c}{\k}$ : exner_pr


scidoc:
Outputs:
 - z_exner_ic :
    $$
    \exnerprime{\ntilde}{\c}{\k-1/2} = \Wlev \exnerprime{\ntilde}{\c}{\k} + (1 - \Wlev) \exnerprime{\ntilde}{\c}{\k-1}, \quad \k \in [\max(1,\nflatlev), \nlev) \\
    \exnerprime{\ntilde}{\c}{\nlev-1/2} = \sum_{\k=\nlev-1}^{\nlev-3} \Wlev_{\k} \exnerprime{\ntilde}{\c}{\k}
    $$
    Interpolate the perturbation exner from full to half levels.
    The ground level is based on quadratic extrapolation (with
    hydrostatic assumption?).
 - z_dexner_dz_c_1 :
    $$
    \exnerprimedz{\ntilde}{\c}{\k} \approx \frac{\exnerprime{\ntilde}{\c}{\k-1/2} - \exnerprime{\ntilde}{\c}{\k+1/2}}{\Dz{\k}}, \quad \k \in [\max(1,\nflatlev), \nlev]
    $$
    Use the interpolated values to compute the vertical derivative
    of perturbation exner at full levels.

Inputs:
 - $\Wlev$ : wgtfac_c
 - $\Wlev_{\k}$ : wgtfacq_c
 - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
 - $\exnerprime{\ntilde}{\c}{\k\pm1/2}$ : z_exner_ic
 - $1 / \Dz{\k}$ : inv_ddqz_z_full

scidoc:
Outputs:
 - z_dexner_dz_c_2 :
    $$
    \exnerprimedzz{\ntilde}{\c}{\k} = - \frac{1}{2} \left( (\vpotempprime{\n}{\c}{\k-1/2} - \vpotempprime{\n}{\c}{\k+1/2}) \dexrefdz{\c}{\k} + \vpotempprime{\n}{\c}{\k} \ddexrefdzz{\c}{\k} \right), \quad \k \in [\nflatgradp, \nlev) \\
    \ddz{\exnerref{}{}} = - \frac{g}{\cpd \vpotempref{}{}}
    $$
    Compute the second vertical derivative of the perturbed exner function.
    This uses the hydrostatic approximation (see eqs. 13 and 7,8 in
    |ICONSteepSlopePressurePaper|).
    Note that the reference state of temperature (eq. 15 in
    |ICONSteepSlopePressurePaper|) is used when computing
    $\ddz{\vpotempref{\c}{\k}}$ in $\ddexrefdzz{\c}{\k}$.

Inputs:
 - $\vpotempprime{\n}{\c}{\k\pm1/2}$ : z_theta_v_pr_ic
 - $\vpotempprime{\n}{\c}{\k}$ : z_rth_pr_2
 - $\dexrefdz{}{}$ : d2dexdz2_fac1_mc
 - $\ddexrefdzz{}{}$ : d2dexdz2_fac2_mc


scidoc:
Outputs:
 - z_gradh_exner :
    $$
    \exnerprimegradh{\ntilde}{\e}{\k} = \Cgrad \Gradn_{\offProv{e2c}} \exnerprime{\ntilde}{\c}{\k}, \quad \k \in [0, \nflatlev)
    $$
    Compute the horizontal gradient (at constant height) of the
    temporal extrapolation of perturbed exner function on flat levels,
    unaffected by the terrain following deformation.

Inputs:
 - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
 - $\Cgrad$ : inverse_dual_edge_lengths

scidoc:
Outputs:
 - z_gradh_exner :
    $$
    \exnerprimegradh{\ntilde}{\e}{\k} &&= \left.\pdxn{\exnerprime{}{}{}}\right|_{s} - \left.\pdxn{h}\right|_{s}\exnerprimedz{}{}{}\\
                                      &&= \Wedge \Gradn_{\offProv{e2c}} \exnerprime{\ntilde}{\c}{\k}
                                        - \pdxn{h} \sum_{\offProv{e2c}} \Whor \exnerprimedz{\ntilde}{\c}{\k},
                                          \quad \k \in [\nflatlev, \nflatgradp]
    $$
    Compute $\exnerprimegradh{}{}{}$ on non-flat levels, affected
    by the terrain following deformation, i.e. those levels for
    which $\pdxn{h} \neq 0$ (eq. 14 in |ICONdycorePaper| or eq. 5
    in |ICONSteepSlopePressurePaper|).

Inputs:
 - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
 - $\Wedge$ : inverse_dual_edge_lengths
 - $\exnerprimedz{\ntilde}{\c}{\k}$ : z_dexner_dz_c_1
 - $\Whor$ : c_lin_e


scidoc:
Outputs:
 - z_gradh_exner :
    $$
    \exnerprimegradh{\ntilde}{\e}{\k} &&= \Wedge (\exnerprime{*}{\c_1}{} - \exnerprime{*}{\c_0}{}) \\
                                      &&= \Wedge \Gradn_{\offProv{e2c}} \left[ \exnerprime{\ntilde}{\c}{\k^*} + \dzgradp \left( \exnerprimedz{\ntilde}{\c}{\k^*} + \dzgradp \exnerprimedzz{\ntilde}{\c}{\k^*} \right) \right],
                                          \quad \k \in [\nflatgradp+1, \nlev)
    $$
    Compute $\exnerprimegradh{}{}{}$ when the height of
    neighboring cells is in another level.
    The usual centered difference approximation is used for the
    gradient (eq. 6 in |ICONSteepSlopePressurePaper|), but instead
    of cell center values, the exner function is reconstructed
    using a second order Taylor-series expansion (eq. 8 in
    |ICONSteepSlopePressurePaper|).
    $k^*$ is the level index of the neighboring (horizontally, not
    terrain-following) cell center and $h^*$ is its height.

Inputs:
 - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
 - $\exnerprimedz{\ntilde}{\c}{\k}$ : z_dexner_dz_c_1
 - $\exnerprimedzz{\ntilde}{\c}{\k}$ : z_dexner_dz_c_2
 - $\Wedge$ : inverse_dual_edge_lengths
 - $\dzgradp$ : zdiff_gradp
 - $\k^*$ : vertoffset_gradp

scidoc:
Outputs:
 - z_hydro_corr :
    $$
    \exnhydrocorr{\e} = \frac{g}{\cpd} \Wedge 4 \frac{ \vpotemp{}{\c_1}{\k} - \vpotemp{}{\c_0}{\k} }{ (\vpotemp{}{\c_1}{\k} + \vpotemp{}{\c_0}{\k})^2 },
    $$
    with
    $$
    \vpotemp{}{\c_i}{\k} = \vpotemp{}{\c_i}{\k^*} + \dzgradp \frac{\vpotemp{}{\c_i}{\k^*-1/2} - \vpotemp{}{\c_i}{\k^*+1/2}}{\Dz{\k^*}}
    $$
    Compute the hydrostatically approximated correction term that
    replaces the downward extrapolation (last term in eq. 10 in
    |ICONSteepSlopePressurePaper|).
    This is only computed for the bottom-most level because all
    edges which have a neighboring cell center inside terrain
    beyond a certain limit use the same correction term at $k^*$
    level in eq. 10 in |ICONSteepSlopePressurePaper| (see also the
    last paragraph on page 3724 for the discussion).
    $\c_i$ are the indexes of the adjacent cell centers using
    $\offProv{e2c}$;
    $k^*$ is the level index of the neighboring (horizontally, not
    terrain-following) cell center and $h^*$ is its height.

Inputs:
 - $\vpotemp{}{\c}{\k}$ : theta_v
 - $\vpotemp{}{\c}{\k\pm1/2}$ : theta_v_ic
 - $\frac{g}{\cpd}$ : grav_o_cpd
 - $\Wedge$ : inverse_dual_edge_lengths
 - $1 / \Dz{\k}$ : inv_ddqz_z_full
 - $\dzgradp$ : zdiff_gradp
 - $\k^*$ : vertoffset_gradp

scidoc:
Outputs:
 - z_gradh_exner :
    $$
    \exnerprimegradh{\ntilde}{\e}{\k} = \exnerprimegradh{\ntilde}{\e}{\k} + \exnhydrocorr{\e} (h_k - h_{k^*}), \quad \e \in \IDXpg
    $$
    Apply the hydrostatic correction term to the horizontal
    gradient (at constant height) of the temporal extrapolation of
    perturbed exner function (eq. 10 in
    |ICONSteepSlopePressurePaper|).
    This is only applied to edges for which the adjacent cell
    center (horizontally, not terrain-following) would be
    underground, i.e. edges in the $\IDXpg$ set.

Inputs:
 - $\exnerprimegradh{\ntilde}{\e}{\k}$ : z_gradh_exner
 - $\exnhydrocorr{\e}$ : hydro_corr_horizontal
 - $(h_k - h_{k^*})$ : pg_exdist

scidoc:
Outputs:
 - vn :
    $$
    \vn{\n+1^*}{\e}{\k} = \vn{\n}{\e}{\k} - \Dt \left( \advvn{\n}{\e}{\k} + \cpd \vpotemp{\n}{\e}{\k} \exnerprimegradh{\ntilde}{\e}{\k} \right)
    $$
    Update the normal wind speed with the advection and pressure
    gradient terms.

Inputs:
 - $\vn{\n}{\e}{\k}$ : vn
 - $\Dt$ : dtime
 - $\advvn{\n}{\e}{\k}$ : ddt_vn_apc_pc[self.ntl1]
 - $\vpotemp{\n}{\e}{\k}$ : z_theta_v_e
 - $\exnerprimegradh{\ntilde}{\e}{\k}$ : z_gradh_exner
 - $\cpd$ : CPD
