scidoc:
Outputs:
 - zeta :
    $$
    \vortvert{\n}{\v}{\k} = \sum_{\offProv{v2e}} \Crot \vn{\n}{\e}{\k}
    $$
    Compute the vorticity on vertices using the discrete Stokes
    theorem (eq. 5 in |BonaventuraRingler2005|).

Inputs:
 - $\Crot$ : geofac_rot
 - $\vn{\n}{\e}{\k}$ : vn


scidoc:
Outputs:
 - vt :
    $$
    \vt{\n}{\e}{\k} = \sum_{\offProv{e2c2e}} \Wrbf \vn{\n}{\e}{\k}
    $$
    Compute the tangential velocity by RBF interpolation from four neighboring
    edges (diamond shape) projected along the tangential direction.

Inputs:
 - $\Wrbf$ : rbf_vec_coeff_e
 - $\vn{\n}{\e}{\k}$ : vn


scidoc:
Outputs:
 - vn_ie :
    $$
    \vn{\n}{\e}{\k-1/2} = \Wlev \vn{\n}{\e}{\k} + (1 - \Wlev) \vn{\n}{\e}{\k-1}, \quad \k \in [1, \nlev)
    $$
    Interpolate the normal velocity from full to half levels.
 - z_kin_hor_e :
    $$
    \kinehori{\n}{\e}{\k} = \frac{1}{2} \left( \vn{\n}{\e}{\k}^2 + \vt{\n}{\e}{\k}^2 \right), \quad \k \in [1, \nlev)
    $$
    Compute the horizontal kinetic energy. Exclude the first full level.

Inputs:
 - $\Wlev$ : wgtfac_e
 - $\vn{\n}{\e}{\k}$ : vn
 - $\vt{\n}{\e}{\k}$ : vt


scidoc:
Outputs:
 - z_w_concorr_me :
    $$
    \wcc{\n}{\e}{\k} = \vn{\n}{\e}{\k} \pdxn{z} + \vt{\n}{\e}{\k} \pdxt{z}, \quad \k \in [\nflatlev, \nlev)
    $$
    Compute the contravariant correction to the vertical wind due to
    terrain-following coordinate. $\pdxn{}$ and $\pdxt{}$ are the
    horizontal derivatives along the normal and tangent directions
    respectively (eq. 17 in |ICONdycorePaper|).
 - vn_ie :
    $$
    \vn{\n}{\e}{-1/2} = \vn{\n}{\e}{0}
    $$
    Set the normal wind at model top equal to the normal wind at the
    first full level.
 - z_vt_ie :
    $$
    \vt{\n}{\e}{-1/2} = \vt{\n}{\e}{0}
    $$
    Set the tangential wind at model top equal to the tangential wind
    at the first full level.
 - z_kin_hor_e :
    $$
    \kinehori{\n}{\e}{0} = \frac{1}{2} \left( \vn{\n}{\e}{0}^2 + \vt{\n}{\e}{0}^2 \right)
    $$
    Compute the horizontal kinetic energy on the first full level.

Inputs:
 - $\vn{\n}{\e}{\k}$ : vn
 - $\vt{\n}{\e}{\k}$ : vt
 - $\pdxn{z}$ : ddxn_z_full
 - $\pdxt{z}$ : ddxt_z_full


scidoc:
Outputs:
 - z_ekinh :
    $$
    \kinehori{\n}{\c}{\k} = \sum_{\offProv{c2e}} \Whor \kinehori{\n}{\e}{\k}
    $$
    Interpolate the horizonal kinetic energy from edge to cell center.

Inputs:
 - $\Whor$ : e_bln_c_s
 - $\kinehori{\n}{\e}{\k}$ : z_kin_hor_e


scidoc:
Outputs:
 - z_w_concorr_mc :
    $$
    \wcc{\n}{\c}{\k} = \sum_{\offProv{c2e}} \Whor \wcc{\n}{\e}{\k}
    $$
    Interpolate the contravariant correction from edge to cell center.
 - w_concorr_c :
    $$
    \wcc{\n}{\c}{\k-1/2} = \Wlev \wcc{\n}{\c}{\k} + (1 - \Wlev) \wcc{\n}{\c}{\k-1}, \quad \k \in [\nflatlev+1, \nlev)
    $$
    Interpolate the contravariant correction from full to half levels.

Inputs:
 - $\wcc{\n}{\e}{\k}$ : z_w_concorr_me
 - $\Whor$ : e_bln_c_s
 - $\Wlev$ : wgtfac_c


scidoc:
Outputs:
 - z_w_con_c :
    $$
    (\w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2}) =
    \begin{cases}
        \w{\n}{\c}{\k-1/2},                        & \k \in [0, \nflatlev+1)     \\
        \w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2}, & \k \in [\nflatlev+1, \nlev) \\
        0,                                         & \k = \nlev
    \end{cases}
    $$
    Subtract the contravariant correction $\wcc{}{}{}$ from the
    vertical wind $\w{}{}{}$ in the terrain-following levels. This is
    done for convevnience here, instead of directly in the advection
    tendency update, because the result needs to be interpolated to
    edge centers and full levels for later use.
    The papers do not use a new symbol for this variable, and the code
    ambiguosly mixes the variable names used for
    $\wcc{}{}{}$ and $(\w{}{}{} - \wcc{}{}{})$.

Inputs:
 - $\w{\n}{\c}{\k\pm1/2}$ : w
 - $\wcc{\n}{\c}{\k\pm1/2}$ : w_concorr_c

scidoc:
Outputs:
 - z_w_con_c_full :
    $$
    (\w{\n}{\c}{\k} - \wcc{\n}{\c}{\k}) = \frac{1}{2} [ (\w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2})
                                                      + (\w{\n}{\c}{\k+1/2} - \wcc{\n}{\c}{\k+1/2}) ]
    $$
    Interpolate the vertical wind with contravariant correction from
    half to full levels.

Inputs:
 - $(\w{\n}{\c}{\k\pm1/2} - \wcc{\n}{\c}{\k\pm1/2})$ : z_w_con_c


scidoc:
Outputs:
 - ddt_vn_apc_pc[ntnd] :
    $$
    \advvn{\n}{\e}{\k} &&= \pdxn{\kinehori{}{}{}} + \vt{}{}{} (\vortvert{}{}{} + \coriolis{}) + \pdz{\vn{}{}{}} (\w{}{}{} - \wcc{}{}{}) \\
                       &&= \Gradn_{\offProv{e2c}} \Cgrad \kinehori{\n}{c}{\k} + \kinehori{\n}{\e}{\k} \Gradn_{\offProv{e2c}} \Cgrad     \\
                       &&+ \vt{\n}{\e}{\k} (\coriolis{\e} + 1/2 \sum_{\offProv{e2v}} \vortvert{\n}{\v}{\k})                             \\
                       &&+ \frac{\vn{\n}{\e}{\k-1/2} - \vn{\n}{\e}{\k+1/2}}{\Dz{k}}
                           \sum_{\offProv{e2c}} \Whor (\w{\n}{\c}{\k} - \wcc{\n}{\c}{\k})
    $$
    Compute the advective tendency of the normal wind (eq. 13 in
    |ICONdycorePaper|).
    The edge-normal derivative of the kinetic energy is computed by
    combining the first order approximation across adiacent cell
    centres (eq. 7 in |BonaventuraRingler2005|) with the edge value of
    the kinetic energy (TODO: this needs explaining and a reference).

Inputs:
 - $\Cgrad$ : coeff_gradekin
 - $\kinehori{\n}{\e}{\k}$ : z_kin_hor_e
 - $\kinehori{\n}{\c}{\k}$ : z_ekinh
 - $\vt{\n}{\e}{\k}$ : vt
 - $\coriolis{\e}$ : f_e
 - $\vortvert{\n}{\v}{\k}$ : zeta
 - $\Whor$ : c_lin_e
 - $(\w{\n}{\c}{\k} - \wcc{\n}{\c}{\k})$ : z_w_con_c_full
 - $\vn{\n}{\e}{\k\pm1/2}$ : vn_ie
 - $\Dz{\k}$ : ddqz_z_full_e
