using Serialization

using ImplicitGlobalGrid
import MPI

const CLI_BENCHMARK_MODE = "--benchmark" in ARGS
const USE_GPU = lowercase(get(ENV, "USE_GPU", "true")) in ("1", "true", "yes")

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
import ParallelStencil: @reset_parallel_stencil

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
    if !CLI_BENCHMARK_MODE
        @info "threads" Threads.nthreads()
    end
end

using Printf

const h_eps = 1e-2

"""
    avx_comp(hv1, hv2, h, ix, iy)

    Compute the x-face average of hv1*hv2/h between cells (ix,iy) and (ix+1,iy).

    # Arguments
    - hv1, hv2: Momentum-like arrays.
    - h: Water-depth array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar face-averaged value in x.
"""
@inline avx_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix+1, iy] * hv2[ix+1, iy] / h[ix+1, iy])

"""
    avy_comp(hv1, hv2, h, ix, iy)

    Compute the y-face average of hv1*hv2/h between cells (ix,iy) and (ix,iy+1).

    # Arguments
    - hv1, hv2: Momentum-like arrays.
    - h: Water-depth array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar face-averaged value in y.
"""
@inline avy_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix, iy+1] * hv2[ix, iy+1] / h[ix, iy+1])

"""
        avx_simp(h, ix, iy)

    Compute the simple x-face average of h^2 between cells (ix,iy) and (ix+1,iy).

    # Arguments
    - h: Water-depth array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar average of squared depth in x.
"""
@inline avx_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix+1, iy] * h[ix+1, iy])

"""
    avy_simp(h, ix, iy)

    Compute the simple y-face average of h^2 between cells (ix,iy) and (ix,iy+1).

    # Arguments
    - h: Water-depth array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar average of squared depth in y.
"""
@inline avy_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix, iy+1] * h[ix, iy+1])

"""
    dxa(h, ix, iy)

    Forward difference in x at cell (ix,iy).

    # Arguments
    - h: Input array.
    - ix, iy: Cell indices.

    # Returns
    - h[ix+1,iy] - h[ix,iy].
"""
@inline dxa(h, ix, iy) = h[ix+1, iy] - h[ix, iy]

"""
    dya(h, ix, iy)

    Forward difference in y at cell (ix,iy).

    # Arguments
    - h: Input array.
    - ix, iy: Cell indices.

    # Returns
    - h[ix,iy+1] - h[ix,iy].
"""
@inline dya(h, ix, iy) = h[ix, iy+1] - h[ix, iy]

"""
    dxb(h, ix, iy)

    Backward difference in x at cell (ix,iy).

    # Arguments
    - h: Input array.
    - ix, iy: Cell indices.

    # Returns
    - h[ix,iy] - h[ix-1,iy].
"""
@inline dxb(h, ix, iy) = h[ix, iy] - h[ix-1, iy]

"""
    dyb(h, ix, iy)

    Backward difference in y at cell (ix,iy).

    # Arguments
    - h: Input array.
    - ix, iy: Cell indices.

    # Returns
    - h[ix,iy] - h[ix,iy-1].
"""
@inline dyb(h, ix, iy) = h[ix, iy] - h[ix, iy-1]

"""
    eta(h, z, ix, iy)

    Compute the free-surface elevation eta = h + z at (ix,iy).

    # Arguments
    - h: Water-depth array.
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar free-surface elevation.
"""
@inline eta(h, z, ix, iy) = h[ix, iy] + z[ix, iy]

"""
    zx_face(z, ix, iy)

    Compute the x-face average of bathymetry between (ix,iy) and (ix+1,iy).

    # Arguments
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar face-averaged bathymetry in x.
"""
@inline zx_face(z, ix, iy) = 0.5 * (z[ix, iy] + z[ix+1, iy])

"""
    zy_face(z, ix, iy)

    Compute the y-face average of bathymetry between (ix,iy) and (ix,iy+1).

    # Arguments
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Scalar face-averaged bathymetry in y.
"""
@inline zy_face(z, ix, iy) = 0.5 * (z[ix, iy] + z[ix, iy+1])

"""
    hx_L(h, z, ix, iy)

    Reconstructed left depth at the x-face between (ix,iy) and (ix+1,iy).

    # Arguments
    - h: Water-depth array.
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Non-negative reconstructed depth on the left side.
"""
@inline hx_L(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy) - zx_face(z, ix, iy))

"""
    hx_R(h, z, ix, iy)

    Reconstructed right depth at the x-face between (ix,iy) and (ix+1,iy).

    # Arguments
    - h: Water-depth array.
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Non-negative reconstructed depth on the right side.
"""
@inline hx_R(h, z, ix, iy) =
    max(0.0, eta(h, z, ix+1, iy) - zx_face(z, ix, iy))

"""
    hy_L(h, z, ix, iy)

    Reconstructed left depth at the y-face between (ix,iy) and (ix,iy+1).

    # Arguments
    - h: Water-depth array.
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Non-negative reconstructed depth on the left side.
"""
@inline hy_L(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy) - zy_face(z, ix, iy))

"""
    hy_R(h, z, ix, iy)

    Reconstructed right depth at the y-face between (ix,iy) and (ix,iy+1).

    # Arguments
    - h: Water-depth array.
    - z: Bathymetry array.
    - ix, iy: Cell indices.

    # Returns
    - Non-negative reconstructed depth on the right side.
"""
@inline hy_R(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy+1) - zy_face(z, ix, iy))

"""
    desing_velocity(hval, qval, vel_eps)

    Compute a depth-limited velocity from depth and momentum.

    # Arguments
    - hval: Local depth value.
    - qval: Local momentum value.
    - vel_eps: Regularization parameter to avoid division by zero.

    # Returns
    - Scalar velocity, zero in dry cells.
"""
@inline function desing_velocity(hval, qval, vel_eps)
    if hval <= 0.0
        return 0.0
    end

    return sqrt(2.0) * hval * qval /
           sqrt(hval^4 + max(hval^4, vel_eps))
end

"""
    vel_u(h, hu, ix, iy, vel_eps)

    Compute the x-velocity from depth and x-momentum at (ix,iy).

    # Arguments
    - h: Water-depth array.
    - hu: x-momentum array.
    - ix, iy: Cell indices.
    - vel_eps: Regularization parameter.

    # Returns
    - Scalar x-velocity.
"""
@inline vel_u(h, hu, ix, iy, vel_eps) =
    desing_velocity(h[ix, iy], hu[ix, iy], vel_eps)

"""
    vel_v(h, hv, ix, iy, vel_eps)

    Compute the y-velocity from depth and y-momentum at (ix,iy).

    # Arguments
    - h: Water-depth array.
    - hv: y-momentum array.
    - ix, iy: Cell indices.
    - vel_eps: Regularization parameter.

    # Returns
    - Scalar y-velocity.
"""
@inline vel_v(h, hv, ix, iy, vel_eps) =
    desing_velocity(h[ix, iy], hv[ix, iy], vel_eps)

"""
    bc_speed_x(h, hu, ix, iy, g)

    Compute characteristic boundary speed in x for the radiative BCs.

    # Arguments
    - h: Water-depth array.
    - hu: x-momentum array.
    - ix, iy: Cell indices.
    - g: Gravity constant.

    # Returns
    - Scalar wave speed estimate in x.
"""
@inline bc_speed_x(h, hu, ix, iy, g) =
    h[ix, iy] > h_eps ?
        abs(hu[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]) :
        0.0

"""
    bc_speed_y(h, hv, ix, iy, g)

    Compute characteristic boundary speed in y for the radiative BCs.

    # Arguments
    - h: Water-depth array.
    - hv: y-momentum array.
    - ix, iy: Cell indices.
    - g: Gravity constant.

    # Returns
    - Scalar wave speed estimate in y.
"""
@inline bc_speed_y(h, hv, ix, iy, g) =
    h[ix, iy] > h_eps ?
        abs(hv[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]) :
        0.0

const g = 1.0

"""
    min_g(A)

    Compute the global minimum of array A across all MPI ranks.

    # Arguments
    - A: Local array on the current rank.

    # Returns
    - Scalar minimum value over the global domain.
"""
min_g(A) = (min_l = minimum(A); MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD))

"""
    max_g(A)

    Compute the global maximum of array A across all MPI ranks.

    # Arguments
    - A: Local array on the current rank.

    # Returns
    - Scalar maximum value over the global domain.
"""
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

"""
    save_domain_decomposition!(outdir; ...)

    Gather and serialize the MPI domain ownership map without using any plotting
    libraries. The output is meant to be consumed by scripts that visualize or
    inspect the decomposition after the simulation has run.
"""
function save_domain_decomposition!(outdir;
        me,
        dims,
        nprocs,
        nx,
        ny,
        nx_global,
        ny_global,
        nx_field,
        ny_field,
        lx,
        ly,
        dx,
        dy,
        ix_roi,
        iy_roi,
        filename="domain_decomposition.jls")

    rank_local = fill(Float32(me), nx - 2, ny - 2)
    rank_global = zeros(Float32, (nx - 2) * dims[1], (ny - 2) * dims[2])
    # DIFF baseline/manual: baseline uses IGG's gather!, manual.jl implements
    # the same rank-0 assembly explicitly with MPI.Gatherv!.
    gather!(rank_local, rank_global)

    mkpath(outdir)

    if me == 0
        payload = (
            format = "hpc4wc_domain_decomposition_v1",
            rank = rank_global,
            dims = Tuple(dims[1:2]),
            nprocs = nprocs,
            local_size_with_halo = (nx, ny),
            global_size_with_halo = (nx_global, ny_global),
            interior_size_global = size(rank_global),
            field_size = (nx_field, ny_field),
            roi_indices = (collect(ix_roi), collect(iy_roi)),
            domain = (lx = lx, ly = ly),
            spacing = (dx = dx, dy = dy),
        )
        fname = joinpath(outdir, filename)
        serialize(fname, payload)
        @info "Saved domain decomposition to $fname"
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

"""
    compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)

    Compute local maximum wave speeds on all x- and y-faces.

    # Arguments
    - max_speed_x, max_speed_y: Output arrays for x- and y-face speeds.
    - h, hu, hv: Water depth and momentum fields.
    - z: Bathymetry field.
    - g: Gravity constant.
    - vel_eps: Velocity regularization parameter.

    # Returns
    - Nothing. Writes to max_speed_x and max_speed_y.
"""
@parallel_indices (ix, iy) function compute_maxspeed!(
    max_speed_x, max_speed_y,
    h, hu, hv, z, g, vel_eps
)
    nx, ny = size(h)

    if ix <= nx - 1 && iy <= ny
        hL = hx_L(h, z, ix, iy)
        hR = hx_R(h, z, ix, iy)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix+1, iy, vel_eps)

        max_speed_x[ix, iy] = max(
            abs(uL) + sqrt(g * hL),
            abs(uR) + sqrt(g * hR)
        )
    end

    if ix <= nx && iy <= ny - 1
        hL = hy_L(h, z, ix, iy)
        hR = hy_R(h, z, ix, iy)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix, iy+1, vel_eps)

        max_speed_y[ix, iy] = max(
            abs(vL) + sqrt(g * hL),
            abs(vR) + sqrt(g * hR)
        )
    end

    return nothing
end

"""
    compute_draining_timestep!(dt_drain, F₁, G₁, h, dt, _dx, _dy)

    Compute the draining timestep constraint per cell based on outgoing fluxes.

    # Arguments
    - dt_drain: Output array of local drain timesteps.
    - F₁, G₁: Mass fluxes in x and y.
    - h: Water-depth array.
    - dt: Current global timestep.
    - _dx, _dy: Inverse grid spacing.

    # Returns
    - Nothing. Writes to dt_drain.
"""
@parallel_indices (ix, iy) function compute_draining_timestep!(
    dt_drain,
    F₁, G₁,
    h,
    dt,
    _dx, _dy
)
    nx, ny = size(h)

    if 2 <= ix <= nx-1 && 2 <= iy <= ny-1
        out_x =
            max(F₁[ix, iy], 0.0) +
            max(-F₁[ix-1, iy], 0.0)

        out_y =
            max(G₁[ix, iy], 0.0) +
            max(-G₁[ix, iy-1], 0.0)

        drain_rate = out_x * _dx + out_y * _dy

        if drain_rate > 0.0
            dt_drain[ix, iy] = min(dt, h[ix, iy] / drain_rate)
        else
            dt_drain[ix, iy] = dt
        end
    end

    return nothing
end

"""
    compute_effective_flux_timesteps!(dtFx, dtGy, dt_drain, F₁, G₁, dt)

    Compute face-wise effective timesteps using upwinded draining limits.

    # Arguments
    - dtFx, dtGy: Output arrays for x- and y-face timesteps.
    - dt_drain: Cell-centered draining timesteps.
    - F₁, G₁: Mass fluxes in x and y.
    - dt: Global timestep.

    # Returns
    - Nothing. Writes to dtFx and dtGy.
"""
@parallel_indices (ix, iy) function compute_effective_flux_timesteps!(
    dtFx, dtGy,
    dt_drain,
    F₁, G₁,
    dt
)
    nxm1, ny = size(F₁)
    nx, nym1 = size(G₁)

    # x-faces
    if ix <= nxm1 && iy <= ny
        if F₁[ix, iy] > 0.0
            dtFx[ix, iy] = min(dt, dt_drain[ix, iy])
        elseif F₁[ix, iy] < 0.0
            dtFx[ix, iy] = min(dt, dt_drain[ix+1, iy])
        else
            dtFx[ix, iy] = dt
        end
    end

    # y-faces
    if ix <= nx && iy <= nym1
        if G₁[ix, iy] > 0.0
            dtGy[ix, iy] = min(dt, dt_drain[ix, iy])
        elseif G₁[ix, iy] < 0.0
            dtGy[ix, iy] = min(dt, dt_drain[ix, iy+1])
        else
            dtGy[ix, iy] = dt
        end
    end

    return nothing
end

"""
    compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, z, g,
                                 max_speed_x, max_speed_y, vel_eps)

    Compute Rusanov fluxes for mass and momentum in both x and y directions.

    # Arguments
    - F₁, F₂, F₃: Fluxes on x-faces (mass, x-momentum, y-momentum).
    - G₁, G₂, G₃: Fluxes on y-faces (mass, x-momentum, y-momentum).
    - hu, hv, h: Momentum and depth fields.
    - z: Bathymetry field.
    - g: Gravity constant.
    - max_speed_x, max_speed_y: Precomputed max wave speeds.
    - vel_eps: Velocity regularization parameter.

    # Returns
    - Nothing. Writes to F* and G* arrays.
"""
@parallel_indices (ix, iy) function compute_1st_2nd_and_3th_flux!(
    F₁, F₂, F₃,
    G₁, G₂, G₃,
    hu, hv, h, z, g,
    max_speed_x, max_speed_y,
    vel_eps
)
    nx, ny = size(h)

    # -------------------------------------------------------------------------
    # x-direction fluxes
    # -------------------------------------------------------------------------
    if ix <= nx - 1 && iy <= ny
        hL = hx_L(h, z, ix, iy)
        hR = hx_R(h, z, ix, iy)

        ηL = eta(h, z, ix, iy)
        ηR = eta(h, z, ix+1, iy)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix+1, iy, vel_eps)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix+1, iy, vel_eps)

        # Reconstruct momenta consistently:
        # momentum = reconstructed depth × cell velocity
        huL = hL * uL
        huR = hR * uR
        hvL = hL * vL
        hvR = hR * vR

        ax = max_speed_x[ix, iy]

        # Mass / free-surface flux
        F₁[ix, iy] =
            0.5 * (huL + huR) -
            0.5 * ax * (ηR - ηL)

        # x-momentum flux
        F₂[ix, iy] =
            0.5 * (
                huL * uL + 0.5 * g * hL^2 +
                huR * uR + 0.5 * g * hR^2
            ) -
            0.5 * ax * (huR - huL)

        # y-momentum transported in x
        F₃[ix, iy] =
            0.5 * (
                huL * vL +
                huR * vR
            ) -
            0.5 * ax * (hvR - hvL)
    end

    # -------------------------------------------------------------------------
    # y-direction fluxes
    # -------------------------------------------------------------------------
    if ix <= nx && iy <= ny - 1
        hL = hy_L(h, z, ix, iy)
        hR = hy_R(h, z, ix, iy)

        ηL = eta(h, z, ix, iy)
        ηR = eta(h, z, ix, iy+1)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix, iy+1, vel_eps)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix, iy+1, vel_eps)

        huL = hL * uL
        huR = hR * uR
        hvL = hL * vL
        hvR = hR * vR

        ay = max_speed_y[ix, iy]

        # Mass / free-surface flux
        G₁[ix, iy] =
            0.5 * (hvL + hvR) -
            0.5 * ay * (ηR - ηL)

        # x-momentum transported in y
        G₂[ix, iy] =
            0.5 * (
                hvL * uL +
                hvR * uR
            ) -
            0.5 * ay * (huR - huL)

        # y-momentum flux
        G₃[ix, iy] =
            0.5 * (
                hvL * vL + 0.5 * g * hL^2 +
                hvR * vR + 0.5 * g * hR^2
            ) -
            0.5 * ay * (hvR - hvL)
    end

    return nothing
end



"""
    update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy,
                            z, g, dt, _dx, _dy)

    Update water depth and momentum using flux divergence and source terms.

    # Arguments
    - h, hu, hv: State arrays updated in place.
    - F₁, G₁, F₂, F₃, G₂, G₃: Flux arrays.
    - dtFx, dtGy: Face-wise timesteps for mass fluxes.
    - z: Bathymetry field.
    - g: Gravity constant.
    - dt: Global timestep.
    - _dx, _dy: Inverse grid spacing.

    # Returns
    - Nothing. Updates h, hu, hv in place.
"""
@parallel_indices (ix, iy) function update_height_momentum!(
    h, hu, hv,
    F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy,
    z, g, dt, _dx, _dy
)
    nx, ny = size(h)

    if 2 <= ix <= nx-1 && 2 <= iy <= ny-1
        ηC = eta(h, z, ix, iy)

        # ---------------------------------------------------------------------
        # x-source term
        # ---------------------------------------------------------------------
        zE = 0.5 * (z[ix, iy] + z[ix+1, iy])
        zW = 0.5 * (z[ix-1, iy] + z[ix, iy])

        hE = max(0.0, ηC - zE)
        hW = max(0.0, ηC - zW)

        hsrc_x = 0.5 * (hE + hW)
        dzdx_face = (zE - zW) * _dx

        # ---------------------------------------------------------------------
        # y-source term
        # ---------------------------------------------------------------------
        zN = 0.5 * (z[ix, iy] + z[ix, iy+1])
        zS = 0.5 * (z[ix, iy-1] + z[ix, iy])

        hN = max(0.0, ηC - zN)
        hS = max(0.0, ηC - zS)

        hsrc_y = 0.5 * (hN + hS)
        dzdy_face = (zN - zS) * _dy

        # ---------------------------------------------------------------------
        # Momentum updates first
        # ---------------------------------------------------------------------
        hu[ix, iy] -= dt * (
            dxb(F₂, ix, iy) * _dx +
            dyb(G₂, ix, iy) * _dy +
            g * hsrc_x * dzdx_face
        )

        hv[ix, iy] -= dt * (
            dxb(F₃, ix, iy) * _dx +
            dyb(G₃, ix, iy) * _dy +
            g * hsrc_y * dzdy_face
        )

        # ---------------------------------------------------------------------
        # Water-depth update
        # Since z is stationary, h_t = η_t
        # ---------------------------------------------------------------------
        h[ix, iy] -= (
            (F₁[ix, iy] * dtFx[ix, iy] -
            F₁[ix-1, iy] * dtFx[ix-1, iy]) * _dx
            +
            (G₁[ix, iy] * dtGy[ix, iy] -
            G₁[ix, iy-1] * dtGy[ix, iy-1]) * _dy
        )

    end

    return nothing
end

"""
    left_bc!(h, hu, hv, g, dt, _dx)

    Apply the left-side radiative boundary condition.

    # Arguments
    - h, hu, hv: State arrays updated in place.
    - g: Gravity constant.
    - dt: Timestep.
    - _dx: Inverse grid spacing in x.

    # Returns
    - Nothing. Updates the left boundary column.
"""
@parallel_indices (iy) function left_bc!(h, hu, hv, g, dt, _dx)
    # Left boundary (ix=1)
    cL = bc_speed_x(h, hu, 1, iy, g) * dt * _dx
    αL = (cL - 1) / (cL + 1)

    h1  = max(0.0, h[2, iy] + αL * (h[2, iy] - h[1, iy]))
    hu1 = hu[2, iy] + αL * (hu[2, iy] - hu[1, iy])
    hv1 = hv[2, iy] + αL * (hv[2, iy] - hv[1, iy])

    if h1 <= h_eps
        hu1 = 0.0
        hv1 = 0.0
    end

    h[1, iy]    = h1
    hu[1, iy]   = hu1
    hv[1, iy]   = hv1

    return nothing
end

"""
    right_bc!(h, hu, hv, g, dt, _dx)

    Apply the right-side radiative boundary condition.

    # Arguments
    - h, hu, hv: State arrays updated in place.
    - g: Gravity constant.
    - dt: Timestep.
    - _dx: Inverse grid spacing in x.

    # Returns
    - Nothing. Updates the right boundary column.
"""
@parallel_indices (iy) function right_bc!(h, hu, hv, g, dt, _dx)
    nx, ny = size(h)

    # Right boundary (ix=nx)
    cR = bc_speed_x(h, hu, nx, iy, g) * dt * _dx
    αR = (cR - 1) / (cR + 1)

    hR  = max(0.0, h[end-1, iy]  + αR * (h[end-1, iy]  - h[end, iy]))
    huR = hu[end-1, iy] + αR * (hu[end-1, iy] - hu[end, iy])
    hvR = hv[end-1, iy] + αR * (hv[end-1, iy] - hv[end, iy])

    if hR <= h_eps
        huR = 0.0
        hvR = 0.0
    end

    h[end, iy]  = hR
    hu[end, iy] = huR
    hv[end, iy] = hvR
    return nothing
end

"""
    bottom_bc!(h, hu, hv, g, dt, _dy)

    Apply the bottom-side radiative boundary condition.

    # Arguments
    - h, hu, hv: State arrays updated in place.
    - g: Gravity constant.
    - dt: Timestep.
    - _dy: Inverse grid spacing in y.

    # Returns
    - Nothing. Updates the bottom boundary row.
"""
@parallel_indices (ix) function bottom_bc!(h, hu, hv, g, dt, _dy)
    # Bottom boundary (iy=1)
    cB = bc_speed_y(h, hv, ix, 1, g) * dt * _dy
    αB = (cB - 1) / (cB + 1)

    hB  = max(0.0, h[ix, 2]  + αB * (h[ix, 2]  - h[ix, 1]))
    huB = hu[ix, 2] + αB * (hu[ix, 2] - hu[ix, 1])
    hvB = hv[ix, 2] + αB * (hv[ix, 2] - hv[ix, 1])

    if hB <= h_eps
        huB = 0.0
        hvB = 0.0
    end

    h[ix, 1]    = hB
    hu[ix, 1]   = huB
    hv[ix, 1]   = hvB
    return nothing
end

"""
    top_bc!(h, hu, hv, g, dt, _dy)

    Apply the top-side radiative boundary condition.

    # Arguments
    - h, hu, hv: State arrays updated in place.
    - g: Gravity constant.
    - dt: Timestep.
    - _dy: Inverse grid spacing in y.

    # Returns
    - Nothing. Updates the top boundary row.
"""
@parallel_indices (ix) function top_bc!(h, hu, hv, g, dt, _dy)
    nx, ny = size(h)

    # Top boundary (iy=ny)
    cT = bc_speed_y(h, hv, ix, ny, g) * dt * _dy
    αT = (cT - 1) / (cT + 1)

    hT  = max(0.0, h[ix, end-1]  + αT * (h[ix, end-1]  - h[ix, end]))
    huT = hu[ix, end-1] + αT * (hu[ix, end-1] - hu[ix, end])
    hvT = hv[ix, end-1] + αT * (hv[ix, end-1] - hv[ix, end])

    if hT <= h_eps
        huT = 0.0
        hvT = 0.0
    end

    h[ix, end]  = hT
    hu[ix, end] = huT
    hv[ix, end] = hvT
    return nothing
end

"""
    dry_cell_fix!(h, hu, hv, h_eps)

    Clamp dry or invalid cells to zero depth and momentum.

    # Arguments
    - h, hu, hv: State arrays updated in place.
    - h_eps: Dry-cell threshold.

    # Returns
    - Nothing. Updates h, hu, hv in place.
"""
@parallel_indices (ix, iy) function dry_cell_fix!(h, hu, hv, h_eps)
    nx, ny = size(h)

    if ix <= nx && iy <= ny
        if !isfinite(h[ix, iy]) || h[ix, iy] <= h_eps
            h[ix, iy]  = 0.0
            hu[ix, iy] = 0.0
            hv[ix, iy] = 0.0
        elseif !isfinite(hu[ix, iy]) || !isfinite(hv[ix, iy])
            hu[ix, iy] = 0.0
            hv[ix, iy] = 0.0
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

"""
    run_baseline(nx_global, ny_global; nt=20, outdir="frames",
                 do_viz=false, benchmark=false)

Run the simple 2D SWE baseline with IGG domain decomposition.
"""
@views function run_baseline(nx_global, ny_global;
            nt=20,
            outdir="frames",
            do_viz=false,
            benchmark=false,
            test=false,
            test_file="",
            warmup=5,
            benchdir="docs/benchmark/baseline.csv")
    lx = 50.0
    ly = 50.0

    # Let IGG/MPI choose a compact 2D process topology instead of forcing 2x2.
    # We query the same MPI_Dims_create choice before init_global_grid so the
    # requested global grid can still be split into uniform local blocks.
    mpi_was_initialized = MPI.Initialized()
    if !mpi_was_initialized
        MPI.Init()
    end
    nprocs_world = MPI.Comm_size(MPI.COMM_WORLD)
    dims_mpi_3d = MPI.Dims_create!(nprocs_world, [0, 0, 1])
    dims_mpi = dims_mpi_3d[1:2]

    if (nx_global - 2) % dims_mpi[1] != 0 || (ny_global - 2) % dims_mpi[2] != 0
        error("Global interior size $(nx_global - 2)x$(ny_global - 2) must be divisible by IGG's chosen MPI topology $(dims_mpi[1])x$(dims_mpi[2]).")
    end

    nx = div(nx_global - 2, dims_mpi[1]) + 2
    ny = div(ny_global - 2, dims_mpi[2]) + 2

    # DIFF baseline/manual: this is the central IGG setup. It creates the
    # Cartesian communicator, computes rank coordinates, and installs IGG's
    # global-index/halo-exchange helpers. manual.jl replaces this block with
    # explicit MPI.Init, MPI.Cart_create, MPI.Cart_get, and MPI.Cart_shift calls.
    me, dims, nprocs, coords, comm_cart = init_global_grid(
        nx, ny, 1;
        init_MPI=false,
        quiet=benchmark,
        select_device=false
    )

    # DIFF baseline/manual: IGG gives coords directly. manual.jl derives the
    # same boundary flags from MPI.Cart_shift neighbors and MPI.PROC_NULL.
    is_left   = coords[1] == 0
    is_right  = coords[1] == dims[1] - 1
    is_bottom = coords[2] == 0
    is_top    = coords[2] == dims[2] - 1
    
    b_width     = (8, 8, 0)


    println("Global domain size (including halos): ", nx_global, " x ", ny_global)
    println("MPI topology: ", dims[1], " x ", dims[2])
    println("Local domain size (including halos): ", nx, " x ", ny)
    println("Time steps: ", nt)


    nvis = benchmark ? 0 : 50  # visualization frequency
    # DIFF baseline/manual: nx_g()/ny_g() and x_g()/y_g() are IGG global-grid
    # helpers. manual.jl computes these values from nx_global/ny_global and a
    # local get_global_indices(...) helper.
    dx = lx / (nx_g() - 1)
    dy = ly / (ny_g() - 1)
    vel_eps = min(dx, dy)^4
    _dx  = 1.0 / dx
    _dy  = 1.0 / dy

    h  = @zeros(nx, ny)
    hu = @zeros(nx, ny)
    hv = @zeros(nx, ny)

    ix_roi = 2:(nx_g() - 1)
    iy_roi = 2:(ny_g() - 1)

    function reset_fields!()
        h .= max.(0.0, η0 .- z)
        hu .= 0.0
        hv .= 0.0
        dt_drain .= 0.0
        dtFx .= 0.0
        dtGy .= 0.0

        @parallel dry_cell_fix!(h, hu, hv, hmin)
        update_halo!(h, hu, hv)

        if is_left;   @parallel (1:ny) left_bc!(h, hu, hv, g, 0.0, _dx);   end
        if is_right;  @parallel (1:ny) right_bc!(h, hu, hv, g, 0.0, _dx);  end
        if is_bottom; @parallel (1:nx) bottom_bc!(h, hu, hv, g, 0.0, _dy); end
        if is_top;    @parallel (1:nx) top_bc!(h, hu, hv, g, 0.0, _dy);    end
        @synchronize()
    end

    function save_test_h!(outdir, test_file="")
        fname = isempty(test_file) ? joinpath(outdir, "test_h.jls") : test_file
        mkpath(dirname(fname))
        nx_v = nx - 2
        ny_v = ny - 2
        h_inn = Array{Float32}(undef, nx_v, ny_v)
        h_inn .= Array(h)[2:end-1, 2:end-1]
        h_v = zeros(Float32, nx_g() - 2, ny_g() - 2)
        gather!(h_inn, h_v)

        if me == 0
            open(fname, "w") do io
                serialize(io, h_v)
            end
            @info "Saved test h array to $fname"
        end
    end

    if !benchmark && !test
        save_domain_decomposition!(outdir;
            me = me,
            dims = dims,
            nprocs = nprocs,
            nx = nx,
            ny = ny,
            nx_global = nx_global,
            ny_global = ny_global,
            nx_field = nx_g() - 2,
            ny_field = ny_g() - 2,
            lx = lx,
            ly = ly,
            dx = dx,
            dy = dy,
            ix_roi = ix_roi,
            iy_roi = iy_roi,
        )
    end

    # DIFF baseline/manual: these coordinates include halo cells through IGG's
    # global x_g/y_g mapping. manual.jl builds equivalent halo-inclusive index
    # ranges by hand.
    xs = [x_g(ix, dx, h) - lx / 2 for ix in 1:nx]
    ys = [y_g(iy, dy, h) - ly / 2 for iy in 1:ny]

    # fluxes
    F₁ = @zeros(nx - 1, ny)
    F₂ = @zeros(nx - 1, ny)
    F₃ = @zeros(nx - 1, ny)

    G₁ = @zeros(nx, ny - 1)
    G₂ = @zeros(nx, ny - 1)
    G₃ = @zeros(nx, ny - 1)

    max_speed_x = @zeros(nx - 1, ny)
    max_speed_y = @zeros(nx, ny - 1)

    z_local  = zeros(nx, ny)
    η0_local = zeros(nx, ny)

    h_base = 15.0
    bump_amplitude = 2.0
    bump_width = max(lx, ly) / 12

    for i in 1:nx
        for j in 1:ny
            x = xs[i]
            y = ys[j]

            z_local[i, j] = 0.0
            η0_local[i, j] = h_base + bump_amplitude * exp(-(x^2 + y^2) / (2 * bump_width^2))
        end
    end

    z  = Data.Array(z_local)
    η0 = Data.Array(η0_local)

    hmin  = 1e-2
    h .= max.(0.0, η0 .- z)

    if test
        save_test_h!(outdir, test_file)
        finalize_global_grid(finalize_MPI=!mpi_was_initialized)
        return 0.0
    end

    dt_drain = @zeros(nx, ny)

    dtFx = @zeros(nx - 1, ny)
    dtGy = @zeros(nx, ny - 1)
    
    time = 0.0

    # -------------------------------------------------------------------------
    # visualization
    # -------------------------------------------------------------------------

    if do_viz && !benchmark
        if me == 0
            println("Using visualization: Array output")
        end
        mkpath(outdir)
        
        nx_v, ny_v = (nx - 2) * dims[1], (ny - 2) * dims[2]
        h_v = zeros(nx_v, ny_v)
        z_v = zeros(nx_v, ny_v)
        h_inn = zeros(nx - 2, ny - 2)
        z_inn = zeros(nx - 2, ny - 2)

        h_inn .= Array(h)[2:end-1, 2:end-1]; gather!(h_inn, h_v)
        z_inn .= Array(z)[2:end-1, 2:end-1]; gather!(z_inn, z_v)

        if me == 0
            frame_id = Ref(0)
            @info "Saving arrays to $outdir"
            function save_array!()
                frame_id[] += 1
                fname = joinpath(outdir, @sprintf("array_frame_%06d.jls", frame_id[]))
                serialize(fname, (h=Array(convert.(Float32, h_v)),))
            end
            function save_array_with_z!()
                frame_id[] += 1
                fname = joinpath(outdir, @sprintf("array_frame_%06d.jls", frame_id[]))
                serialize(fname, (h=Array(convert.(Float32, h_v)), z=Array(convert.(Float32, z_v))))
            end
            save_array_with_z!()
        end
    end

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    dt = 0.0
    loop_walltime = 0.0

    if benchmark && warmup > 0
        reset_fields!()
        dt = 0.0
        for it in 1:warmup
            @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)
            if it == 1 || it % 10 == 0
                dt = 0.9 / (max_g(max_speed_x) * _dx + max_g(max_speed_y) * _dy)
            end
            @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, z, g, max_speed_x, max_speed_y, vel_eps)
            @hide_communication b_width begin
                @parallel compute_draining_timestep!(dt_drain, F₁, G₁, h, dt, _dx, _dy)
                update_halo!(dt_drain)
            end
            @parallel compute_effective_flux_timesteps!(dtFx, dtGy, dt_drain, F₁, G₁, dt)
            @hide_communication b_width begin
                @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy, z, g, dt, _dx, _dy)
                update_halo!(h, hu, hv)
            end
            @parallel dry_cell_fix!(h, hu, hv, hmin)
            if is_left;   @parallel (1:ny) left_bc!(h, hu, hv, g, dt, _dx);   end
            if is_right;  @parallel (1:ny) right_bc!(h, hu, hv, g, dt, _dx);  end
            if is_bottom; @parallel (1:nx) bottom_bc!(h, hu, hv, g, dt, _dy); end
            if is_top;    @parallel (1:nx) top_bc!(h, hu, hv, g, dt, _dy);    end
        end
        @synchronize()
        MPI.Barrier(comm_cart)
        reset_fields!()
        @synchronize()
        MPI.Barrier(comm_cart)
    end

    @synchronize()
    MPI.Barrier(comm_cart)
    loop_t0 = time_ns()

    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)
        
        if !benchmark && 0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy) < dt 
            println("Warning: Local dt = ", 0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy), " is bigger than the current CLF, at iteration ", it)
        end

        if it % 10 == 0 || it == 1
            dt =  0.9 / (max_g(max_speed_x) * _dx + max_g(max_speed_y) * _dy)
        end

        time += dt

        if !isfinite(dt)
            error("Non-finite dt at iteration $it: dt=$dt, max_sx=$(maximum(max_speed_x)), max_sy=$(maximum(max_speed_y))")
        end

        @parallel compute_1st_2nd_and_3th_flux!(
            F₁, F₂, F₃,
            G₁, G₂, G₃,
            hu, hv, h, z, g,
            max_speed_x, max_speed_y, vel_eps
        )

        @hide_communication b_width begin
            @parallel compute_draining_timestep!(
                dt_drain,
                F₁, G₁,
                h,
                dt,
                _dx, _dy
            )
            update_halo!(dt_drain)
        end

        
        @parallel compute_effective_flux_timesteps!(
            dtFx, dtGy,
            dt_drain,
            F₁, G₁,
            dt
        )
        
        @hide_communication b_width begin
            @parallel update_height_momentum!(
                h, hu, hv,
                F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy,
                z, g, dt, _dx, _dy
            )           
            update_halo!(h, hu, hv)
        end

        @parallel dry_cell_fix!(h, hu, hv, hmin)

        # Apply BCs only on ranks that border the global domain boundaries
        if is_left
            @parallel (1:ny) left_bc!(h, hu, hv, g, dt, _dx)
        end

        if is_right
            @parallel (1:ny) right_bc!(h, hu, hv, g, dt, _dx)
        end

        if is_bottom
            @parallel (1:nx) bottom_bc!(h, hu, hv, g, dt, _dy)
        end

        if is_top
            @parallel (1:nx) top_bc!(h, hu, hv, g, dt, _dy)
        end
        
        @parallel dry_cell_fix!(h, hu, hv, hmin)

        if do_viz && !benchmark && it % nvis == 0

            # DIFF baseline/manual: field output gathering uses IGG gather! here.
            # manual.jl uses gather_global_array_manual!(...) with MPI.Gatherv!.
            h_inn .= Array(h)[2:end-1, 2:end-1]; gather!(h_inn, h_v)
            z_inn .= Array(z)[2:end-1, 2:end-1]; gather!(z_inn, z_v)
            
            if me == 0
                save_array!()
            end
        end
        
        if me == 0 && !benchmark
            percent = 100 * it / nt
            print("\rProgress: $(round(percent, digits=1)) %")
            flush(stdout)
        end
    end

    @synchronize()
    loop_walltime = MPI.Allreduce((time_ns() - loop_t0) * 1e-9, MPI.MAX, comm_cart)

    if benchmark
        if me == 0
            cells_per_step = nx_g() * ny_g()
            cell_updates_per_second = cells_per_step * nt / loop_walltime
            println(
                "BENCHMARK ",
                "walltime_seconds=", @sprintf("%.9f", loop_walltime), " ",
                "nt=", nt, " ",
                "global_size=", nx_g(), "x", ny_g(), " ",
                "local_size=", nx, "x", ny, " ",
                "nprocs=", nprocs, " ",
                "steps_per_second=", @sprintf("%.6f", nt / loop_walltime), " ",
                "cell_updates_per_second=", @sprintf("%.6e", cell_updates_per_second)
            )

            mkpath(dirname(benchdir))
            file_exists = isfile(benchdir)
            open(benchdir, "a") do io
                if !file_exists
                    println(io, "solver,nprocs,topology,nx_global,ny_global,nx_local,ny_local,nt,run,walltime,steps_per_second,cell_updates_per_second")
                end
                topology_str = "$(dims[1])x$(dims[2])"
                @printf(io, "%s,%d,%s,%d,%d,%d,%d,%d,%d,%.9f,%.6f,%.6e\n",
                    "baseline", nprocs, topology_str,
                    nx_g(), ny_g(), nx, ny, nt, 1,
                    loop_walltime, nt / loop_walltime, cell_updates_per_second
                )
            end
        end
        # DIFF baseline/manual: IGG owns grid teardown here. manual.jl finalizes
        # MPI only if it initialized MPI itself.
        finalize_global_grid(finalize_MPI=!mpi_was_initialized)
        return loop_walltime
    end

    if me == 0
        println("\nSimulation completed.")
        println("Total simulation time: $(round(time, digits=2)) seconds")

        if do_viz
            println("\nSaved $(frame_id[]) frames to: $(abspath(outdir))")
        end
    end

    # DIFF baseline/manual: IGG owns grid teardown here. manual.jl finalizes MPI
    # only if it initialized MPI itself.
    finalize_global_grid(finalize_MPI=!mpi_was_initialized)
    return loop_walltime

end

input_nx = 802
input_ny = 802
input_nt = 2000
input_outdir = "docs/frames/baseline"
input_do_viz = false
input_benchmark = false
input_test = false
input_test_file = ""
input_benchdir = "docs/benchmark/baseline.csv"
input_warmup = 5

for i in 1:length(ARGS)
    if ARGS[i] == "--nx"
        global input_nx = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--ny"
        global input_ny = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--nt"
        global input_nt = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--outdir"
        global input_outdir = ARGS[i+1]
    elseif ARGS[i] == "--viz"
        global input_do_viz = true
    elseif ARGS[i] == "--benchmark"
        global input_benchmark = true
    elseif ARGS[i] == "--test"
        global input_test = true
    elseif ARGS[i] == "--test-file"
        global input_test_file = ARGS[i+1]
    elseif ARGS[i] == "--benchdir"
        global input_benchdir = ARGS[i+1]
    elseif ARGS[i] == "--warmup"
        global input_warmup = parse(Int, ARGS[i+1])
    end
end

run_baseline(input_nx, input_ny; nt=input_nt,
    outdir = input_outdir,
    do_viz = input_do_viz && !input_benchmark,
    benchmark = input_benchmark,
    test = input_test,
    test_file = input_test_file,
    warmup = input_warmup,
    benchdir = input_benchdir
)
