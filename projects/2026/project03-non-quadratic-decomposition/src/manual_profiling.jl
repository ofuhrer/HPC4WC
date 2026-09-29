using Serialization
using NVTX

import MPI


const CLI_BENCHMARK_MODE = "--benchmark" in ARGS
const USE_GPU = lowercase(get(ENV, "USE_GPU", "false")) in ("1", "true", "yes")
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
import ParallelStencil: @reset_parallel_stencil

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
    CUDA.allowscalar(false)
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


@inline function make_send_buffer(A_slice)
    buffer = copy(A_slice)

    @static if USE_GPU
        CUDA.synchronize()
    end

    return buffer
end

@inline function backend_synchronize()
    @static if USE_GPU
        CUDA.synchronize()
    end

    return nothing
end

const g = 1.0

"""
    min_g(A)

    Compute the global minimum of array A across all MPI ranks.

    # Arguments
    - A: Local array on the current rank.

    # Returns
    - Scalar minimum value over the global domain.
"""
min_g(A, comm=MPI.COMM_WORLD) = (min_l = minimum(A); MPI.Allreduce(min_l, MPI.MIN, comm))

"""
    max_g(A)

    Compute the global maximum of array A across all MPI ranks.

    # Arguments
    - A: Local array on the current rank.

    # Returns
    - Scalar maximum value over the global domain.
"""
max_g(A, comm=MPI.COMM_WORLD) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, comm))

function gather_global_array_manual!(local_interior, global_array, comm_cart)
    # DIFF manual/baseline: this replaces IGG's gather!. Each rank sends its
    # local interior as a flat buffer; rank 0 unpacks chunks according to the
    # Cartesian rank coordinates.
    me = MPI.Comm_rank(comm_cart)
    nprocs = MPI.Comm_size(comm_cart)
    dims, _, _ = MPI.Cart_get(comm_cart)

    local_nx, local_ny = size(local_interior)
    sendbuf = vec(local_interior)
    counts = fill(length(sendbuf), nprocs)

    if me == 0
        expected_size = (local_nx * dims[1], local_ny * dims[2])
        @assert size(global_array) == expected_size

        recvbuf_data = similar(sendbuf, sum(counts))
        recvbuf = MPI.VBuffer(recvbuf_data, counts)
        MPI.Gatherv!(sendbuf, recvbuf, comm_cart; root=0)

        offset = 1
        for rank in 0:(nprocs - 1)
            coords = MPI.Cart_coords(comm_cart, rank)
            ix0 = coords[1] * local_nx + 1
            iy0 = coords[2] * local_ny + 1
            chunk = @view recvbuf_data[offset:(offset + counts[rank + 1] - 1)]
            global_array[ix0:(ix0 + local_nx - 1), iy0:(iy0 + local_ny - 1)] .=
                reshape(chunk, local_nx, local_ny)
            offset += counts[rank + 1]
        end
    else
        MPI.Gatherv!(sendbuf, nothing, comm_cart; root=0)
    end

    return global_array
end

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
        comm_cart,
        ix_roi,
        iy_roi,
        filename="domain_decomposition.jls")

    rank_local = fill(Float32(me), nx - 2, ny - 2)
    rank_global = zeros(Float32, (nx - 2) * dims[1], (ny - 2) * dims[2])
    # DIFF manual/baseline: manual gather path; baseline.jl calls IGG gather!.
    gather_global_array_manual!(rank_local, rank_global, comm_cart)

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
                 do_viz=false, benchmark=false, benchdir = "frames")

Run the simple 2D SWE baseline with IGG domain decomposition.
"""

@views function get_global_indices(nx, ny, coords)
    # DIFF manual/baseline: manual replacement for IGG's x_g/y_g global-index
    # mapping. These ranges include halo cells, so neighboring ranks overlap by
    # one cell exactly where halos live.
    nx_local = nx - 2
    ny_local = ny - 2

    # Halo-inclusive global indices. Neighboring ranks overlap by one cell.
    ix_start = coords[1] * nx_local + 1
    iy_start = coords[2] * ny_local + 1

    ix_end = ix_start + nx - 1
    iy_end = iy_start + ny - 1

    return (ix_start:ix_end, iy_start:iy_end)
end

@views function x_g(ix, dx)
    return (ix - 1) * dx
end

@views function y_g(iy, dy)
    return (iy - 1) * dy
end

"""
Allocate reusable send and receive buffers for one 2D field.

Left/right boundaries contain ny values.
Bottom/top boundaries contain nx values.

`similar(A, ...)` creates CPU vectors when A is an Array and GPU vectors
when A is a CuArray.
"""
function allocate_halo_buffers(A)
    nx, ny = size(A)

    allocate_vector(n) = similar(A, eltype(A), (n,))

    return (
        send_left   = allocate_vector(ny),
        recv_left   = allocate_vector(ny),
        send_right  = allocate_vector(ny),
        recv_right  = allocate_vector(ny),

        send_bottom = allocate_vector(nx),
        recv_bottom = allocate_vector(nx),
        send_top    = allocate_vector(nx),
        recv_top    = allocate_vector(nx),
    )
end

function update_halo_scalar!(
    A,
    buffers,
    comm_cart,
    neighbors_x,
    neighbors_y,
    tag_base,
)
    left, right = neighbors_x
    bottom, top = neighbors_y

    # -------------------------------------------------------------------------
    # X-direction: pack reusable left/right buffers
    # -------------------------------------------------------------------------

    @views begin
        buffers.send_left  .= A[2, :]
        buffers.send_right .= A[end-1, :]
    end

    # Ensure GPU packing has completed before MPI reads the send buffers.
    backend_synchronize()

    # Send left interior boundary and receive right halo.
    MPI.Sendrecv!(
        buffers.send_left,
        left,
        tag_base,
        buffers.recv_right,
        right,
        tag_base,
        comm_cart,
    )

    # Send right interior boundary and receive left halo.
    MPI.Sendrecv!(
        buffers.send_right,
        right,
        tag_base + 1,
        buffers.recv_left,
        left,
        tag_base + 1,
        comm_cart,
    )

    # Unpack received x-direction halos.
    @views begin
        if right != MPI.PROC_NULL
            A[end, :] .= buffers.recv_right
        end

        if left != MPI.PROC_NULL
            A[1, :] .= buffers.recv_left
        end
    end

    # Complete unpacking before using the updated x halos or reusing buffers.
    backend_synchronize()

    # -------------------------------------------------------------------------
    # Y-direction: pack reusable bottom/top buffers
    # -------------------------------------------------------------------------

    @views begin
        buffers.send_bottom .= A[:, 2]
        buffers.send_top    .= A[:, end-1]
    end

    backend_synchronize()

    # Send bottom interior boundary and receive top halo.
    MPI.Sendrecv!(
        buffers.send_bottom,
        bottom,
        tag_base + 2,
        buffers.recv_top,
        top,
        tag_base + 2,
        comm_cart,
    )

    # Send top interior boundary and receive bottom halo.
    MPI.Sendrecv!(
        buffers.send_top,
        top,
        tag_base + 3,
        buffers.recv_bottom,
        bottom,
        tag_base + 3,
        comm_cart,
    )

    # Unpack received y-direction halos.
    @views begin
        if top != MPI.PROC_NULL
            A[:, end] .= buffers.recv_top
        end

        if bottom != MPI.PROC_NULL
            A[:, 1] .= buffers.recv_bottom
        end
    end

    backend_synchronize()

    return nothing
end

function update_halo!(
    h,
    hu,
    hv,
    buffers,
    comm_cart,
    neighbors_x,
    neighbors_y,
)
    update_halo_scalar!(
        h,
        buffers,
        comm_cart,
        neighbors_x,
        neighbors_y,
        0,
    )

    update_halo_scalar!(
        hu,
        buffers,
        comm_cart,
        neighbors_x,
        neighbors_y,
        10,
    )

    update_halo_scalar!(
        hv,
        buffers,
        comm_cart,
        neighbors_x,
        neighbors_y,
        20,
    )

    return nothing
end

function update_halo_dt_drain!(
    dt_drain,
    buffers,
    comm_cart,
    neighbors_x,
    neighbors_y,
)
    update_halo_scalar!(
        dt_drain,
        buffers,
        comm_cart,
        neighbors_x,
        neighbors_y,
        30,
    )

    return nothing
end

@views function run_baseline(nx_global, ny_global;
            nt=100,
            runs=10,                      # number of runs 
            outdir="frames",
            do_viz=false,
            benchmark=false,
            test=false,
            test_file="",
            topology=(2, 2),
            warmup=50,
            benchdir = "docs/benchmarks/sim_benchmarks.csv")
            
    lx = 50.0
    ly = 50.0

    dims_mpi = collect(topology)
    if length(dims_mpi) != 2 || any(dims_mpi .< 1)
        error("MPI topology must contain two positive dimensions, got $topology.")
    end

    if (nx_global - 2) % dims_mpi[1] != 0 || (ny_global - 2) % dims_mpi[2] != 0
        error("nx_global-2 and ny_global-2 must be divisible by the MPI topology.")
    end

    # Choose local domain size including halos.
    nx = div(nx_global - 2, dims_mpi[1]) + 2
    ny = div(ny_global - 2, dims_mpi[2]) + 2

    # DIFF manual/baseline: manual.jl owns MPI setup directly. baseline.jl lets
    # IGG's init_global_grid(...) create and manage the Cartesian grid.
    owns_mpi = !MPI.Initialized()
    if owns_mpi
        MPI.Init()
    end
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if nprocs != prod(dims_mpi)
        error("manual.jl currently expects exactly $(prod(dims_mpi)) MPI ranks for a $(dims_mpi[1])x$(dims_mpi[2]) topology, got $nprocs.")
    end

    # DIFF manual/baseline: explicit Cartesian communicator creation. This is
    # hidden inside init_global_grid(...) in baseline.jl.
    comm_cart = MPI.Cart_create(MPI.COMM_WORLD, dims_mpi; periodic=(false, false), reorder=false)
    me = MPI.Comm_rank(comm_cart)
    dims, periods, coords = MPI.Cart_get(comm_cart)

    # DIFF manual/baseline: explicit neighbor lookup. baseline.jl only needs
    # coords from IGG and calls IGG's update_halo! later.
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1) 
    neighbors_y = MPI.Cart_shift(comm_cart, 1, 1)

    # DIFF manual/baseline: physical-boundary ranks are detected from missing
    # Cartesian neighbors. baseline.jl checks coords against dims.
    is_left   = neighbors_x[1] == MPI.PROC_NULL
    is_right  = neighbors_x[2] == MPI.PROC_NULL
    is_bottom = neighbors_y[1] == MPI.PROC_NULL
    is_top    = neighbors_y[2] == MPI.PROC_NULL

    if me == 0 && !benchmark
        println("Global domain size (including halos): ", nx_global, " x ", ny_global)
        println("MPI topology: ", dims_mpi[1], " x ", dims_mpi[2])
        println("Local domain size (including halos): ", nx, " x ", ny)
        println("Time steps: ", nt)
    end

    # DIFF manual/baseline: no IGG nx_g()/ny_g() helpers here; use the explicit
    # input global sizes.
    dx = lx / (nx_global - 1)
    dy = ly / (ny_global - 1)
    vel_eps = min(dx, dy)^4
    _dx  = 1.0 / dx
    _dy  = 1.0 / dy

    ix_roi = 2:(nx_global - 1)
    iy_roi = 2:(ny_global - 1)

    if !benchmark && !test
        save_domain_decomposition!(outdir;
            me = me,
            dims = dims,
            nprocs = nprocs,
            nx = nx,
            ny = ny,
            nx_global = nx_global,
            ny_global = ny_global,
            nx_field = nx_global - 2,
            ny_field = ny_global - 2,
            lx = lx,
            ly = ly,
            dx = dx,
            dy = dy,
            comm_cart = comm_cart,
            ix_roi = ix_roi,
            iy_roi = iy_roi,
            filename = "domain_decomposition_$(dims_mpi[1])x$(dims_mpi[2]).jls"
        )
        return nothing
    end
    
    # DIFF manual/baseline: build halo-inclusive coordinate arrays manually.
    # baseline.jl uses IGG's x_g/y_g mapping directly.
    ix_g, iy_g = get_global_indices(nx, ny, coords)
    xs = [x_g(ix, dx) - lx / 2 for ix in ix_g]
    ys = [y_g(iy, dy) - ly / 2 for iy in iy_g]

    # Allokation der Arrays (einmalig für alle Runs!)
    h  = @zeros(nx, ny)
    hu = @zeros(nx, ny)
    hv = @zeros(nx, ny)

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

    function save_test_h!(outdir, test_file="")
        fname = isempty(test_file) ? joinpath(outdir, "test_h.jls") : test_file
        mkpath(dirname(fname))
        h_inn = Array{Float32}(undef, nx - 2, ny - 2)
        h_inn .= Array(h)[2:end-1, 2:end-1]
        h_v = zeros(Float32, nx_global - 2, ny_global - 2)
        gather_global_array_manual!(h_inn, h_v, comm_cart)

        if me == 0
            open(fname, "w") do io
                serialize(io, h_v)
            end
            @info "Saved test h array to $fname"
        end
    end

    if test
        save_test_h!(outdir, test_file)
        if owns_mpi
            MPI.Finalize()
        end
        return nothing
    end

    dt_drain = @zeros(nx, ny)
    halo_buffers = allocate_halo_buffers(h)
    dtFx = @zeros(nx - 1, ny)
    dtGy = @zeros(nx, ny - 1)

    # helper function to reset fields before each run
    function reset_fields!()
        h .= max.(0.0, η0 .- z)
        hu .= 0.0
        hv .= 0.0
        F₁ .= 0.0; F₂ .= 0.0; F₃ .= 0.0
        G₁ .= 0.0; G₂ .= 0.0; G₃ .= 0.0
        max_speed_x .= 0.0; max_speed_y .= 0.0
        dt_drain .= 0.0; dtFx .= 0.0; dtGy .= 0.0
        
        # 2. Halos und Ränder SOFORT synchronisieren (Wichtig!)
        update_halo!(h, hu, hv, halo_buffers, comm_cart, neighbors_x, neighbors_y)
        
        if is_left;   @parallel (1:ny) left_bc!(h, hu, hv, g, 0.0, _dx);   end
        if is_right;  @parallel (1:ny) right_bc!(h, hu, hv, g, 0.0, _dx);  end
        if is_bottom; @parallel (1:nx) bottom_bc!(h, hu, hv, g, 0.0, _dy); end
        if is_top;    @parallel (1:nx) top_bc!(h, hu, hv, g, 0.0, _dy);    end
        
        @parallel dry_cell_fix!(h, hu, hv, hmin)
        
        # 3. GPU/MPI Ausführung garantieren
        @synchronize()
    end

    # WARMUP-PHASE (before doing benchmarks)
    warmup_steps = benchmark ? warmup : 0
    if benchmark && warmup_steps > 0
        reset_fields!()
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)
        dt_warm = 0.9 / (max_g(max_speed_x, comm_cart) * _dx + max_g(max_speed_y, comm_cart) * _dy)

        for it in 1:warmup_steps
            @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)
            @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, z, g, max_speed_x, max_speed_y, vel_eps)
            @parallel compute_draining_timestep!(dt_drain, F₁, G₁, h, dt_warm, _dx, _dy)
            update_halo_dt_drain!(dt_drain, halo_buffers, comm_cart, neighbors_x, neighbors_y)
            @parallel compute_effective_flux_timesteps!(dtFx, dtGy, dt_drain, F₁, G₁, dt_warm)
            @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy, z, g, dt_warm, _dx, _dy)
            update_halo!(h, hu, hv, halo_buffers, comm_cart, neighbors_x, neighbors_y)
            
            if is_left;   @parallel (1:ny) left_bc!(h, hu, hv, g, dt_warm, _dx);   end
            if is_right;  @parallel (1:ny) right_bc!(h, hu, hv, g, dt_warm, _dx);  end
            if is_bottom; @parallel (1:nx) bottom_bc!(h, hu, hv, g, dt_warm, _dy); end
            if is_top;    @parallel (1:nx) top_bc!(h, hu, hv, g, dt_warm, _dy);    end

            @parallel dry_cell_fix!(h, hu, hv, hmin)
        end

        @synchronize()
        MPI.Barrier(comm_cart)
        
        reset_fields!()
        @synchronize()

        MPI.Barrier(comm_cart)
    end

    # vector for walltimes
    num_repeats = benchmark ? runs : 1
    walltimes = Float64[]

    NVTX.@range "profile" begin
        for r in 1:num_repeats
            reset_fields!()
            time = 0.0
            dt = 0.0

            @synchronize()
            MPI.Barrier(comm_cart)
            loop_t0 = time_ns()

            for it in 1:nt
                NVTX.@range "timestep" begin
                    @parallel compute_maxspeed!(
                        max_speed_x, max_speed_y,
                        h, hu, hv, z, g, vel_eps
                    )

                    if it % 10 == 0 || it == 1
                        NVTX.@range "global CFL reduction" begin
                            dt = 0.9 / (
                                max_g(max_speed_x, comm_cart) * _dx +
                                max_g(max_speed_y, comm_cart) * _dy
                            )
                        end
                    end

                    time += dt

                    @parallel compute_1st_2nd_and_3th_flux!(
                        F₁, F₂, F₃, G₁, G₂, G₃,
                        hu, hv, h, z, g,
                        max_speed_x, max_speed_y, vel_eps
                    )

                    @parallel compute_draining_timestep!(
                        dt_drain, F₁, G₁, h, dt, _dx, _dy
                    )

                    NVTX.@range "halo dt_drain" begin
                        update_halo_dt_drain!(
                            dt_drain,
                            halo_buffers,
                            comm_cart,
                            neighbors_x,
                            neighbors_y
                        )
                    end

                    @parallel compute_effective_flux_timesteps!(
                        dtFx, dtGy, dt_drain, F₁, G₁, dt
                    )

                    @parallel update_height_momentum!(
                        h, hu, hv,
                        F₁, G₁, F₂, F₃, G₂, G₃,
                        dtFx, dtGy,
                        z, g, dt, _dx, _dy
                    )

                    NVTX.@range "halo state" begin
                        update_halo!(
                            h, hu, hv,
                            halo_buffers,
                            comm_cart,
                            neighbors_x,
                            neighbors_y
                        )
                    end

                    @parallel dry_cell_fix!(h, hu, hv, hmin)

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
                end
            end

            @synchronize()
            MPI.Barrier(comm_cart)

            run_walltime =
                MPI.Allreduce((time_ns() - loop_t0) * 1e-9,
                            MPI.MAX, comm_cart)

            push!(walltimes, run_walltime)
        end
    end

    # write to csv file..
    if benchmark && me == 0
        mkpath(dirname(benchdir))
        file_exists = isfile(benchdir)

        open(benchdir, "a") do io
            if !file_exists
                println(io, "solver,nprocs,topology,nx_global,ny_global,nx_local,ny_local,nt,run,walltime,steps_per_second,cell_updates_per_second")
            end

            cells_per_step = nx_global * ny_global
            solver_type = "manual"
            topology_str = "$(dims_mpi[1])x$(dims_mpi[2])"

            for (r, wtime) in enumerate(walltimes)
                cup_sec = cells_per_step * nt / wtime
                @printf(io, "%s,%d,%s,%d,%d,%d,%d,%d,%d,%.9f,%.6f,%.6e\n",
                    solver_type, nprocs, topology_str,
                    nx_global, ny_global, nx, ny, nt, r,
                    wtime, nt / wtime, cup_sec
                )
            end
        end
        println("All $num_repeats benchmark runs saved to: $benchdir")
    end

    if owns_mpi
        MPI.Finalize()
    end

    return walltimes
end

function main()
    input_nx = 32770
    input_ny = 8194
    input_nt = 100
    input_runs = 1    # Default: 10 Durchläufe
    input_outdir = "docs/frames/manual"
    input_benchdir = "docs/benchmark/walltime_4_ranks.csv"
    input_do_viz = false
    input_benchmark = true
    input_test = false
    input_test_file = ""
    input_warmup = 5
    top_x = 4
    top_y = 4

    for i in 1:length(ARGS)
        if ARGS[i] == "--topo" && i < length(ARGS)
            raw_str = ARGS[i+1]
            clean_str = replace(raw_str, "(" => "", ")" => "", " " => "")
            parts = split(clean_str, ",")
            top_x = parse(Int, parts[1])
            top_y = parse(Int, parts[2])
        elseif ARGS[i] == "--nx"
            input_nx = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--ny"
            input_ny = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--nt"
            input_nt = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--runs"
            input_runs = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--outdir"
            input_outdir = ARGS[i+1]
        elseif ARGS[i] == "--benchmark"
            input_benchmark = true
        elseif ARGS[i] == "--test"
            input_test = true
        elseif ARGS[i] == "--test-file"
            input_test_file = ARGS[i+1]
        elseif ARGS[i] == "--benchdir"
            input_benchdir = ARGS[i+1]
        elseif ARGS[i] == "--warmup"
            input_warmup = parse(Int, ARGS[i+1]); i += 1

        end
    end

    run_baseline(input_nx, input_ny; 
        nt=input_nt,
        runs=input_runs,
        outdir = input_outdir,
        do_viz = input_do_viz && !input_benchmark && !input_test,
        benchmark = input_benchmark,
        test = input_test,
        test_file = input_test_file,
        topology = (top_x, top_y),
        warmup = input_warmup,
        benchdir = input_benchdir
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
