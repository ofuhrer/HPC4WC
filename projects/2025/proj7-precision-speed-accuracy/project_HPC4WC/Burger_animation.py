import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys


def read_snapshots(filename, nx, ny, nz, n_halo, num_snapshots, dtype=np.float64()):
    """
    Function to read snapshots of velocity fields from binary files.
    """
        
    snapshots = []
    nx_tot = nx + 2 * n_halo
    ny_tot = ny + 2 * n_halo
    header_size = 24 #6 * 4  # 6 int32 fields × 4 bytes each (to skip metadata)
    footer_size = 24 # trailing bytes
    field_size = nx_tot * ny_tot * nz * np.dtype(dtype).itemsize
    k = 0 # Vertical level to plot

    with open(filename, "rb") as f:
        for i in range(num_snapshots):

            header_bytes = f.read(header_size)

            if len(header_bytes) < header_size:
                break

            field_bytes = f.read(field_size)
            if len(field_bytes) < field_size:
                break

            data = np.frombuffer(field_bytes, dtype=dtype).reshape((nz, ny_tot, nx_tot))

            inner = data[0, n_halo:n_halo+ny, n_halo:n_halo+nx]

            snapshots.append(inner)
    return snapshots

def update(frame, interval_t):
    """
    Function to update the plots for each frame in the animation
    """
    im_u.set_array(u_snapshots[frame])
    im_v.set_array(v_snapshots[frame])
    im_mag.set_array(np.sqrt(u_snapshots[frame]**2 + v_snapshots[frame]**2))
    
    # Update titles
    ax_u.set_title(f"u field (t={frame*interval_t:.3f}s)")
    ax_v.set_title(f"v field (t={frame*interval_t:.3f}s)")
    ax_mag.set_title(f"Velocity Magnitude (t={frame*interval_t:.3f}s)")
    
    return [im_u, im_v, im_mag]

if __name__ == "__main__":
    """
    Signature to call from bash:
    python burger_animation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --prec $prec --nums $nums
    """

    # Parameters (adjust according to how simulation was run)
    nx, ny, nz, n_halo = int(sys.argv[2]), int(sys.argv[4]), int(sys.argv[6]), 3
    T_end = int(sys.argv[8])  # End time of the simulation (see 2d_Burger.cpp for consistency)
    precision = str(sys.argv[10])
    num_snapshots = int(sys.argv[12])
    interval_t = T_end / (num_snapshots - 1)

    if precision == 'longdouble':
        dtype = np.longdouble()
    elif precision == 'double':
        dtype = np.float64()
    elif precision == 'single':
        dtype = np.float32()
    else:
        dtype = np.float16()

    dx = 1 / (nx - 1)

    # Construct filenames dynamically based on grid dimensions and precision
    u_filename = f"output_data/u_out_field_nx_{nx}_ny_{ny}_nz_{nz}_{precision}.dat"
    v_filename = f"output_data/v_out_field_nx_{nx}_ny_{ny}_nz_{nz}_{precision}.dat"

    # Read snapshots from the dynamically generated filenames
    u_snapshots = read_snapshots(u_filename, nx, ny, nz, n_halo, num_snapshots, dtype=dtype)
    v_snapshots = read_snapshots(v_filename, nx, ny, nz, n_halo, num_snapshots, dtype=dtype)

    # Set up figure with 3 subplots
    fig, (ax_u, ax_v, ax_mag) = plt.subplots(1, 3, figsize=(15, 5), rasterized=True)
    fig.suptitle(f"2D Burgers' Equation: nx={nx}, ny={ny}, nz={nz}, Halo={n_halo}, Precision={precision}", fontsize=14)

    # Plot initial frames
    im_u = ax_u.imshow(u_snapshots[0], cmap='seismic', origin='lower', extent=[0, 1, 0, 1], vmin=-1, vmax=1)
    im_v = ax_v.imshow(v_snapshots[0], cmap='seismic', origin='lower', extent=[0, 1, 0, 1], vmin=-0.4, vmax=0.4)
    im_mag = ax_mag.imshow(np.sqrt(u_snapshots[0]**2 + v_snapshots[0]**2), cmap='viridis', origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=1.1)

    # Add colorbars
    fig.colorbar(im_u, ax=ax_u, label='u velocity', shrink=0.6)
    fig.colorbar(im_v, ax=ax_v, label='v velocity', shrink=0.6)
    fig.colorbar(im_mag, ax=ax_mag, label='Magnitude', shrink=0.6)

    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=0.4)

    # Define folder and filename for saving the animation
    os.makedirs("animations", exist_ok=True)
    gif_filename = f"animations/burgers_animation_nx_{nx}_ny_{ny}_nz_{nz}_precision_{precision}.gif"

    # Animate and save the animation as GIF
    anim = FuncAnimation(fig, update, frames=len(u_snapshots), interval=2, blit=False, fargs=(interval_t,))
    anim.save(gif_filename, writer='pillow', fps=15, dpi=150)

    plt.tight_layout()
    # plt.show()