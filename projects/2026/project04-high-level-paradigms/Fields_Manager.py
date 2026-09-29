import numpy as np
import matplotlib.pyplot as plt

class Fields_Manager:
    def __init__(self,
        in_field_path:str="in_field.npy",
        out_field_path:str="out_field.npy",
        compare_field_path:str="compare_field.npy"):

        self.in_field_path = in_field_path
        self.out_field_path = out_field_path
        self.compare_field_path = compare_field_path

    def build_field(self, field_type:str, nx:int, ny:int, nz:int, num_halo:int):
        '''
        Wrapper that calls required field type
        '''
        if field_type == "broad_center_spike":
            self.build_broad_center_spike(nx, ny, nz, num_halo)
        elif field_type == "single_center_spike":
            self.build_single_center_spike(nx, ny, nz, num_halo)
        elif field_type == "random":
            self.build_random_field(nx, ny, nz, num_halo)
        elif field_type == "test_field":
            self.build_test_field(nx, ny, nz, num_halo)
        else:
            print(f"! Warning: Unknown test field style: {field_type}. Using broad center spike instead.")
            self.build_broad_center_spike(nx, ny, nz, num_halo)


    def build_broad_center_spike(self, nx, ny, nz, num_halo):
        field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
        field[
            nz // 4 : 3 * nz // 4,
            num_halo + ny // 4 : num_halo + 3 * ny // 4,
            num_halo + nx // 4 : num_halo + 3 * nx // 4,
        ] = 1.0
        np.save(self.in_field_path, field)

    def build_single_center_spike(self, nx, ny, nz, num_halo):
        field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
        field[:, num_halo + ny // 2, num_halo + nx // 2] = 1.0
        np.save(self.in_field_path, field)

    def build_random_field(self, nx, ny, nz, num_halo):
        field = np.random.rand(nz, ny + 2 * num_halo, nx + 2 * num_halo)
        np.save(self.in_field_path, field)

    def build_test_field(self, nx, ny, nz, num_halo):
        field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))

        # 1. broad spike
        field[
            nz // 4 : 3 * nz // 4,
            num_halo + ny // 8 : num_halo + 3 * ny // 8,
            num_halo + nx // 8 : num_halo + 3 * nx // 8,
        ] = 1.0

        # 2. small spikes
        spike_size = np.ceil(min(nx, ny) // 30)
        x1, y1 = int(0.6*nx) + num_halo, int(0.1*ny) + num_halo
        x2, y2 = int(0.9*nx) + num_halo, int(0.2*ny) + num_halo
        x3, y3 = int(0.75*nx) + num_halo, int(0.4*ny) + num_halo

        field[:, y1:y1+spike_size, x1:x1+spike_size] = 1.0
        field[:, y2:y2+spike_size, x2:x2+spike_size] = 1.0
        field[:, y3:y3+spike_size, x3:x3+spike_size] = 1.0

        # 3.empty

        # 4. random
        field[:, num_halo + ny//2 : ny + 2*num_halo, num_halo + nx//2 : nx + 2*num_halo] = np.random.rand(nz, ny//2 + num_halo, nx//2 + num_halo)

        np.save(self.in_field_path, field)


    def plot_field_comparison(self, plot_filepath):
        fields = {}

        try:
            fields["in_field"] = np.load(self.in_field_path)
        except FileNotFoundError:
            print("No in field found for plotting")

        try:
            fields["out_field"] = np.load(self.out_field_path)
        except FileNotFoundError:
            print("No out field found for plotting")

        try:
            fields["compare_field"] = np.load(self.compare_field_path)
        except FileNotFoundError:
            print("No compare field found for plotting")
        
        if len(fields) <= 0:
            print("Nothing to plot.")
            return

        n_plots = len(fields)

        fig, axs = plt.subplots(1, n_plots)

        for (field_name, field), ax in zip(fields.items(), axs):
            ax.imshow(field[field.shape[0] // 2, :, :])
            ax.set_title(field_name)
        
        plt.tight_layout()
        plt.savefig(plot_filepath, dpi=300)


