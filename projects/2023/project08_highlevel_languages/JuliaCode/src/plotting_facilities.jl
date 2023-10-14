module PlottingFacilities
using Gadfly

export plot_array_at_z

function plot_array_at_z(arr, z)
    Gadfly.spy(arr[:, :, z])
end
end #module