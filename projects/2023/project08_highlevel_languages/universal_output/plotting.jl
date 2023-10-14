using Gadfly
using DataFrames
using CSV
import Cairo, Fontconfig

Dimensions = ["32x32x32","64x64x64","128x128x128","256x256x256"]

julia = DataFrame(CSV.File("./universal_output/jl_benchmarks.csv"))
julia_names = DataFrame(CSV.File("./universal_output/jl_benchmarks.csv",header=false,limit=1))

python = DataFrame(CSV.File("./universal_output/py_benchmarks.csv"))
python_mpi = DataFrame(CSV.File("./universal_output/py_mpi_benchmarks.csv"))
python_names = DataFrame(CSV.File("./universal_output/py_benchmarks.csv",header=false,limit=1))

rust = DataFrame(CSV.File("./universal_output/rust_benchmarks.csv"))
rust_names = DataFrame(CSV.File("./universal_output/rust_benchmarks.csv",header=false,limit=1))


pjulia = plot(x=Dimensions,y=log10.(julia[:,1]),Geom.line,Guide.xlabel("Dimensions"), Guide.ylabel("Runtime in log₁₀ (t) ms"),Geom.point,
                color=julia_names[:,1],linestyle=[:dash],
            Theme(point_size=8pt, major_label_font="CMU Serif", major_label_font_size=26pt,
                minor_label_font="CMU Serif", minor_label_font_size=24pt, key_title_font_size=26pt, key_label_font_size=26pt),
            layer(x=Dimensions,y=log10.(julia[:,2]), Geom.line, Geom.point, 
                color=julia_names[:,2]),
            layer(x=Dimensions,y=log10.(julia[:,3]), Geom.line, Geom.point, 
                color=julia_names[:,3]),
            layer(x=Dimensions,y=log10.(julia[:,4]), Geom.line, Geom.point, 
                color=julia_names[:,4]),
            layer(x=Dimensions,y=log10.(julia[:,5]), Geom.line, Geom.point, 
                color=julia_names[:,5]), 
            )
draw(PNG("julia.png", 15inch, 10inch), pjulia)

ppython = plot(x=Dimensions,y=log10.(python[:,1]),Geom.line,Guide.xlabel("Dimensions"), Guide.ylabel("Runtime in log₁₀ (t) ms"),Geom.point,
                color=python_names[:,1],linestyle=[:dash],
                Theme(point_size=8pt, major_label_font="CMU Serif", major_label_font_size=26pt,
                    minor_label_font="CMU Serif", minor_label_font_size=24pt, key_title_font_size=26pt, key_label_font_size=26pt),
            layer(x=Dimensions,y=log10.(python[:,2]), Geom.line, Geom.point, 
                color=python_names[:,2]),
            layer(x=Dimensions,y=log10.(python[:,3]), Geom.line, Geom.point, 
                color=python_names[:,3]),
            layer(x=Dimensions,y=log10.(python_mpi[:,1]), Geom.line, Geom.point, 
                color=["mpi"]),
            )
draw(PNG("python.png", 15inch, 10inch), ppython)

prust = plot(x=Dimensions,y=log10.(rust[:,1]),Geom.line,Guide.xlabel("Dimensions"), Guide.ylabel("Runtime in log₁₀ (t) ms"),Geom.point,
                color=rust_names[:,1],linestyle=[:dash],
                Theme(point_size=8pt, major_label_font="CMU Serif", major_label_font_size=26pt,
                    minor_label_font="CMU Serif", minor_label_font_size=24pt, key_title_font_size=26pt, key_label_font_size=26pt),
            layer(x=Dimensions,y=log10.(rust[:,2]), Geom.line, Geom.point, 
                color=rust_names[:,2]),
            layer(x=Dimensions,y=log10.(rust[:,3]), Geom.line, Geom.point, 
                color=rust_names[:,3]),
            layer(x=Dimensions,y=log10.(rust[:,4]), Geom.line, Geom.point, 
                color=rust_names[:,4]),
            )
draw(PNG("rust.png", 15inch, 10inch), prust)

comparison = plot(x=Dimensions,y=log10.(julia[:,4]),Geom.line,Guide.xlabel("Dimensions"), Guide.ylabel("Runtime in log₁₀ (t) ms"),Geom.point,
                color=["Julia (cuda)"],linestyle=[:dash],
                Theme(point_size=8pt, major_label_font="CMU Serif", major_label_font_size=26pt,
                    minor_label_font="CMU Serif", minor_label_font_size=24pt, key_title_font_size=26pt, key_label_font_size=26pt),
            layer(x=Dimensions,y=log10.(python[:,3]), Geom.line, Geom.point, 
                color=["Python (cupy)"]),
            layer(x=Dimensions,y=log10.(rust[:,1]), Geom.line, Geom.point, 
                color=["Rust (naive)"]),
)
