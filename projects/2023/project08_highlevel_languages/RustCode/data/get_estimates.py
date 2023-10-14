import json
import pprint as pp
import pandas as pd


def get_all_means():
    df = pd.DataFrame(columns=["naive", "vectorized", "iterators", "rayiterators", "view_iterators", "view_rayiterators"])
    # empty data frame, easiest to make it larger than necessary to at least avoid any errors
    # from not being able to save values into df
    for i in range(4):
        df.loc[i] = [0, 0, 0, 0, 0 , 0]
    pp.pprint(df)

    # iterate through the different versions of the rust benchmarks, see RustCode/benches/bench.rs
    # versions = ["naive", "vectorized", "iterators", "rayiterators"]
    versions = ["naive", "vectorized", "iterators", "rayiterators", "view_iterators", "view_rayiterators"]
    for version in versions:
        # iterate through problem sizes. Note the files have to exist, i.e. file not found errors will happen
        # if any of the benchmarks were not run and files dont exist.
        for i in [0, 1, 2, 3]: # <--- loop over the problem sizes
            # new/ is the latest benchmark
            path = "./target/criterion/stencil_computations/" + version + "/" + str(i) + "/base/"
            f = open(path + "estimates.json")
            data = json.load(f)
            # values are in seconds *10^-6
            estimate = data["median"]["confidence_interval"]["lower_bound"] * 1e-6
            # print(round(estimate,4), "ms")
            # df[version][i] = estimate
            df.loc[i, version] = round(estimate, 6)
            f.close()

    pp.pprint(df)
    df.to_csv("../universal_output/rust_benchmarks.csv", sep=",", index=False)


get_all_means()
