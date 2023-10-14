# Performance Comparison: Rust and Julia stencil computation

## About
This file should give an overview of the completed and upcoming tasks.

If you decide to start working on a task, please annotate your name to it in parantheses, e.g. (-> Bob)

Once the task is completed, mark it as done by setting an 'x' in the check box.

Feel free to add more tasks.

Tasks in the code should be marked with:
```julia
    # TODO: Some task description
```
or
```rust
    // TODO: Some task description
```
s.t. TODOs can be globally found through VSCode search.

If there is something that needs to be discussed by the group, add it to the [Notes and General Questions](#notes-and-general-questions) section.

## Usage
All benchmarks (all stencil versions in Python, Julia, Rust for all problem sizes respectively) can be run from the home directory with `./bench_everything.sh`. Note this will take some time.

### Julia
How to run the Julia related files:
1. Be in the project directory (hpcwc-performance-comparison)
2. Start Julia with `julia --project=JuliaCode`
3. Import package via `using JuliaCode`
4. Call `JuliaCode.run()`

### Python
0. Create a new virtual environment `python venv ./PythonCode` once
1. Load the environment `source ./PythonCode/bin/activate`
2. Install the dependencies once `pip install -r ./PythonCode/requirements.txt`
3. If you need to add a dependency, install it then use `pip freeze > PythonCode/requirements.txt` to adapt the changes for everyone (repeat step 2 if somebody else did this)
4. To run a version an produce an output run `python run.py` (uncomment the import corresponding to the version you want to run)
5. To run the benchmarks, run `python benches.py`

### Rust
0. Check if Cargo is available e.g. with `cargo --version`. If not follow steps for installation for example [here](https://doc.rust-lang.org/book/ch01-01-installation.html). 
1. Navigate to `./RustCode/`
2. To run with debug profile and without optimizations type `cargo run --` followed by the desired input values
3. To run with optimizations type `cargo run --release --` followed by the desired input values
4. To run the unit tests, type `cargo test`
5. To run the benchmarks run `cargo bench`

## Starting off
- [x] Creating git repo (-> Seva)
- [x] Finalizing folder structure (-> Seva)
- [x] Set up Overleaf (-> Michelle)
- [x] Copy Python reference implementation (-> Seva)
- [x] Create a virtual environment for Python (-> Seva)
- [x] Adapt the Python code s.t. it becomes comparable to other implementations (-> Paul, Michelle)


## Implementation
### Benchmarks
The goal is to have fair comparisons between the given languages. We need to use similar benchmarking techniques for all of them:
- [x] Python (time)
- [x] Julia (benchmarktools)
- [x] Rust (Criterion)
- [ ] Discussion

### Python
Adapt code s.t. fair comparisons are possible:
- [x] Base case stencil
- [x] Vectorized version (NumPy)
- [x] Parallel version (mpi4py?)
- [x] GPU version (cupy)
- [ ] Platform independent version (GT4Py)

### Julia
Write the following code:
- [x] Base case stencil (indexed) (-> Michelle)
- [x] Vectorized version (using views/LinearAlgebra package) (-> Seva)
- [x] Parallel version (parallel implementation, force Julia to use all theads) (-> Seva)
- [x] GPU version (CUDA.jl) (-> Seva)
- [ ] Platform independent version (Stencil.jl, maybe own code)

### Rust
Write the following code:
- [x] Base case stencil (-> Paul)
- [x] Vectorized version (once using basic slice operation and once using iterators)(-> Paul)
- [x] Parallel version (using rayon) (-> Paul)


## Report
- [x] Set up chapter titles & structure (introduction, problem statement, ..., conclusion) (-> Seva)
- [x] Abstract (-> Michelle)
- [x] Introduction (-> Paul)
- [x] Problem Statement (-> Paul)
- [ ] Main part (-> all)
- [ ] Conclusion (-> all)
- [ ] Add references if neccessary


## Details
### Rust
#### Running the simulation:
Use the main function resp. main.rs to run the simulation from command line, i.e. when you are not performing benchmarks. \
The easiest way to run the simulation is to use cargo run and supply cmd line arguments: \
```cargo run -- 64 64 64 100 2``` or ```cargo run --release  -- 64 64 64 100 2``` \
The arguments are -- nx ny nz num_iter num_halo, `--release` builds the optimized (instead of default debug) target. \
All in all four versions of the stencil computation were implementes:
- The naive implementation using basic for loops.
- A version using slicing and arithmetic operations provided by the ndarray package.
- A version using iterators over ndarrays
- A multithreaded version using iterators over ndarrays as well as the data parallelism package rayon.

To run either one (but not several at the same time) go to main.rs and uncomment the use statement corresponding to the version
you want to use. (Note that the `update_halo` function always runs the default naive version. This is mainly to make sure that there are no large overheads through inefficient versions of this function. )

#### Plotting:
This function will save 2D slices of the 3D array as csv files which then can be plotted using the provided
python script ```./data/plots.py```.  Without any changes to the code two files will be produced and the array slices will be saved to ./data/in_field.csv and ./data/out_field.csv. The python script `plots.py` can take as cmd line input a variable number of csv file names (without the .csv extension). and will produce a .png file of the same name. 
E.g. run ```python sdata/plots.py in_field out_field```.

#### Benchmarks:
The benchmarks in ```./benches/bench.rs``` will time all of the above mentioned stencil implementations and 
save the report in ```./target/criterion/report/index.html```. See this report for a number of statistical analyses and plots.
You can run the benchmarks from `RustCode/` with `cargo bench --verbose`. To extract all means (mean times over some number of repeated benchmarks for a certain stencil implementation),
run ```py3 ./data/get_means.py```.

A few things to keep in mind:
- The benchmarking happens within the function `stencil_benchmarks_criterion`. Before you run the benchmarks, 
  be aware of which of the stencil versions are actually being benchmarked. Depending on the problem size and the machine you are running on, the benchmark might take some time. 
- There problem size(s) used in the benchmarks are taken from `universal_input/input_dimensions.csv` so check that file as
  well. Adapt the loop in `stencil_benchmarks_criterion` if you only want to run certain problem sizes. 
- For convenience the shell script `bench_get_means.sh` will compile and run the benchmarks and collect all means into a csv
  file saved into `universal_output/`. Common errors when running this script might be mismatches in dimensions between the input dimensios, the loop in the benchmarks and the loop in `get_means.py`.

#### Unit Tests:
There are unit tests for all functions in the different versions of stencil computation. To run all tests type: `cargo test`. The unit tests belonging to a specific version are within the source file corresponding to that version. If you want to just run a specific version's unit tests run e.g. `cargo test stencil2d_naive::unit_tests`. Append `-- --nocapture` to enable printing during tests (disabled by default).

## Notes and General Questions
### Plotting facilities
Seva wants to have homogeneous plots throughout the document and therefore would like to decide on which plotting facility to use. Proposals were:
- Gadfly.jl (benefit: nice graphics in svg format)
- matplotlib (benefit: everybody has python and matplotlib available) \

[Gadfly.jl](https://github.com/GiovineItalia/Gadfly.jl/tree/v1.3.4) was used for the final report. 
### I/O
To make a fair comparison we need to be able to have same dimensions across all programs. This will require loading dimension data and storing the results in a uniform manner. In `universal_input/input_dimensions` the first three columns are the dimensions of the array: nx, ny, nz. The next row is the diffusion coefficient alpha, which is constant for all problem sizes. The last column is the number of iterations to perform, which is also constant. Finally, each row corresponds to one problem size.

### File Acces and Console
All the paths should be relative to the home directory (hpcwc-performance-comparison). If not they should be relative to the language specific directory. In this case the outer shell scrips are updated such that it still works.

### Language minimum versions 
- Julia 1.9
- Python 3.9
- Rust 1.70

In order to have multiple versions available, version managers like "juliaup" or "rustup" can be used.
 
### Recommended VSCode extensions
- Julia (Julia Language Support)
- Python (IntelliSense (Pylance))
- rust-analyzer (Rust language support)



___
# Useful Things from the Default Readme
## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.ethz.ch/vsemenov/hpcwc-performance-comparison/-/settings/integrations)

## Collaborate with your team

- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***


## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Authors
This comparison has been created by Michelle Schneider, Paul Ochs and Vsevolod Semenov in terms of the course "High Performance Computing for Weather and Climate" at the Swiss Federal Institute of Technology, ETH.

## License
No license so far...

