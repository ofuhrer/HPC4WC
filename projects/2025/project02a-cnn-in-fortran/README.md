# cnn-in-fortran
 HPC4WC group project about CNN performance in HPC

List of contents:

### /src/train_and_test_unet:
##### Code for training and validation the UNet temperature field extrapolation model.
dataset.py: Creates a iteratable pytorch dataset from temperature field data.\
dataloader.py: Class for dataloader for both training and validation.\
models.py: Implementation of the UNet model.\
train_and_test.py: Main training and validation loop.\
training_record_plot & visulization.py: Plotting scripts for results and training statistics.\
pt2ts.py: Converts and save pytorch model to torchscript.


### /src/time_counting:
##### Code for measuring different benchmark times.
plots/: Contains plots used in the report.\
Makefile: makefile for compiling Fortran codes.\
Time_Counting-Depth.ipynb: Notebook for measuring inference, data loading, model loading time for different depths.\
Time_Counting.ipynb: Notebook for measuring inference, data loading, model loading time for model D5.\
Validate.ipynb: Notebook for Validation of model outputs.\
cuda_sync.c: function for GPU synchronization call in Fortran.\
dataloader.py: same as in train_and_test_unet.\
dataset.py: same as in train_and_test_unet.\
get_one.ipynb: Notebook for extracting one sample from the dataset for inference.\
infer_fortran_cpu.f90: code for inference in Fortran on CPU.\
infer_fortran_cuda.f90: code for inference in Fortran on GPU.\
models.py: same as in train_and_test_unet.\
validate_fortran_cuda.f90: validation code for Fortran output (run in Validate.ipynb)
