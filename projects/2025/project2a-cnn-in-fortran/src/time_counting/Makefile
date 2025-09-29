FC = gfortran

FFLAGS = -I/users/class171/FTorchbin/include \
         -I/users/class171/FTorchbin/include/ftorch \
         -I/users/class171/Project/FTorch/build/modules

# CUDA Runtime 和 ftorch 库路径
LDFLAGS = -L/users/class171/FTorchbin/lib64 -L/user-environment/env/default/lib64 -lftorch -lcudart

SRC_VALIDATE = validate_fortran_cuda.f90
SRC_INFER_CUDA = infer_fortran_cuda.f90
SRC_INFER_CPU  = infer_fortran_cpu.f90

EXE_VALIDATE = validate_fortran_cuda.x
EXE_INFER_CUDA = infer_fortran_cuda.x
EXE_INFER_CPU  = infer_fortran_cpu.x

MODEL ?= checkpoint/UNet_cuda_fortran.pt
DATADIR ?= data

# ---------------- 编译规则 ----------------
all: $(EXE_VALIDATE) $(EXE_INFER_CUDA) $(EXE_INFER_CPU)

# validate 和 CPU 不依赖 cuda_sync.o
$(EXE_VALIDATE): $(SRC_VALIDATE)
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

$(EXE_INFER_CUDA): $(SRC_INFER_CUDA) cuda_sync.o
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

$(EXE_INFER_CPU): $(SRC_INFER_CPU)
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

# ---------------- 运行规则 ----------------
run_validate: $(EXE_VALIDATE)
	./$(EXE_VALIDATE) $(MODEL) $(DATADIR)

run_infer_cuda: $(EXE_INFER_CUDA)
	LD_LIBRARY_PATH=/user-environment/env/default/lib64:$$LD_LIBRARY_PATH ./$(EXE_INFER_CUDA) $(MODEL) $(DATADIR)

run_infer_cpu: $(EXE_INFER_CPU)
	./$(EXE_INFER_CPU) $(MODEL) $(DATADIR)

# ---------------- 清理 ----------------
clean:
	rm -f $(EXE_VALIDATE) $(EXE_INFER_CUDA) $(EXE_INFER_CPU) *.o *.mod
