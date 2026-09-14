# =============================================================================
# CNN Forward Pass — C++ vs PyTorch Numerical Verification
# HPC4WC Course, ETH Zürich 2026
# =============================================================================

# -----------------------------------------------------------------------------
# Terminal colors (ANSI escape codes)
# -----------------------------------------------------------------------------
C_RESET       := \033[0m
C_BOLD        := \033[1m
C_RED         := \033[31m
C_GREEN       := \033[32m
C_YELLOW      := \033[33m
C_CYAN        := \033[36m
C_BOLD_GREEN  := \033[1;32m
C_BOLD_CYAN   := \033[1;36m
C_BOLD_YELLOW := \033[1;33m

# -----------------------------------------------------------------------------
# Compiler & flags
# -----------------------------------------------------------------------------
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -mcpu=native -ftree-vectorize -fopenmp-simd -Wall -Wextra -Wpedantic -DNDEBUG
PYTHON   := python3

UENV_VIEW := pytorch/v2.9.1:v2

# Training device: make train DEVICE=cpu  (default: cuda)
DEVICE ?= cuda

# Benchmark run counts:  make run-local NUM_RUNS=20 NUM_WARMUP_RUNS=5
NUM_RUNS        ?= 30
NUM_WARMUP_RUNS ?= 10

# Benchmark batch size:  make run-local BATCH_SIZE=64
BATCH_SIZE      ?= 1

# Debug build:     make DEBUG=1 build
ifdef DEBUG
  CXXFLAGS := -std=c++17 -O0 -g -fsanitize=address,undefined -fopenmp-simd -Wall -Wextra
endif

# ASCII art image:  make PRINT_ASCII=1 build
ifdef PRINT_ASCII
  CXXFLAGS += -DPRINT_ASCII
endif

# Emit Assembly:  make ASM=1 build
# Generates annotated .s assembly files in build/ and vectorization reports.
# Routed through the cluster's uenv view so the compiler matches submit_run.sh.
ifdef ASM
  CXXFLAGS += -S -fverbose-asm -fopt-info-vec-optimized
  CXX := uenv run --view=default $(UENV_VIEW) -- g++
endif

CXXFLAGS += -DNUM_RUNS=$(NUM_RUNS) -DNUM_WARMUP_RUNS=$(NUM_WARMUP_RUNS) -DBATCH_SIZE=$(BATCH_SIZE)

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------
BUILD_DIR := build
SRC_CPP   := src/cpp
HEADERS   := $(SRC_CPP)/include
SRC_PY    := src/python
LOGS_DIR  := logs
SCRIPTS   := scripts

# -----------------------------------------------------------------------------
# Sources & binary
# -----------------------------------------------------------------------------
CPP_SRCS := $(SRC_CPP)/benchmark.cpp $(SRC_CPP)/cnn_internals.cpp $(SRC_CPP)/cnn.cpp
CPP_HDRS := $(HEADERS)/cnn_internals.h $(HEADERS)/timing.h $(HEADERS)/cnn.h $(HEADERS)/utils.h
TARGET   := $(BUILD_DIR)/cnn_forward

# -----------------------------------------------------------------------------
# Phony declarations
# -----------------------------------------------------------------------------
.PHONY: help all build run verify train check-env\
        run-local verify-local train-local all-local \
        asm logs clean _check_slurm _data

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@printf '\n$(C_BOLD)CNN HPC4WC — available targets$(C_RESET)\n\n'
	@printf '$(C_BOLD)Cluster (sbatch):$(C_RESET)\n'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'run'             'Submit C++ build + forward pass to Slurm'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'verify'          'Submit C++ build + run + Python verifier to Slurm'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'train'           'Submit GPU training job to Slurm'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'all'             'Submit run + verify + train to Slurm'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'logs'            'Tail the latest Slurm log file'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'check-env'       'Submit environment/version check to Slurm (for report)'
	@printf '\n$(C_BOLD)Local (login node):$(C_RESET)\n'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'build'           'Compile the C++ forward-pass binary'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'asm'             'Generate C++ assembly files (.s) in build/'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'run-local'       'Execute the C++ forward pass directly'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'verify-local'    'Run the Python/PyTorch verifier directly'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'train-local'     'Run PyTorch training directly (DEVICE=cuda|cpu)'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'all-local'       'build → run-local → verify-local'
	@printf '\n'
	@printf '  $(C_CYAN)%-22s$(C_RESET) %s\n' 'clean'           'Remove build artefacts and checkpoints (keeps logs/)'
	@printf '\n$(C_BOLD)Overrides:$(C_RESET)\n'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'ASM=1'          'Generate annotated assembly output (.s) in build/'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'DEBUG=1'        'Build with -O0 -g -fsanitize=address,undefined'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'PRINT_ASCII=1'  'Enable ASCII art image print in C++ binary'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'DEVICE=cpu'     'Use CPU for training (default: cuda)'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'NUM_RUNS=N'     'Timed benchmark runs for run/run-local (default: 30)'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'NUM_WARMUP_RUNS=N' 'Warmup runs for run/run-local (default: 10)'
	@printf '  $(C_YELLOW)%-22s$(C_RESET) %s\n' 'BATCH_SIZE=N'   'Images per forward pass for run/run-local (default: 1)'
	@printf '\n'


FLAGS_SENTINEL := $(BUILD_DIR)/.cxxflags

.PHONY: FORCE
FORCE:

$(FLAGS_SENTINEL): FORCE
	@mkdir -p $(BUILD_DIR)
	@echo '$(CXXFLAGS)' | cmp -s - $@ || echo '$(CXXFLAGS)' > $@

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
$(TARGET): $(CPP_SRCS) $(CPP_HDRS) $(FLAGS_SENTINEL)
	@mkdir -p $(BUILD_DIR)
ifdef ASM
	@printf '$(C_BOLD_YELLOW)Generating Assembly files (.s) in $(BUILD_DIR)/...$(C_RESET)\n'
	$(CXX) $(CXXFLAGS) -I$(SRC_CPP) -S $(SRC_CPP)/cnn_internals.cpp -o $(BUILD_DIR)/cnn_internals.s
	$(CXX) $(CXXFLAGS) -I$(SRC_CPP) -S $(SRC_CPP)/cnn.cpp -o $(BUILD_DIR)/cnn.s
	$(CXX) $(CXXFLAGS) -I$(SRC_CPP) -S $(SRC_CPP)/benchmark.cpp -o $(BUILD_DIR)/benchmark.s
	@printf '$(C_BOLD_GREEN)Assembly files generated in $(BUILD_DIR)/$(C_RESET)\n'
endif
	$(CXX) $(filter-out -S, $(CXXFLAGS)) -I$(SRC_CPP) -o $@ $(CPP_SRCS)
	@printf '$(C_BOLD_GREEN)Build successful:$(C_RESET) $@\n'

build: $(TARGET)

# Dedicated shortcut target to generate assembly only
asm:
	@$(MAKE) ASM=1 build

# -----------------------------------------------------------------------------
# Dataset — fetched into data/ on first use (needs internet; on Santis this
# runs on the login node, before any sbatch). Idempotent: a no-op once present.
# -----------------------------------------------------------------------------
_data:
	@$(PYTHON) $(SRC_PY)/download_data.py

# -----------------------------------------------------------------------------
# Local targets (login-node / quick dev testing)
# -----------------------------------------------------------------------------
run-local: _data $(TARGET)
	@printf '$(C_BOLD_CYAN)--- C++ Forward Pass (local) ---$(C_RESET)\n'
	cd $(SRC_CPP) && ../../$(TARGET)
	@printf '$(C_BOLD_CYAN)--- Python/PyTorch Forward Pass (local) ---$(C_RESET)\n'
	cd $(SRC_PY) && $(PYTHON) benchmark.py --num-runs $(NUM_RUNS) --num-warmup-runs $(NUM_WARMUP_RUNS) --batch-size $(BATCH_SIZE)

verify-local: _data $(TARGET)
	@printf '$(C_BOLD_CYAN)--- Python/PyTorch Verifier (local) ---$(C_RESET)\n'
	cd $(SRC_CPP) && ../../$(TARGET) verify
	cd $(SRC_PY) && $(PYTHON) verify.py

train-local: _data
	@printf '$(C_BOLD_CYAN)--- PyTorch Training (local, device=$(DEVICE)) ---$(C_RESET)\n'
	cd $(SRC_PY) && $(PYTHON) -u train.py --device $(DEVICE)

all-local: build run-local verify-local

# -----------------------------------------------------------------------------
# Slurm
# -----------------------------------------------------------------------------
_check_slurm:
	@command -v sbatch >/dev/null 2>&1 || \
        { printf '$(C_RED)Error:$(C_RESET) sbatch not found — this target requires a Slurm cluster.\n'; exit 1; }

run: _data _check_slurm
	@mkdir -p $(LOGS_DIR)
	@sbatch --export=ALL,NUM_RUNS=$(NUM_RUNS),NUM_WARMUP_RUNS=$(NUM_WARMUP_RUNS),BATCH_SIZE=$(BATCH_SIZE) $(SCRIPTS)/submit_run.sh

verify: _data _check_slurm
	@mkdir -p $(LOGS_DIR)
	@sbatch $(SCRIPTS)/submit_verify.sh

train: _data _check_slurm
	@mkdir -p $(LOGS_DIR)
	@sbatch $(SCRIPTS)/submit_pytorch.sh

all: run verify train

check-env: _check_slurm
	@mkdir -p $(LOGS_DIR)
	@sbatch $(SCRIPTS)/submit_check_env.sh

logs:
	@if [ -z "$$(ls -A $(LOGS_DIR)/ 2>/dev/null)" ]; then \
        printf '$(C_YELLOW)No log files yet in $(LOGS_DIR)/$(C_RESET)\n'; \
    else \
        tail -f $(LOGS_DIR)/$$(ls -t $(LOGS_DIR)/ | head -1); \
    fi

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
clean:
	rm -f $(TARGET)
	rm -f $(FLAGS_SENTINEL)
	rm -f $(SRC_PY)/model-fold-*.pth
	@printf '$(C_GREEN)Clean complete.$(C_RESET) (logs/ and build/*.s left untouched)\n'
