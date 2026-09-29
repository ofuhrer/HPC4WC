import numpy as np
from pathlib import Path

SRC = "array/test_field_fortran_cpu_64_64_64_1024.npy"   # reference field
TARGETS_FILE = "targets.txt"                              # target fields

RTOL = 1e-4
ATOL = 1e-4


def compare_pair(src_f, trg, rtol, atol):
    trg_f = np.load(trg)
    if src_f.shape != trg_f.shape:
        return False, f"shape mismatch {src_f.shape} vs {trg_f.shape}"
    matches = np.allclose(src_f, trg_f, rtol=rtol, atol=atol, equal_nan=True)
    if matches:
        return True, "equal"
    else:
        max_diff = np.max(np.abs(src_f - trg_f))
        return False, f"max abs diff: {max_diff:.6e}"


def main():
    src_path = Path(SRC)
    if not src_path.exists():
        raise FileNotFoundError(f"Reference file '{SRC}' not found.")

    targets_path = Path(TARGETS_FILE)
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file '{TARGETS_FILE}' not found.")

    src_f = np.load(src_path)

    with open(targets_path) as f:
        targets = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    targets = list(dict.fromkeys(targets)) 

    if not targets:
        raise ValueError(f"No targets found in '{TARGETS_FILE}'.")

    all_equal = True

    for trg in targets:
        trg_path = Path(trg)
        if not trg_path.exists():
            print(f"[FAIL] trg not found: '{trg}'")
            all_equal = False
            continue

        matches, detail = compare_pair(src_f, trg, RTOL, ATOL)
        status = "OK" if matches else "FAIL"
        print(f"[{status}] '{SRC}' vs '{trg}' ({detail})")
        if not matches:
            all_equal = False

    print()
    print(f"RESULT: {all_equal}")


if __name__ == "__main__":
    main()