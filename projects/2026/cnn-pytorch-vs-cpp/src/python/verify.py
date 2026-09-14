import json
import os
import struct
import sys

import numpy as np
import torch
from model import CNN

ATOL = 1e-4

def load_bin(filepath, shape):
    '''
      A binary file is read and reshaped into a PyTorch tensor.
    '''
    data = np.fromfile(filepath, dtype=np.float32)
    return torch.tensor(data).view(shape)

# 1. The model is initialized.
model = CNN(in_channels=1, num_classes=10)
model.eval()

# 2. Binary weights are loaded back into PyTorch to guarantee identical state.
w_path = 'weights_cpp/'

with torch.no_grad():
    model.model[0].weight.copy_(load_bin(w_path + 'model_0_weight.bin', (32, 1, 3, 3)))
    model.model[0].bias.copy_(load_bin(w_path + 'model_0_bias.bin', (32,)))

    model.model[3].weight.copy_(load_bin(w_path + 'model_3_weight.bin', (64, 32, 3, 3)))
    model.model[3].bias.copy_(load_bin(w_path + 'model_3_bias.bin', (64,)))

    model.model[6].weight.copy_(load_bin(w_path + 'model_6_weight.bin', (128, 64, 3, 3)))
    model.model[6].bias.copy_(load_bin(w_path + 'model_6_bias.bin', (128,)))

    model.model[10].weight.copy_(load_bin(w_path + 'model_10_weight.bin', (10, 128)))
    model.model[10].bias.copy_(load_bin(w_path + 'model_10_bias.bin', (10,)))

# 3. The full MNIST test split is loaded directly from the IDX file.
mnist_path = '../../data/MNIST/raw/t10k-images-idx3-ubyte'

if not os.path.exists(mnist_path):
    sys.exit(f"MNIST test data not found at {mnist_path} — "
             f"run 'python3 download_data.py' (needs internet).")

with open(mnist_path, 'rb') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    image_data = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)

image_tensor = torch.tensor(image_data, dtype=torch.float32).view(num, 1, rows, cols) / 255.0

# 4. The PyTorch forward pass is executed over all images.
with torch.no_grad():
    torch_logits = model(image_tensor).numpy()  # (num, 10)
torch_pred = torch_logits.argmax(axis=1)


# 5. Discover which C++ implementations were verified, via the manifest
#    benchmark.cpp writes in `verify` mode
with open(w_path + 'cpp_verify_manifest.json') as f:
    implementations = json.load(f)

print(f"Comparing {len(implementations)} C++ implementation(s) against PyTorch "
      f"over {num} images (atol={ATOL:g}).\n")

all_passed = True
summary = []

for impl in implementations:
    name = impl['name']
    cpp_logits = np.fromfile(w_path + impl['file'], dtype=np.float32).reshape(num, 10)

    max_abs_diff = np.max(np.abs(torch_logits - cpp_logits))
    cpp_pred = cpp_logits.argmax(axis=1)
    pred_mismatches = int(np.sum(torch_pred != cpp_pred))
    passed = bool(np.allclose(torch_logits, cpp_logits, atol=ATOL) and pred_mismatches == 0)
    all_passed &= passed
    summary.append((name, max_abs_diff, pred_mismatches, passed))

    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"    Max absolute logit difference: {max_abs_diff:.5e}")
    print(f"    Prediction mismatches:         {pred_mismatches} / {num}\n")

print("=== Summary ===")
print(f"{'Implementation':<42} | {'Status':<6} | {'Max |diff|':<12} | Mismatches")
print("-" * 80)
for name, diff, mism, passed in summary:
    print(f"{name:<42} | {'PASS' if passed else 'FAIL':<6} | {diff:<12.3e} | {mism}/{num}")

# Non-zero exit on failure, so `make verify` and the Slurm job actually fail
# instead of reporting success.
if not all_passed:
    failed = [name for name, _, _, passed in summary if not passed]
    print(f"\n{len(failed)} implementation(s) FAILED: {', '.join(failed)}")
    sys.exit(1)

print(f"\nAll {len(summary)} implementation(s) match PyTorch within atol={ATOL:g}.")


