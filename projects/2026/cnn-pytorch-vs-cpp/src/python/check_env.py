"""
check_env.py

Run this on Santis, inside the same uenv + venv as used for train.py / benchmark.py,
to assess exact software-stack.

Use with submit_check_env.sh via  make check-env 
"""

import sys
import platform

import torch


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("Python")
    print(sys.version)
    print(platform.platform())

    section("PyTorch")
    print("torch.__version__     :", torch.__version__)
    print("torch.version.cuda    :", torch.version.cuda)
    print("cudnn version         :", torch.backends.cudnn.version())
    print("mkldnn available      :", torch.backends.mkldnn.is_available())
    print("num threads (default) :", torch.get_num_threads())

    section("torch.__config__.show()")
    print(torch.__config__.show())

    section("Other packages actually used in this project")
    # torch, numpy: used everywhere (model.py, train.py, benchmark.py, verify.py)
    # torchvision, sklearn: used only in train.py (MNIST loading, K-fold splits)
    # Note: scikit-learn installs as "scikit-learn" but imports as "sklearn".
    packages = ["numpy", "torchvision", "sklearn"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            print(f"{pkg:12s}:", getattr(mod, "__version__", "unknown"))
        except ImportError:
            print(f"{pkg:12s}: not installed")


if __name__ == "__main__":
    main()
