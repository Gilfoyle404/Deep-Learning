#!/usr/bin/env python3
# test_kernel.py

import os
import sys
import torch
import habana_frameworks.torch as htorch  # noqa: F401

# Use ~ so it matches your actual $HOME (root or vjawahar)
KERNEL_LIB = os.path.expanduser(
    "~/tpckernels/Habana_Custom_Kernel/build/libelement_wise_mul.so"
)


def test_kernel():
    print("=" * 70)
    print("Habana Custom Kernel - Basic Environment Test")
    print("=" * 70)

    # 1) Check that the kernel library file exists on disk
    print("\n[1/4] Checking kernel library file...")
    print(f"    Checking path: {KERNEL_LIB}")
    if not os.path.exists(KERNEL_LIB):
        print(f"    ✗ FAILED: Kernel library not found at:\n        {KERNEL_LIB}")
        return False

    size_bytes = os.path.getsize(KERNEL_LIB)
    print(f"    ✓ PASSED: Found kernel library")
    print(f"    Size : {size_bytes:,} bytes")

    # 2) Check HPU availability
    print("\n[2/4] Checking HPU availability...")
    if not torch.hpu.is_available():
        print("    ✗ FAILED: torch.hpu.is_available() is False")
        return False

    print("    ✓ PASSED: HPU is available")

    # 3) Print HPU device info
    print("\n[3/4] Getting HPU device information...")
    try:
        device = torch.device("hpu")
        device_name = torch.hpu.get_device_name(0)
        print("    ✓ PASSED: HPU device detected")
        print(f"    Device: {device}")
        print(f"    Name  : {device_name}")
    except Exception as e:
        print(f"    ✗ FAILED: Could not query HPU device: {e}")
        return False

    # 4) Create test tensors on HPU and compute reference output
    print("\n[4/4] Creating test tensors and computing reference...")
    try:
        shape = (2, 4, 3, 1, 64)
        input0 = torch.randn(shape, dtype=torch.float32, device=device)
        input1 = torch.randn(shape, dtype=torch.float32, device=device)
        output_ref = input0 * input1

        print("    ✓ PASSED: Tensors created and reference output computed")
        print(f"    Tensor shape : {shape}")
        print(f"    Num elements : {input0.numel():,}")
        print(f"    Sample values (first 5 elements):")
        print(f"      input0: {input0.flatten()[:5].tolist()}")
        print(f"      input1: {input1.flatten()[:5].tolist()}")
        print(f"      output: {output_ref.flatten()[:5].tolist()}")
    except Exception as e:
        print(f"    ✗ FAILED: Error creating tensors or computing reference: {e}")
        return False

    print("\n" + "=" * 70)
    print("All basic checks passed.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)
