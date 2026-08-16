# Custom TPC-C Kernels for Intel Gaudi

Hand-written TPC-C kernels targeting Intel Gaudi2 accelerators, plus Python
scripts for exercising the Habana (SynapseAI) HPU backend from PyTorch.

TPC-C is the C-like kernel language used to program Gaudi's Tensor Processor
Cores (TPCs) via Habana's Custom Kernel toolchain. The kernels in this repo
implement the forward/backward ops needed for a small fully-connected
network: linear (dense) layers, ReLU, and element-wise multiply.

## Repository structure

```
.
├── kernels/gaudi2/            # TPC-C kernel source (.c)
│   ├── linear_forward.c       # y = x @ W^T + b
│   ├── linear_backward.c      # grad_input / grad_weight / grad_bias
│   ├── relu_forward.c         # out = max(0, x)
│   ├── relu_backward.c        # grad_in = grad_out where x > 0, else 0
│   └── elementwise_multiply.c # out = a * b
└── python/
    ├── train_mnist_tpc.py     # MNIST training baseline on HPU
    └── test_kernel.py         # Sanity checks for a built custom kernel .so
```

## Kernels (`kernels/gaudi2/`)

All kernels operate on the standard 5D TPC index space
`(depth, width, height, batch, fifthDim)` and process data in 64-element
vector chunks (`float64` / `v_f32_*` vector intrinsics), which is the native
vector width for Gaudi2 TPCs.

| File | Signature | Description |
|---|---|---|
| `linear_forward.c` | `main(input, weight, bias, output)` | Dense layer forward pass: accumulates `input * weight` over the input-feature dimension and adds `bias`. |
| `linear_backward.c` | `main(grad_output, input, weight, grad_input, grad_weight, grad_bias)` | Computes gradients for a dense layer. |
| `relu_forward.c` | `main(input, output)` | Elementwise `max(0, x)`. |
| `relu_backward.c` | `main(grad_output, input, grad_input)` | Passes gradient through where the forward input was positive, zeroes it otherwise. |
| `elementwise_multiply.c` | `main(input0, input1, output)` | Elementwise product of two tensors. |

These are kernel source files only — compiling them into a loadable
`.so` requires Habana's TPC-C compiler / Custom Kernel build tooling
(`Habana_Custom_Kernel`), which is not part of this repository.

## Python scripts (`python/`)

### `train_mnist_tpc.py`

Trains a 3-layer fully-connected network (`784 → 128 → 64 → 10`) on MNIST
using `torch.device("hpu")`. It tries to import `TPCReLU` / `TPCLinear` from
a `custom_ops_tpc` module; if that import fails it falls back to plain
`nn.ReLU` / `nn.Linear`.

**Important:** even when `custom_ops_tpc` is available, this script does not
call the hand-written kernels in `kernels/gaudi2/` directly — it does not use
Habana's Custom Op registration API (no GUID, no `registerUserCustomOp`).
`custom_ops_tpc.py` (the wrapper module) is not included in this repo. Until
it, and the Custom Op registration that binds it to the compiled kernels, are
added, this script is an HPU-optimized PyTorch baseline rather than a true
custom-kernel integration.

```bash
python python/train_mnist_tpc.py
```

### `test_kernel.py`

A basic environment/sanity check: confirms a compiled kernel library exists
at `~/tpckernels/Habana_Custom_Kernel/build/libelement_wise_mul.so`, checks
HPU availability, and computes a CPU/PyTorch reference for element-wise
multiply to compare a kernel's output against.

```bash
python python/test_kernel.py
```

## Requirements

- An Intel Gaudi2 accelerator with the Habana SynapseAI software stack
  installed
- `habana_frameworks.torch` (Habana's PyTorch bridge)
- PyTorch, `torchvision`
- Habana's `Habana_Custom_Kernel` build tooling, to compile the `.c` kernels
  into a `.so` before `test_kernel.py` can find them

## Status / known gaps

- `custom_ops_tpc.py` (TPCReLU/TPCLinear wrappers referenced by
  `train_mnist_tpc.py`) is not present in this repo.
- No build script (e.g. CMake/Makefile) is included for compiling the
  kernels in `kernels/gaudi2/` — use Habana's Custom Kernel build tooling.
- No Habana Custom Op registration (GUID + `registerUserCustomOp`) wiring
  the compiled kernels into the PyTorch training path yet.
