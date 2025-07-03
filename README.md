# BatchNLPKernels.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://klamike.github.io/BatchNLPKernels.jl/dev/)
[![Build Status](https://github.com/klamike/BatchNLPKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/klamike/BatchNLPKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)

`BatchNLPKernels.jl` provides [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels for evaluating problem data from a (parametric) [`ExaModel`](https://github.com/exanauts/ExaModels.jl) for batches of solutions (and parameters). Currently the following functions (as well as their non-parametric variants) are exported:

- `obj_batch!(::BatchModel, X, Θ)`
- `grad_batch!(::BatchModel, X, Θ)`
- `cons_nln_batch!(::BatchModel, X, Θ)`
- `jac_coord_batch!(::BatchModel, X, Θ)`
- `hess_coord_batch!(::BatchModel, X, Θ, Y; obj_weight=1.0)`
- `jprod_nln_batch!(::BatchModel, X, Θ, V)`
- `jtprod_nln_batch!(::BatchModel, X, Θ, V)`
- `hprod_batch!(::BatchModel, X, Θ, Y, V; obj_weight=1.0)`


To use these functions, first wrap your `ExaModel` in a `BatchModel`:

```julia
using BatchNLPKernels

# model::ExaModel = ...

max_batch_size = 64
bm = BatchModel(model, max_batch_size)
```
This pre-allocates work and output buffers. By default, only the buffers to support `obj` and `cons` are allocated. You can specify which buffers to allocate by passing a `BatchModelConfig` to the `BatchModel` constructor.

Then, you can call the batch functions as follows:

```julia
objs = obj_batch!(bm, X, Θ)
```

where `X` and `Θ` are (device) matrices with dimensions `(nvar, batch_size)` and `(nθ, batch_size)` respectively.


Note that the functions do operate in-place, on the buffers stored in the `BatchModel`. For convenience, they also return (view of) the relevant buffers.


