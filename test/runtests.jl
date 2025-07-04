using BatchNLPKernels
using Test
using ExaModels
using KernelAbstractions


using OpenCL, pocl_jll, AcceleratedKernels
ExaModels.convert_array(x, ::OpenCLBackend) = CLArray(x)
ExaModels.sort!(array::CLArray; lt = isless) = AcceleratedKernels.sort!(array; lt=lt)
function Base.findall(f::F, bitarray::CLArray) where {F<:Function}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
Base.findall(bitarray::CLArray) = Base.findall(identity, bitarray)


include("luksan.jl")
include("test_diff.jl")
include("api.jl")
include("config.jl")