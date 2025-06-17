using JuMP, ExaModels

include("JuMPToExa/JuMPToExa.jl")

m = Model()
@variable m x
@variable m y
@variable m p ∈ Parameter(1.0)

exagraph = JuMP.value(
    JuMPToExa(m, Float32),
    sin(x) + 4.0x*sqrt(y)*p^2 + log((x^3-y)^(-2.0f0))
    # 2x
    # *(x,y,p)
)
fn = ExaModels._simdfunction(exagraph, 0, 0, 0)  # not sure how offsets work...

using KernelAbstractions, Metal

@kernel function kerfb(y, @Const(f), @Const(itr), @Const(x))
    I = @index(Global)
    @inbounds y[ExaModels.offset0(f, itr, I)] = f.f(itr[I], @view(x[:, I]))
end

y_m = Metal.zeros(1,2)
itr_m = MtlArray([1.0f0 2.0f0;])
X_m = MtlArray([2.0f0 5.0f0;3.0f0 8.0f0])

kerfb(MetalBackend())(y_m, fn, itr_m, X_m, ndrange=size(X_m, 2)); y_m
y_c = fn.f.(eachcol(Matrix(itr_m)), eachcol(Matrix(X_m)))
@assert all(y_c .== Matrix(y_m)[1, :])
