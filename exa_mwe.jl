using JuMP, MathOptInterface, ExaModels
const ExaMOI = Base.get_extension(ExaModels, :ExaModelsMOI)
include("JuMPToExa/JuMPToExa.jl")

m = Model()
@variable m x
@variable m y
@variable m p ∈ Parameter(1.0)
@variable m q ∈ Parameter(1.0)

exagraph = JuMP.value(
    JuMPToExa(m),
    # sin(x) + 4.0x*sqrt(y)*p^2 + log((x^3-y)^(-2.0f0)) - cos(x)*p*q^(-4)
    # 2x
    # cos(*(1x,2y,3p,4q))
    sum(2i for i in [x,y,p,q])
)
fn = ExaModels._simdfunction(exagraph, 0, 0, 0)  # not sure how offsets work...

using KernelAbstractions, Metal
const ExaKA = Base.get_extension(ExaModels, :ExaModelsKernelAbstractions)

@kernel function kerfb(y, @Const(f), @Const(itr), @Const(x))
    I = @index(Global)
    @inbounds y[1, ExaModels.offset0(f, itr, I)] = f.f(@view(itr[:, I]), @view(x[:, I]))
end

B = 10
y_m = Metal.zeros(1,B)
itr_m = Metal.rand(2, B)
X_m = Metal.rand(2, B)

kerfb(MetalBackend())(y_m, fn, itr_m, X_m, ndrange=size(X_m, 2)); y_m
y_c = fn.f.(eachcol(Matrix(itr_m)), eachcol(Matrix(X_m)))
@assert all(y_c .≈ Matrix(y_m)[1, :])
