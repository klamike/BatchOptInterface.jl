using JuMP, MathOptInterface, ExaModels
const ExaMOI = Base.get_extension(ExaModels, :ExaModelsMOI)
include("JuMPToExa/JuMPToExa.jl")

m = GenericModel{Float32}()
@variable m x
@variable m y
@variable m p ∈ Parameter(1.0)
@variable m q ∈ Parameter(1.0)

exagraph = JuMP.value(
    JuMPToExa(m, Float32),
    # sin(x) + 4.0x*sqrt(y)*p^2 + log((x^3-y)^(-2.0f0)) - cos(x)*p*q^(-4)
    # 2x
    # cos(*(1x,2y,3p,4q))
    sum(2i for i in [x,y,p,q])
)
fn = ExaModels._simdfunction(exagraph, 0, 0, 0)  # not sure how offsets work...
# ExaModels.Node2{
#     typeof(+),
#     ExaModels.Node2{
#         typeof(+),
#         ExaModels.Node2{
#             typeof(+),
#             ExaModels.Node2{
#                 typeof(*),
#                 Float32,
#                 ExaModels.Var{Int64}
#             },
#             ExaModels.Node2{
#                 typeof(*),
#                 Float32,
#                 ExaModels.Var{Int64}
#             }
#         },
#         ExaModels.Node2{
#             typeof(*),
#             Float32,
#             ExaModels.ParIndexed{ExaModels.ParSource, 1}
#         }
#     },
#     ExaModels.Node2{
#         typeof(*),
#         Float32,
#         ExaModels.ParIndexed{ExaModels.ParSource, 2}
#     }
# }
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

core = ExaCore(Float32)
bin = ExaMOI.BinNull()
bin = ExaMOI.exafy_obj(moi_function(
    sum(2.0f0*i for i in [x,y,p,q])
), bin)
ExaMOI.build_objective(core, bin)


function _obj(backend, objbuffer, obj, x)
    if !isempty(obj.itr)
        kerfb(backend)(objbuffer, obj.f, obj.itr, x; ndrange = length(obj.itr))
    end
    _obj(backend, objbuffer, obj.inner, x)
    ExaKA.synchronize(backend)
end
function _obj(backend, objbuffer, obj, x)
    if !isempty(obj.itr)
        ExaKA.kerf(backend)(objbuffer, obj.f, obj.itr, x; ndrange = length(obj.itr))
    end
    _obj(backend, objbuffer, obj.inner, x)
    ExaKA.synchronize(backend)
end

core.obj = ExaModels.Objective(
    ExaModels.Objective(
        core.obj.inner.inner,
        core.obj.inner.f,
        MtlArray(convert(Vector{Tuple{Int32,Float32}}, core.obj.inner.itr))
    ),
    core.obj.f,
    begin 
        m = MtlMatrix{Int32}(undef, 1, B)
        m[1,:] .= 1:B
        m
    end
)
X_e = [X_m; itr_m]
y_e = Metal.zeros(1,B)
_obj(MetalBackend(), y_e, core.obj, X_e)
@assert all(y_c .≈ Matrix(y_e)[1, :])

# core.obj.inner.f.f
# ExaModels.Node2{
#     typeof(*),
#     ExaModels.Var{ExaModels.ParIndexed{ExaModels.ParSource, 1}},
#     ExaModels.ParIndexed{ExaModels.ParSource, 2}
# }