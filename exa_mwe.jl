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