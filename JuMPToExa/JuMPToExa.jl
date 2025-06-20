using JuMP, ExaModels
const MOIN = MOI.Nonlinear


struct JuMPToExa{T} <: Function
    v::Vector{AbstractVariableRef}
    p::Vector{AbstractVariableRef}

    function JuMPToExa(model::GenericModel{TM}; T=nothing) where {TM}
        T = isnothing(T) ? value_type(TM) : T
        v, p = [], []
        for vp in all_variables(model)
            JuMP.is_parameter(vp) ? push!(p, vp) : push!(v, vp)
        end
        new{T}(v, p)
    end
end

function (jte::JuMPToExa)(vr::VariableRef)
    is_param = JuMP.is_parameter(vr)
    source = is_param ? ExaModels.ParSource() : ExaModels.VarSource()
    pv = is_param ? jte.p : jte.v
    idx = findfirst(==(vr), pv)
    @assert !isnothing(idx)
    return source[idx]
end

_maybe_cast(::Type{T}, x::Real) where {T<:Real} = T(x)
_maybe_cast(::Type{T}, x::T) where {T} = x
_maybe_cast(::Any, x::ExaModels.AbstractNode) = x

function JuMP.value(f::JuMPToExa{F}, ex::GenericAffExpr{T,V}) where {F,T,V}
    ret = _maybe_cast(F, ex.constant)
    for (var, coef) in ex.terms
        ret += _maybe_cast(F, coef) * f(var)
    end
    return _maybe_cast(F, ret)
end
function JuMP.value(
    f::JuMPToExa{F},
    ex::GenericQuadExpr{CoefType,VarType},
) where {F,CoefType,VarType}
    ret = _maybe_cast(F, value(f, ex.aff))
    for (vars, coef) in ex.terms
        ret += _maybe_cast(F, coef) * f(vars.a) * f(vars.b)
    end
    return _maybe_cast(F, ret)
end

include("nonlinear.jl")