using JuMP, ExaModels
const MOIN = MOI.Nonlinear


struct JuMPToExa{T} <: Function
    v::Vector{VariableRef}
    p::Vector{VariableRef}

    function JuMPToExa(model::Model, ::Type{T}=Float64) where {T<:Real}
        v, p = VariableRef[], VariableRef[]
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
    return source[idx]
end

_maybe_cast(::JuMPToExa{T}, x::Real) where {T<:Real} = T(x)
_maybe_cast(::JuMPToExa{T}, x::T) where {T<:Real} = x
_maybe_cast(::JuMPToExa, x::ExaModels.AbstractNode) = x

function JuMP.value(f::JuMPToExa{F}, ex::GenericAffExpr{T,V}) where {F,T,V}
    # S = Base.promote_op(f, V)
    # U = Base.promote_op(*, T, S)
    # ret = convert(U, ex.constant)
    ret = ex.constant
    for (var, coef) in ex.terms
        ret += _maybe_cast(f, coef) * f(var)
    end
    return _maybe_cast(f, ret)
end
function JuMP.value(
    f::JuMPToExa{F},
    ex::GenericQuadExpr{CoefType,VarType},
) where {F,CoefType,VarType}
    # RetType = Base.promote_op(
    #     (ctype, vtype) -> ctype * f(vtype) * f(vtype),
    #     CoefType,
    #     VarType,
    # )
    # ret = convert(RetType, value(f, ex.aff))
    ret = value(f, ex.aff)
    for (vars, coef) in ex.terms
        ret += _maybe_cast(f, coef) * f(vars.a) * f(vars.b)
    end
    return _maybe_cast(f, ret)
end

include("nonlinear.jl")