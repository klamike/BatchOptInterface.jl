using Revise, JuMP, ExaModels
const MOIN = MOI.Nonlinear

float_type = Float32
jump_to_exa(vr::VariableRef) = begin
    source = JuMP.is_parameter(vr) ? ExaModels.ParSource() : ExaModels.VarSource()
    return source[index(vr).value]
end

_maybe_cast(x::Real) = float_type(x)
_maybe_cast(x::ExaModels.AbstractNode) = x
function JuMP._evaluate_expr(
    registry::MOI.Nonlinear.OperatorRegistry,
    f::typeof(jump_to_exa),
    expr::JuMP.GenericNonlinearExpr,
)
    # The result_stack needs to be ::Real because operators like || return a
    # ::Bool. Also, some inputs may be ::Int.
    stack, result_stack = Any[expr], Any[]  # removed cast to Real
    while !isempty(stack)
        arg = pop!(stack)
        if arg isa GenericNonlinearExpr
            push!(stack, (arg,))  # wrap in (,) to catch when we should eval it.
            for child in arg.args
                push!(stack, child)
            end
        elseif arg isa Tuple{<:GenericNonlinearExpr}
            f_expr = only(arg)
            op, nargs = f_expr.head, length(f_expr.args)
            # TODO(odow): uses private function
            result = if !MOI.Nonlinear._is_registered(registry, op, nargs)
                model = owner_model(f_expr)
                udf = MOI.get(model, MOI.UserDefinedFunction(op, nargs))
                if udf === nothing
                    return error(
                        "Unable to evaluate nonlinear operator $op because " *
                        "it was not added as an operator.",
                    )
                end
                first(udf)((pop!(result_stack) for _ in 1:nargs)...)
            elseif nargs == 1 && haskey(registry.univariate_operator_to_id, op)
                x = pop!(result_stack)
                MOIN.eval_univariate_function(registry, op, x)
            elseif haskey(registry.multivariate_operator_to_id, op)
                args = Union{ExaModels.AbstractNode,float_type}[
                    _maybe_cast(pop!(result_stack)) for _ in 1:nargs  # added cast to float_type if not AbstractNode
                ]
                MOIN.eval_multivariate_function(registry, op, args)
            elseif haskey(registry.logic_operator_to_id, op)
                @assert nargs == 2
                x = pop!(result_stack)
                y = pop!(result_stack)
                MOI.Nonlinear.eval_logic_function(registry, op, x, y)
            else
                @assert haskey(registry.comparison_operator_to_id, op)
                @assert nargs == 2
                x = pop!(result_stack)
                y = pop!(result_stack)
                MOI.Nonlinear.eval_comparison_function(registry, op, x, y)
            end
            push!(result_stack, result)
        else
            push!(result_stack, JuMP._evaluate_expr(registry, f, arg))
        end
    end
    return only(result_stack)
end

function JuMP.value(var_value::typeof(jump_to_exa), ex::GenericAffExpr{T,V}) where {T,V}
    # S = Base.promote_op(var_value, V)
    # U = Base.promote_op(*, T, S)
    # ret = convert(U, ex.constant)
    ret = ex.constant
    for (var, coef) in ex.terms
        ret += coef * var_value(var)
    end
    return ret
end
function JuMP.value(
    var_value::typeof(jump_to_exa),
    ex::GenericQuadExpr{CoefType,VarType},
) where {CoefType,VarType}
    # RetType = Base.promote_op(
    #     (ctype, vtype) -> ctype * var_value(vtype) * var_value(vtype),
    #     CoefType,
    #     VarType,
    # )
    # ret = convert(RetType, value(var_value, ex.aff))
    ret = value(var_value, ex.aff)
    for (vars, coef) in ex.terms
        ret += coef * var_value(vars.a) * var_value(vars.b)
    end
    return ret
end

function MOIN.eval_univariate_function(operator::MOIN._UnivariateOperator, x::T) where {T<:ExaModels.AbstractNode}
    ret = operator.f(x)
    MOIN.check_return_type(T, ret)
    return ret  # removed cast to T
end

function MOIN.eval_univariate_function(
    registry::MOIN.OperatorRegistry,
    id::Integer,
    x::T,
) where {T<:ExaModels.AbstractNode}
    if id <= registry.univariate_user_operator_start
        f, _ = MOIN._eval_univariate(id, x)
        return f  # removed cast to T
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return MOIN.eval_univariate_function(operator, x)
end

function MOIN.eval_multivariate_function(
    registry::MOIN.OperatorRegistry,
    op::Symbol,
    x::AbstractVector{T},
) where {T<:Union{ExaModels.AbstractNode,float_type}}  # removed cast to T
    if op == :+
        return sum(x; init = zero(float_type))
    elseif op == :-
        @assert length(x) == 2
        return x[1] - x[2]
    elseif op == :*
        return prod(x; init = one(float_type))
    elseif op == :^
        @assert length(x) == 2
        # Use _nan_pow here to avoid throwing an error in common situations like
        # (-1.0)^1.5.
        return MOIN._nan_pow(x[1], x[2])
    elseif op == :/
        @assert length(x) == 2
        return x[1] / x[2]
    elseif op == :ifelse
        @assert length(x) == 3
        return ifelse(Bool(x[1]), x[2], x[3])
    elseif op == :atan
        @assert length(x) == 2
        return atan(x[1], x[2])
    elseif op == :min
        return minimum(x)
    elseif op == :max
        return maximum(x)
    end
    id = registry.multivariate_operator_to_id[op]
    offset = id - registry.multivariate_user_operator_start
    operator = registry.registered_multivariate_operators[offset]
    @assert length(x) == operator.N
    ret = operator.f(x)
    MOIN.check_return_type(T, ret)
    return ret  # removed cast to T
end

m = Model()
@variable m x
@variable m y
@variable m p ∈ Parameter(1.0)

exagraph = JuMP.value(
    jump_to_exa,
    # sin(x) + 4.0x*sqrt(y)*p^2 + log((x^3-y)^(-2.0f0))
    # 2x
    # *(x,y,p)
)
# ExaModels.Node1{
#     typeof(sin),
#     ExaModels.Node2{
#         typeof(+),
#         ExaModels.ParIndexed{ExaModels.ParSource, 3},
#         ExaModels.Node2{
#             typeof(*),
#             ExaModels.Node2{
#                 typeof(^),
#                 ExaModels.Var{Int64},
#                 Float64
#             },
#             ExaModels.Var{Int64}
#         }
#     }
# }
fn = ExaModels._simdfunction(exagraph, 0, 0, 0)  # not sure how offsets work...