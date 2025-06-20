function JuMP.value(
    f::JuMPToExa{T},
    expr::JuMP.GenericNonlinearExpr,
) where {T<:Real}
    registry = MOI.Nonlinear.OperatorRegistry()
    # The result_stack needs to be ::Real because operators like || return a
    # ::Bool. Also, some inputs may be ::Int.
    stack, result_stack = Any[expr], Any[]
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
                MOIN_eval_univariate_function(f, registry, op, x)
            elseif haskey(registry.multivariate_operator_to_id, op)
                args = [pop!(result_stack) for _ in 1:nargs]
                MOIN_eval_multivariate_function(f, registry, op, args)
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

function MOIN_eval_univariate_function(
    f::JuMPToExa{F},
    registry::MOIN.OperatorRegistry,
    op::Symbol,
    x::T,
) where {F<:Real,T<:Union{ExaModels.AbstractNode,Real}}
    id = registry.univariate_operator_to_id[op]
    return MOIN_eval_univariate_function(f, registry, id, x)
end

function MOIN_eval_univariate_function(
    ::JuMPToExa{F},
    operator::MOIN._UnivariateOperator, x::T
) where {F<:Real,T<:Union{ExaModels.AbstractNode,Real}}
    ret = operator.f(x)
    return _maybe_cast(F, ret)
end

function MOIN_eval_univariate_function(
    f::JuMPToExa{F},
    registry::MOIN.OperatorRegistry,
    id::Integer,
    x::T,
) where {F<:Real,T<:Union{ExaModels.AbstractNode,Real}}
    if id <= registry.univariate_user_operator_start
        v, _ = MOIN._eval_univariate(id, x)
        return _maybe_cast(F, v)
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return MOIN_eval_univariate_function(f, operator, x)
end

function MOIN_eval_multivariate_function(
    ::JuMPToExa{F},
    registry::MOIN.OperatorRegistry,
    op::Symbol,
    x::AbstractVector{T},
) where {F<:Real,T<:Union{ExaModels.AbstractNode,Real}}
    if op == :+
        return _maybe_cast(F, sum(x; init = zero(F)))
    elseif op == :-
        @assert length(x) == 2
        return _maybe_cast(F, x[1] - x[2])
    elseif op == :*
        return _maybe_cast(F, prod(x; init = one(F)))
    elseif op == :^
        @assert length(x) == 2
        # Use _nan_pow here to avoid throwing an error in common situations like
        # (-1.0)^1.5.
        return _maybe_cast(F, MOIN._nan_pow(x[1], x[2]))
    elseif op == :/
        @assert length(x) == 2
        return _maybe_cast(F, x[1] / x[2])
    elseif op == :ifelse
        @assert length(x) == 3
        return _maybe_cast(F, ifelse(Bool(x[1]), x[2], x[3]))
    elseif op == :atan
        @assert length(x) == 2
        return _maybe_cast(F, atan(x[1], x[2]))
    elseif op == :min
        return _maybe_cast(F, minimum(x))
    elseif op == :max
        return _maybe_cast(F, maximum(x))
    end
    id = registry.multivariate_operator_to_id[op]
    offset = id - registry.multivariate_user_operator_start
    operator = registry.registered_multivariate_operators[offset]
    @assert length(x) == operator.N
    ret = operator.f(x)
    return _maybe_cast(F, ret)
end

function eval_comparison_function(
    ::MOIN.OperatorRegistry,
    op::Symbol,
    lhs::T,
    rhs::T,
) where {T}
    if op == :<=
        return _maybe_cast(Bool, lhs <= rhs)
    elseif op == :>=
        return _maybe_cast(Bool, lhs >= rhs)
    elseif op == :(==)
        return _maybe_cast(Bool, lhs == rhs)
    elseif op == :<
        return _maybe_cast(Bool, lhs < rhs)
    else
        @assert op == :>
        return _maybe_cast(Bool, lhs > rhs)
    end
end

function eval_logic_function(
    ::MOIN.OperatorRegistry,
    op::Symbol,
    lhs::T,
    rhs::T,
) where {T}
    if op == :&&
        return _maybe_cast(Bool, lhs && rhs)
    else
        @assert op == :||
        return _maybe_cast(Bool, lhs || rhs)
    end
end