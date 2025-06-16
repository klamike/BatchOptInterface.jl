function dag_to_expressions(dag::SymbolicAD._DAG)
    exprs = Vector{Any}(undef, length(dag.tape))
    
    for (i, node) in enumerate(dag.tape)
        exprs[i] = if node.operator == SymbolicAD._kNODE_PARAMETER
            :(p[$(node.data)])
        elseif node.operator == SymbolicAD._kNODE_VARIABLE
            :(x[$(node.data)])
        elseif node.operator == SymbolicAD._kNODE_VALUE
            reinterpret(Float64, node.data)
        else
            @assert node.operator > 0
            type, op, nargs = SymbolicAD._operator_to_type_id_nargs(node.operator)
            # child_indices = dag.children[node.data:node.data+nargs-1]
            # child_exprs = [exprs[child_indices[j]] for j in 1:nargs]
            
            if type == :univariate
                op_symbol = dag.registry.univariate_operators[op]
                ret = :($op_symbol($(exprs[dag.children[node.data]])))
                println(ret)
                ret
            elseif type == :univariate_derivative
                op_symbol = dag.registry.univariate_operators[op]
                ret = _get_derivative_expr(op_symbol, exprs[dag.children[node.data]])
                println(ret)
                ret
            elseif type == :univariate_second_derivative
                op_symbol = dag.registry.univariate_operators[op]
                ret = _get_second_derivative_expr(op_symbol, exprs[dag.children[node.data]])
                println(ret)
                ret
            elseif type == :multivariate
                op_symbol = dag.registry.multivariate_operators[op]
                ret = :($op_symbol($(exprs[dag.children[node.data:node.data+nargs-1]]...)))
                println(ret)
                ret
            elseif type == :logic
                op_symbol = dag.registry.logic_operators[op]
                ret = :($op_symbol($(exprs[dag.children[node.data:node.data+nargs-1] .≈ 1.0]...)))
                println(ret)
                ret
            elseif type == :comparison
                op_symbol = dag.registry.comparison_operators[op]
                ret = :($op_symbol($(exprs[dag.children[node.data:node.data+nargs-1]]...)))
                println(ret)
                ret
            else
                error("Unknown operator $type")
            end
        end
    end
    
    return [_to_expr(exprs[i]) for i in dag.indices]
end

function _get_derivative_expr(op::Symbol, arg_expr)
    for (symbol, derivative_expr, _) in MOI.Nonlinear.SYMBOLIC_UNIVARIATE_EXPRESSIONS
        if symbol == op
            return _replace_expr_symbol(copy(derivative_expr), :x, arg_expr)
        end
    end
    error("No derivative for $op")
end
function _get_second_derivative_expr(op::Symbol, arg_expr)
    for (symbol, _, second_derivative_expr) in MOI.Nonlinear.SYMBOLIC_UNIVARIATE_EXPRESSIONS
        if symbol == op
            isnothing(second_derivative_expr) && error("No second derivative for $op")
            return _replace_expr_symbol(copy(second_derivative_expr), :x, arg_expr)
        end
    end
    error("No second derivative for $op")
end
function _replace_expr_symbol(expr::Expr, old_symbol::Symbol, new_expr)
    new_expr_copy = copy(expr)
    for (i, arg) in enumerate(new_expr_copy.args)
        new_expr_copy.args[i] = _replace_expr_symbol(arg, old_symbol, new_expr)
    end
    return new_expr_copy
end
_replace_expr_symbol(expr::Symbol, old_symbol::Symbol, new_expr) = expr == old_symbol ? new_expr : expr
_replace_expr_symbol(expr::Any, ::Symbol, ::Any) = expr



function myevaluate!(dag::SymbolicAD._DAG, x::AbstractVector{T}, p::Vector{Float64}) where T
    reg = dag.registry
    for (i, node) in enumerate(dag.tape)
        @inbounds dag.result[i] = if node.operator == SymbolicAD._kNODE_PARAMETER
            p[node.data]
        elseif node.operator == SymbolicAD._kNODE_VARIABLE
            x[node.data]
        elseif node.operator == SymbolicAD._kNODE_VALUE
            reinterpret(Float64, node.data)
        else
            @assert node.operator > 0
            type, op, nargs = SymbolicAD._operator_to_type_id_nargs(node.operator)
            if type == :univariate
                @assert nargs == 1
                MOI.Nonlinear.eval_univariate_function(
                    reg,
                    op,
                    dag.result[dag.children[node.data]],
                )
            elseif type == :univariate_derivative
                @assert nargs == 1
                MOI.Nonlinear.eval_univariate_gradient(
                    reg,
                    op,
                    dag.result[dag.children[node.data]],
                )
            elseif type == :univariate_second_derivative
                @assert nargs == 1
                MOI.Nonlinear.eval_univariate_hessian(
                    reg,
                    op,
                    dag.result[dag.children[node.data]],
                )
            elseif type == :multivariate
                for j in 1:nargs
                    dag.cache[j] = dag.result[dag.children[node.data+j-1]]
                end
                MOI.Nonlinear.eval_multivariate_function(
                    reg,
                    reg.multivariate_operators[op],
                    view(dag.cache, 1:nargs),
                )
            elseif type == :logic
                @assert nargs == 2
                MOI.Nonlinear.eval_logic_function(
                    reg,
                    reg.logic_operators[op],
                    dag.result[dag.children[node.data]] ≈ 1.0,
                    dag.result[dag.children[node.data+1]] ≈ 1.0,
                )
            else
                @assert type == :comparison
                @assert nargs == 2
                MOI.Nonlinear.eval_comparison_function(
                    reg,
                    reg.comparison_operators[op],
                    dag.result[dag.children[node.data]],
                    dag.result[dag.children[node.data+1]],
                )
            end
        end
    end
    return
end
function myevaluate!(dag::SymbolicAD._DAG, g::SymbolicAD._SymbolicInstance, x)
    if length(dag.input) != length(g.x)
        resize!(dag.input, length(g.x))
    end
    for (i, j) in enumerate(g.x)
        dag.input[i] = x[j]
    end
    myevaluate!(dag, dag.input, g.p)
    for (i, j) in enumerate(dag.indices)
        g.result[i] = dag.result[j]
    end
    return
end
function myevaluate!(model::SymbolicAD.Evaluator, x)
    # if x == model.x
    #     return
    # end
    # if length(x) != length(model.x)
    #     resize!(model.x, length(x))
    # end
    # copyto!(model.x, x)
    o = model.objective
    if o !== nothing
        myevaluate!(model.dag[o.hash], o, x)
    end
    for h in collect(keys(model.constraint_index_by_hash))
        dag = model.dag[h]
        for i in model.constraint_index_by_hash[h]
            myevaluate!(dag, model.constraints[i], x)
        end
    end
    return
end