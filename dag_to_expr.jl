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
            child_indices = dag.children[node.data:node.data+nargs-1]
            child_exprs = [exprs[child_indices[j]] for j in 1:nargs]
            
            if type == :univariate
                op_symbol = dag.registry.univariate_operators[op]
                :($op_symbol($(only(child_exprs))))
            elseif type == :univariate_derivative
                op_symbol = dag.registry.univariate_operators[op]
                _get_derivative_expr(op_symbol, child_exprs[1])
            elseif type == :univariate_second_derivative
                op_symbol = dag.registry.univariate_operators[op]
                _get_second_derivative_expr(op_symbol, child_exprs[1])
            elseif type == :multivariate
                op_symbol = dag.registry.multivariate_operators[op]
                :($op_symbol($(child_exprs...)))
            elseif type == :logic
                op_symbol = dag.registry.logic_operators[op]
                :($op_symbol($(child_exprs...)))
            elseif type == :comparison
                op_symbol = dag.registry.comparison_operators[op]
                :($op_symbol($(child_exprs...)))
            else
                error("Unknown operator $type")
            end
        end
    end
    
    return [exprs[i] for i in dag.indices]
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
_replace_expr_symbol(expr::Any, old_symbol::Symbol, new_expr) = expr
function _extract_dag_variables(dag::SymbolicAD._DAG)
    variables = Int[]
    for node in dag.tape
        if node.operator == SymbolicAD._kNODE_VARIABLE && !(node.data in variables)
            push!(variables, node.data)
        end
    end
    return sort!(variables)
end