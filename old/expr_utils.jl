mutable struct CachingExpr
    expr::Expr
    fn
    fn_set::Bool
    kernel_module::Module
end
function Base.convert(::Type{CachingExpr}, expr::Expr)
    return CachingExpr(expr, nothing, false, KernelEvaluationModule)
end

function _wrap_expr_for_evaluation(expr::Expr)
    return :((x,p)-> @inbounds begin $expr end)
end

function _convert_constants(expr::CachingExpr, T)
    return _convert_constants(expr.expr, T)
end
function _convert_constants(expr::Expr, T)
    new_args = [_convert_constants(arg, T) for arg in expr.args]
    return Expr(expr.head, new_args...)
end
_convert_constants(expr, ::Any) = expr
_convert_constants(expr::Real, T) = T(expr)
_convert_constants(expr::Integer, ::Any) = expr

_to_expr(c) = quote $c end
_to_expr(c::Expr) = c

function _to_int_index(expr::Expr)
    if expr.head == :ref && length(expr.args) == 2
        var_name = expr.args[1]
        index = expr.args[2]
        isa(index, MOI.VariableIndex) && return Expr(:ref, var_name, index.value)
    end
    new_args = [_to_int_index(arg) for arg in expr.args]
    return Expr(expr.head, new_args...)
end
_to_int_index(expr) = expr