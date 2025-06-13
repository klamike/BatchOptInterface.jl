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