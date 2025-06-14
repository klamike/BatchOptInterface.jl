module KernelEvaluationModule
    using KernelAbstractions
end

function _wrap_expr_for_kernel(expr::Expr, name)
    return :(@kernel $name(_o_,@Const(_x_),@Const(_p_))-> @inbounds begin
        i = @index(Global)
        x = @view(_x_[:, i])
        p = _p_
        _o_[i] = $expr
    end; $name)
end
