import MathOptInterface as MOI
import MathOptInterface.Nonlinear.SymbolicAD as SymbolicAD

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
using KernelAbstractions
const KA = KernelAbstractions

include("dag_to_expr.jl")
include("evaluator_to_expr.jl")

# Dedicated module for kernel evaluation to avoid name conflicts/age issues
module KernelEvaluationModule
    using KernelAbstractions
end

mutable struct CachingExpr
    expr::Expr
    fn
    fn_set::Bool
    kernel_module::Module
end

function Base.convert(::Type{CachingExpr}, expr::Expr)
    return CachingExpr(expr, (x,p)->nothing, false, KernelEvaluationModule)
end

function build_typed_expressions(
    evaluator::SymbolicAD.Evaluator;
    expr_type=CachingExpr,
    container_type=Vector,
    array_type=Vector,
    number_type=Float64
)
    return convert(
        EvaluatorExpressions{
            expr_type,
            container_type{expr_type},
            array_type{number_type},
            container_type{array_type{number_type}}
        },
        _evaluator_to_expressions(evaluator)
    )
end

struct ExprEvaluator
    expressions::EvaluatorExpressions
    objective
    objective_gradient
    constraints
    constraint_jacobian
    hessian_lagrangian
    temp_hessian_batch
    jacobian_structure
    hessian_structure
end
# TODO: get n_vars and n_constraints from expressions
function ExprEvaluator(expressions::EvaluatorExpressions, n_vars, n_constraints; backend=KA.CPU(), zeros=KA.zeros, batch_size=64)
    number_type = eltype(expressions.objective_p)
    return ExprEvaluator(
        expressions,
        zeros(backend, number_type, batch_size),
        zeros(backend, number_type, n_vars, batch_size),
        zeros(backend, number_type, n_constraints, batch_size),
        zeros(backend, number_type, length(expressions.jacobian_structure), batch_size),
        zeros(backend, number_type, length(expressions.hessian_structure), batch_size),
        zeros(backend, number_type, length(expressions.hessian_structure), batch_size),
        Vector{Tuple{Int,Int}}(undef, 0),
        Vector{Tuple{Int,Int}}(undef, 0)
    )
end

"""
    evaluate_expressions_batch(variables, x_batch, σ_batch, μ_batch, expressions; backend=nothing)

Evaluate symbolic expressions with Hessian Lagrangian scaling.

Returns NamedTuple with: `objective`, `objective_gradient`, `constraints`, 
`constraint_jacobian`, `hessian_lagrangian`, `jacobian_structure`, `hessian_structure`
"""
function evaluate_expressions_batch(
    expressions::EvaluatorExpressions,
    x_batch::AbstractMatrix,
    σ_batch::AbstractVector,
    μ_batch::AbstractMatrix; # dual indexing?
    backend=KA.CPU(),
    number_type=eltype(x_batch),
    expr_evaluator=nothing
)
    # TODO: assert shapes are consistent
    n_vars, n_points = size(x_batch)
    n_constraints = length(expressions.constraints)

    # Pre-allocate working memory
    _expr_evaluator = isnothing(expr_evaluator) ? ExprEvaluator(expressions, n_vars, n_constraints, backend=backend, batch_size=n_points) : expr_evaluator
    objective = _expr_evaluator.objective
    objective_gradient = _expr_evaluator.objective_gradient
    constraints = _expr_evaluator.constraints
    constraint_jacobian = _expr_evaluator.constraint_jacobian
    hessian_lagrangian = _expr_evaluator.hessian_lagrangian
    temp_hessian_batch = _expr_evaluator.temp_hessian_batch

    # Evaluate all components
    _eval_objective_batch!(objective, expressions, x_batch, number_type, backend)
    _eval_objective_gradient_batch!(objective_gradient, expressions, x_batch, number_type, backend)
    _eval_constraints_batch!(constraints, expressions, x_batch, number_type, backend)
    _eval_constraint_jacobian_batch!(constraint_jacobian, expressions, x_batch, number_type, backend)
    _eval_hessian_lagrangian_batch!(hessian_lagrangian, temp_hessian_batch, expressions, x_batch, σ_batch, μ_batch, number_type, backend)

    return _expr_evaluator
end


"""
    _maybe_eval_batch(output, expr, x_batch, p, var_map, number_type, backend)

Evaluate expression over batch with variable remapping.

!!! warning
    Uses `eval` and `invokelatest` in the GPU case.
"""
function _maybe_eval_batch(output, expr::CachingExpr, x_batch::AbstractMatrix, p_batch, number_type, ::KA.CPU)
    func = if !(expr.fn_set)
        # Remap expression variables
        remapped_expr = _convert_constants(expr, number_type)
        wrapped_expr = _wrap_expr_for_evaluation(remapped_expr)
        temp_func = @RuntimeGeneratedFunction(wrapped_expr)
        expr.fn = temp_func
        expr.fn_set = true
        expr.fn
    else
        expr.fn
    end
    
    _eval_batch_cpu(output, func, x_batch, p_batch)
    return nothing
end

function _maybe_eval_batch(output, expr, x_batch::AbstractMatrix, p_batch, number_type, backend)
    func = if !(expr isa CachingExpr && expr.fn_set)
        # Remap expression variables
        remapped_expr = _convert_constants(expr, number_type)

        if KA.isgpu(backend)
            kernel_expr = _wrap_expr_for_kernel(remapped_expr, gensym("_expr_kernel"))
            # make kernel (compilation seems to be lazy)
            expr.fn = Base.eval(KernelEvaluationModule, kernel_expr)
            expr.fn_set = true

            expr.fn
        else
            @assert expr isa CachingExpr
            wrapped_expr = _wrap_expr_for_evaluation(remapped_expr)
            temp_func = @RuntimeGeneratedFunction(wrapped_expr)
            expr.fn = temp_func
            expr.fn_set = true

            expr.fn
        end
    else
        expr.fn
    end
    if KA.isgpu(backend)
        try
            # TODO: all this invokelatest/module stuff is a mess
            kernel_instance = Base.invokelatest(func, backend, 64)
            Base.invokelatest(kernel_instance, output, x_batch, p_batch, ndrange=size(x_batch, 2))
        finally
            
        end
    else
        @assert expr isa CachingExpr
        _eval_batch_cpu(output, func, x_batch, p_batch)
    end

    return nothing
end
_maybe_eval_batch(output, ::Nothing, x_batch::AbstractMatrix, p_batch, number_type, backend) = nothing

@generated function _eval_batch_cpu(output, func::F, x_batch, p_batch) where F
    quote
        @inbounds for (i, col) in enumerate(eachcol(x_batch))
            output[i] = func(col, p_batch)
        end
    end
end

function _wrap_expr_for_kernel(expr::Expr, name)
    return :(@kernel $name(_o_,@Const(_x_),@Const(_p_))-> @inbounds begin
        i = @index(Global)
        x = @view(_x_[:, i])
        p = _p_
        _o_[i] = $expr
    end; $name)
end
function _wrap_expr_for_evaluation(expr::Expr)
    return :((x,p)-> @inbounds begin $expr end)
end

function _convert_constants(expr::CachingExpr, number_type)
    return _convert_constants(expr.expr, number_type)
end
function _convert_constants(expr::Expr, number_type)
    new_args = [_convert_constants(arg, number_type) for arg in expr.args]
    return Expr(expr.head, new_args...)
end
_convert_constants(expr, ::Any) = expr
_convert_constants(expr::Real, T) = T(expr)
_convert_constants(expr::Integer, ::Any) = expr

function _eval_expressions!(output, expressions::AbstractVector, parameter_arrays, x_batch::AbstractMatrix, number_type, backend::KA.CPU; shared_parameters=false)
    n_exprs = length(expressions)
    iszero(n_exprs) && (fill!(output, zero(eltype(x_batch))); return)

    ero = eachrow(output)
    if shared_parameters
        @inbounds for i in 1:n_exprs
            _maybe_eval_batch(ero[i], expressions[i], x_batch, parameter_arrays[i], number_type, backend)
        end
    else
        @inbounds for i in 1:n_exprs
            _maybe_eval_batch(ero[i], expressions[i], x_batch, parameter_arrays[i], number_type, backend)
        end
    end
end
function _eval_expressions!(output, expressions::AbstractVector, parameter_arrays, x_batch::AbstractMatrix, number_type, backend; shared_parameters=false)
    n_exprs = length(expressions)
    # iszero(n_exprs) && (fill!(output, zero(eltype(x_batch))); return)
    iszero(n_exprs) && return
    
    for j in 1:n_exprs
        _maybe_eval_batch(@view(output[j, :]), expressions[j], x_batch, shared_parameters ? parameter_arrays : parameter_arrays[j], number_type, backend)
    end
end

function _eval_expressions!(output, expression::CachingExpr, parameter_array, x_batch::AbstractMatrix, number_type, backend; shared_parameters=false)
    _maybe_eval_batch(output, expression, x_batch, parameter_array, number_type, backend)
end

function _eval_expressions!(output, ::Nothing, ::Any, x_batch::AbstractMatrix, ::Any, ::Any; shared_parameters=false)
    fill!(output, zero(eltype(x_batch)))
end

function _eval_objective_batch!(output::AbstractVector, expressions::EvaluatorExpressions, x_batch::AbstractMatrix, number_type, backend)
    _eval_expressions!(output, expressions.objective, expressions.objective_p, x_batch, number_type, backend)
end

function _eval_objective_gradient_batch!(output::AbstractMatrix, expressions::EvaluatorExpressions, x_batch::AbstractMatrix, number_type, backend)
    _eval_expressions!(output, expressions.objective_gradient, expressions.objective_gradient_p, x_batch, number_type, backend, shared_parameters=true)
end

function _eval_constraints_batch!(output::AbstractMatrix, expressions::EvaluatorExpressions, x_batch::AbstractMatrix, number_type, backend)
    _eval_expressions!(output, expressions.constraints, expressions.constraint_p, x_batch, number_type, backend)
end

function _eval_constraint_jacobian_batch!(output::AbstractMatrix, expressions::EvaluatorExpressions, x_batch::AbstractMatrix, number_type, backend)
    @inbounds for k in 1:length(expressions.jacobian_structure)
        _maybe_eval_batch(@view(output[k, :]), expressions.constraint_jacobian[k], x_batch, expressions.constraint_jacobian_p[k], number_type, backend)
    end
end

function _eval_hessian_lagrangian_batch!(
    output::AbstractMatrix,
    temp_hessian_batch::AbstractMatrix,
    expressions::EvaluatorExpressions,
    x_batch::AbstractMatrix,
    σ_batch::AbstractVector,
    μ_batch::AbstractMatrix,
    number_type,
    backend
)
    _eval_hessian_terms_batch!(temp_hessian_batch, expressions, x_batch, number_type, backend)
    _apply_hessian_scaling_batch!(output, temp_hessian_batch, expressions, σ_batch, μ_batch, backend)
end

function _eval_hessian_terms_batch!(
    output::AbstractMatrix,
    expressions::EvaluatorExpressions,
    x_batch::AbstractMatrix,
    number_type,
    backend::KA.CPU
)
    ero = eachrow(output)
    @inbounds for i in 1:length(expressions.hessian_lagrangian)
        _maybe_eval_batch(ero[i], expressions.hessian_lagrangian[i], x_batch, expressions.hessian_lagrangian_p[i], number_type, backend)
    end
end
function _eval_hessian_terms_batch!(
    output::AbstractMatrix,
    expressions::EvaluatorExpressions,
    x_batch::AbstractMatrix,
    number_type,
    backend
)
    for k in 1:length(expressions.hessian_lagrangian)
        _maybe_eval_batch(@view(output[k, :]), expressions.hessian_lagrangian[k], x_batch, expressions.hessian_lagrangian_p[k], number_type, backend)
    end
end

function _apply_hessian_scaling_batch!(
    output::AbstractMatrix,
    hessian_terms::AbstractMatrix,
    expressions::EvaluatorExpressions,
    σ_batch::AbstractVector,
    μ_batch::AbstractMatrix,
    ::KA.CPU
)
    fill!(output, zero(eltype(output)))
    
    # Process objective hessian terms first
    obj_offset = 0
    if !isnothing(expressions.objective)
        obj_hess_count = length(expressions.objective_H_structure)
        @inbounds for k in 1:obj_hess_count
            for i in axes(output, 2)
                output[k, i] = σ_batch[i] * hessian_terms[k, i]
            end
        end
        obj_offset = obj_hess_count
    end
    
    # Process constraint hessian terms
    constraint_offset = obj_offset
    for constraint_idx in 1:length(expressions.constraints)
        constraint_hess_count = length(expressions.constraint_H_structures[constraint_idx])
        @inbounds for k in 1:constraint_hess_count
            global_idx = constraint_offset + k
            for i in axes(output, 2)
                output[global_idx, i] = μ_batch[constraint_idx, i] * hessian_terms[global_idx, i]
            end
        end
        constraint_offset += constraint_hess_count
    end
end
function _apply_hessian_scaling_batch!(
    output::AbstractMatrix,
    hessian_terms::AbstractMatrix,
    expressions::EvaluatorExpressions,
    σ_batch::AbstractVector,
    μ_batch::AbstractMatrix,
    backend
)
    fill!(output, zero(eltype(output)))
    
    # Process objective hessian terms first
    obj_offset = 0
    if !isnothing(expressions.objective)
        obj_hess_count = length(expressions.objective_H_structure)
        for k in 1:obj_hess_count
            output[k, :] = σ_batch .* hessian_terms[k, :]
        end
        obj_offset = obj_hess_count
    end
    
    # Process constraint hessian terms
    constraint_offset = obj_offset
    for constraint_idx in 1:length(expressions.constraints)
        constraint_hess_count = length(expressions.constraint_H_structures[constraint_idx])
        for k in 1:constraint_hess_count
            global_idx = constraint_offset + k
            output[global_idx, :] = μ_batch[constraint_idx, :] .* hessian_terms[global_idx, :]
        end
        constraint_offset += constraint_hess_count
    end
end
