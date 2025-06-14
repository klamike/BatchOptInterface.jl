import MathOptInterface as MOI
import MathOptInterface.Nonlinear.SymbolicAD as SymbolicAD

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using KernelAbstractions
const KA = KernelAbstractions

include("kernel_utils.jl")
include("expr_utils.jl")
include("dag_to_expr.jl")
include("evaluator_to_expr.jl")


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

struct ExprEvaluator  # stores results
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
    T = eltype(expressions.objective_p)
    return ExprEvaluator(
        expressions,
        zeros(backend, T, batch_size),
        zeros(backend, T, n_vars, batch_size),
        zeros(backend, T, n_constraints, batch_size),
        zeros(backend, T, length(expressions.jacobian_structure), batch_size),
        zeros(backend, T, length(expressions.hessian_structure), batch_size),
        zeros(backend, T, length(expressions.hessian_structure), batch_size),
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
    x_batch::AbstractMatrix;
    backend=KA.CPU(),
    number_type=eltype(x_batch),
    expr_evaluator=nothing
)
    # TODO: assert shapes are consistent
    N, B = size(x_batch)
    M = length(expressions.constraints)

    # Pre-allocate working/results memory
    _expr_evaluator = isnothing(expr_evaluator) ? ExprEvaluator(expressions, N, M, backend=backend, batch_size=B) : expr_evaluator
    objective = _expr_evaluator.objective
    objective_gradient = _expr_evaluator.objective_gradient
    constraints = _expr_evaluator.constraints
    constraint_jacobian = _expr_evaluator.constraint_jacobian

    # Evaluate all components
    _eval_objective_batch!(objective, expressions, x_batch, number_type, backend)
    _eval_objective_gradient_batch!(objective_gradient, expressions, x_batch, number_type, backend)
    _eval_constraints_batch!(constraints, expressions, x_batch, number_type, backend)
    _eval_constraint_jacobian_batch!(constraint_jacobian, expressions, x_batch, number_type, backend)

    return _expr_evaluator
end

function evaluate_expressions_batch(  # with hessian
    expressions::EvaluatorExpressions,
    x_batch::AbstractMatrix,
    σ_batch::AbstractVector,
    μ_batch::AbstractMatrix;
    backend=KA.CPU(),
    number_type=eltype(x_batch),
    expr_evaluator=nothing
)
    ret = evaluate_expressions_batch(expressions, x_batch, backend=backend, number_type=number_type, expr_evaluator=expr_evaluator)
    _eval_hessian_lagrangian_batch!(ret.hessian_lagrangian, ret.temp_hessian_batch, expressions, x_batch, σ_batch, μ_batch, number_type, backend)
    return ret
end


"""
    _maybe_eval_batch(output, expr, x_batch, p, var_map, T, BK)

Evaluate expression over batch with variable remapping.

!!! warning
    Uses `eval` and `invokelatest` in the GPU case.
"""
function _maybe_eval_batch(output, expr::CachingExpr, x_batch::AbstractMatrix, p_batch, T, ::KA.CPU)
    func = if !(expr.fn_set)
        remapped_expr = _convert_constants(expr, T)
        wrapped_expr = _wrap_expr_for_evaluation(remapped_expr)
        # TODO: above steps can happen earlier
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

function _maybe_eval_batch(output, expr::CachingExpr, x_batch::AbstractMatrix, p_batch, T, BK)  # gpu-compatible version
    @assert KA.isgpu(BK)
    
    if !(expr.fn_set)
        remapped_expr = _convert_constants(expr, T)
        kernel_expr = _wrap_expr_for_kernel(remapped_expr, gensym("_expr_kernel"))
        expr.fn = Base.eval(KernelEvaluationModule, kernel_expr)
        expr.fn_set = true
    end

    func = expr.fn

    try
        # TODO: all this invokelatest/module stuff is a mess
        kernel_instance = Base.invokelatest(func, BK, 64)
        Base.invokelatest(kernel_instance, output, x_batch, p_batch, ndrange=size(x_batch, 2))
    finally
        # no synchronization here; user is responsible
    end

    return nothing
end
_maybe_eval_batch(output, ::Nothing, x_batch::AbstractMatrix, p_batch, T, BK) = nothing

@generated function _eval_batch_cpu(output, func::F, x_batch, p_batch) where F
    quote
        @inbounds for (i, col) in enumerate(eachcol(x_batch))
            output[i] = func(col, p_batch)
        end
    end
end

function _eval_expressions!(output, exprs::AbstractVector, ps, X::AbstractMatrix, T, BK::KA.CPU; shared_parameters=false)
    n_exprs = length(exprs)
    iszero(n_exprs) && (fill!(output, zero(eltype(output))); return)

    ero = eachrow(output)
    if shared_parameters
        @inbounds for i in 1:n_exprs
            _maybe_eval_batch(ero[i], exprs[i], X, ps[i], T, BK)
        end
    else
        @inbounds for i in 1:n_exprs
            _maybe_eval_batch(ero[i], exprs[i], X, ps[i], T, BK)
        end
    end
end
function _eval_expressions!(output, exprs::AbstractVector, ps, X::AbstractMatrix, T, BK; shared_parameters=false)
    n_exprs = length(exprs)
    iszero(n_exprs) && (fill!(output, zero(eltype(output))); return)
    ero = eachrow(output)
    if shared_parameters
        @inbounds for j in 1:n_exprs
            _maybe_eval_batch(ero[j], exprs[j], X, ps, T, BK)
        end
    else
        @inbounds for j in 1:n_exprs
            _maybe_eval_batch(ero[j], exprs[j], X, ps[j], T, BK)
        end
    end
end

function _eval_expressions!(output, expr::CachingExpr, p, X::AbstractMatrix, T, BK; shared_parameters=false)
    _maybe_eval_batch(output, expr, X, p, T, BK)
end

function _eval_expressions!(output, ::Nothing, ::Any, ::Any, ::Any, ::Any; shared_parameters=false)
    fill!(output, zero(eltype(output)))
end

function _eval_objective_batch!(output::AbstractVector, evexprs::EvaluatorExpressions, X::AbstractMatrix, T, BK)
    _eval_expressions!(output, evexprs.objective, evexprs.objective_p, X, T, BK)
end

function _eval_objective_gradient_batch!(output::AbstractMatrix, evexprs::EvaluatorExpressions, X::AbstractMatrix, T, BK)
    _eval_expressions!(output, evexprs.objective_gradient, evexprs.objective_gradient_p, X, T, BK, shared_parameters=true)
end

function _eval_constraints_batch!(output::AbstractMatrix, evexprs::EvaluatorExpressions, X::AbstractMatrix, T, BK)
    _eval_expressions!(output, evexprs.constraints, evexprs.constraint_p, X, T, BK)
end

function _eval_constraint_jacobian_batch!(output::AbstractMatrix, evexprs::EvaluatorExpressions, X::AbstractMatrix, T, BK)
    @inbounds for k in 1:length(evexprs.jacobian_structure)
        _maybe_eval_batch(@view(output[k, :]), evexprs.constraint_jacobian[k], X, evexprs.constraint_jacobian_p[k], T, BK)
    end
end

function _eval_hessian_lagrangian_batch!(
    output::AbstractMatrix,
    temp_hessian_batch::AbstractMatrix,
    evexprs::EvaluatorExpressions,
    X::AbstractMatrix,
    σ::AbstractVector,
    μ::AbstractMatrix,
    T,
    BK
)
    _eval_hessian_terms_batch!(temp_hessian_batch, evexprs, X, T, BK)
    _apply_hessian_scaling_batch!(output, temp_hessian_batch, evexprs, σ, μ, BK)
end

function _eval_hessian_terms_batch!(
    output::AbstractMatrix,
    evexprs::EvaluatorExpressions,
    X::AbstractMatrix,
    T,
    BK::KA.CPU
)
    ero = eachrow(output)
    @inbounds for i in 1:length(evexprs.hessian_lagrangian)
        _maybe_eval_batch(ero[i], evexprs.hessian_lagrangian[i], X, evexprs.hessian_lagrangian_p[i], T, BK)
    end
end
function _eval_hessian_terms_batch!( # gpu-compatible version
    output::AbstractMatrix,
    evexprs::EvaluatorExpressions,
    X::AbstractMatrix,
    T,
    BK
)
    for k in 1:length(evexprs.hessian_lagrangian)
        _maybe_eval_batch(@view(output[k, :]), evexprs.hessian_lagrangian[k], X, evexprs.hessian_lagrangian_p[k], T, BK)
    end
end

function _apply_hessian_scaling_batch!(
    output::AbstractMatrix,
    hessian_terms::AbstractMatrix,
    evexprs::EvaluatorExpressions,
    σ::AbstractVector,
    μ::AbstractMatrix,
    ::KA.CPU
)
    fill!(output, zero(eltype(output)))
    
    # Process objective hessian terms first
    obj_offset = 0
    if !isnothing(evexprs.objective)
        obj_hess_count = length(evexprs.objective_H_structure)
        @inbounds for k in 1:obj_hess_count
            for i in axes(output, 2)
                output[k, i] = σ[i] * hessian_terms[k, i]
            end
        end
        obj_offset = obj_hess_count
    end
    
    # Process constraint hessian terms
    constraint_offset = obj_offset
    for constraint_idx in 1:length(evexprs.constraints)
        constraint_hess_count = length(evexprs.constraint_H_structures[constraint_idx])
        @inbounds for k in 1:constraint_hess_count
            global_idx = constraint_offset + k
            for i in axes(output, 2)
                output[global_idx, i] = μ[constraint_idx, i] * hessian_terms[global_idx, i]
            end
        end
        constraint_offset += constraint_hess_count
    end
end
function _apply_hessian_scaling_batch!( # gpu-compatible version
    output::AbstractMatrix,
    hessian_terms::AbstractMatrix,
    evexprs::EvaluatorExpressions,
    σ::AbstractVector,
    μ::AbstractMatrix,
    BK
)
    fill!(output, zero(eltype(output)))
    
    # Process objective hessian terms first
    obj_offset = 0
    if !isnothing(evexprs.objective)
        obj_hess_count = length(evexprs.objective_H_structure)
        for k in 1:obj_hess_count
            output[k, :] = σ .* hessian_terms[k, :]
        end
        obj_offset = obj_hess_count
    end
    
    # Process constraint hessian terms
    constraint_offset = obj_offset
    for constraint_idx in 1:length(evexprs.constraints)
        constraint_hess_count = length(evexprs.constraint_H_structures[constraint_idx])
        for k in 1:constraint_hess_count
            global_idx = constraint_offset + k
            output[global_idx, :] = μ[constraint_idx, :] .* hessian_terms[global_idx, :]
        end
        constraint_offset += constraint_hess_count
    end
end
