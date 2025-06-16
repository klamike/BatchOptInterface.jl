using JuMP, Ipopt, Random, LinearAlgebra
const MOIN = MOI.Nonlinear
const MOINS = MOIN.SymbolicAD

seed = 42
n_assets = 10
risk_aversion = 2.0

function create_nlp_model(model::JuMP.Model)
    rows = Vector{ConstraintRef}(undef, 0)
    nlp = MOI.Nonlinear.Model()
    for ci in all_constraints(model, include_variable_in_set_constraints=true)
        co = constraint_object(ci)
        if !(co.func isa VariableRef)
            push!(rows, ci)
            MOI.Nonlinear.add_constraint(nlp, co.func, co.set)
        end
    end
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    return nlp, rows
end
function create_evaluator(model::JuMP.Model; x=all_variables(model), mode=MOIN.ExprGraphOnly())
    nlp, rows = create_nlp_model(model)
    evaluator = MOI.Nonlinear.Evaluator(nlp, mode, index.(x))
    MOI.initialize(evaluator, [:ExprGraph])
    return evaluator, rows, nlp
end

rng = Random.MersenneTwister(seed)
μ = 0.05 .+ 0.15 * rand(rng, n_assets)
A = randn(rng, n_assets, n_assets)
Σ = (A' * A) / n_assets + 0.01 * I
m = Model(Ipopt.Optimizer);set_silent(m)
@variable(m, 0.0 <= w[1:n_assets] <= 1.0)
@constraint(m, budget, sum(w) + sin(w[1]*w[3]) == 1)
@constraint(m, transaction_cost, sum(w[i]^1.5 for i in 1:n_assets) - cos(w[2]*w[4]) <= 2.0)
@constraint(m, risk_budget, sum(w[i] * exp(0.1 * w[i]) for i in 1:n_assets) <= n_assets * 0.5)
portfolio_return = sum(μ[i] * w[i] for i in 1:n_assets)
quadratic_risk = sum(w[i] * Σ[i,j] * w[j] for i in 1:n_assets, j in 1:n_assets)
nonlinear_penalty = sum(w[i]^2 * log(1 + w[i]) for i in 1:n_assets)
@objective(m, Max, tan(w[4]*w[5]^2) *portfolio_return - risk_aversion * (quadratic_risk + 0.1 * nonlinear_penalty))

evaluator, rows, nlp = create_evaluator(m, mode=MOIN.SymbolicMode())

obj = objective_function(m)
x = filter(!is_parameter, all_variables(m))

gradient_and_hessian(co::ConstraintRef) = gradient_and_hessian(constraint_object(co).func)
gradient_and_hessian(co::AbstractJuMPScalar) = gradient_and_hessian(moi_function(co))
gradient_and_hessian(co::MOI.AbstractScalarFunction) = MOINS.gradient_and_hessian(vi -> !is_parameter(VariableRef(m, vi)), co)[2:end]
_drop_empty(vec) = eltype(vec) isa AbstractArray ? filter(!isempty, vec) : vec
batch_gradient_and_hessian(cos) = _drop_empty(zip(gradient_and_hessian.(cos)...))
moifn_o∇, oH, moifn_o∇² = gradient_and_hessian(obj)
moifn_c∇s, cH, moifn_c∇²s = batch_gradient_and_hessian(rows)

batch_parse_expression(expr) = MOIN.parse_expression.(Ref(nlp), expr)
moiex_o∇ = batch_parse_expression(moifn_o∇)
moiex_o∇² = batch_parse_expression(moifn_o∇²)
moiex_c∇s = batch_parse_expression.(moifn_c∇s)
moiex_c∇²s = batch_parse_expression.(moifn_c∇²s)

T = Float64
convert_to_expr(expr) = MOIN.convert_to_expr.(Ref(evaluator), expr, moi_output_format=true)
ex_o∇ = convert_to_expr(moiex_o∇)
ex_o∇² = convert_to_expr(moiex_o∇²)
ex_c∇s = convert_to_expr.(moiex_c∇s) 
ex_c∇²s = convert_to_expr.(moiex_c∇²s)

convert_constants(expr) = expr
convert_constants(expr::Real) = T(expr)
convert_constants(expr::Integer) = expr
convert_constants(expr::Expr) = Expr(expr.head, convert_constants.(expr.args)...)
convert_constants(vex::Vector{Expr}) = convert_constants.(vex)
ex_o∇ = convert_constants(ex_o∇)
ex_o∇² = convert_constants(ex_o∇²)
ex_c∇s = convert_constants.(ex_c∇s)
ex_c∇²s = convert_constants.(ex_c∇²s)

to_int_index(vex::Vector{Expr}) = _to_int_index.(vex)
_to_int_index(expr) = expr
_to_int_index(expr::Expr) = begin
    if expr.head == :ref && length(expr.args) == 2
        var_name, index = expr.args
        isa(index, MOI.VariableIndex) && return Expr(:ref, var_name, index.value)
    end
    return Expr(expr.head, _to_int_index.(expr.args)...)
end
ex_o∇ = to_int_index(ex_o∇)
ex_o∇² = to_int_index(ex_o∇²)
ex_c∇s = to_int_index.(ex_c∇s)
ex_c∇²s = to_int_index.(ex_c∇²s)


# CPU: RGF + @generated loops
#       (I think) this lets Julia compile the batch version of each function we generated
# TODO: look into LoopVectorization.jl
using RuntimeGeneratedFunctions, StaticArrays
RuntimeGeneratedFunctions.init(@__MODULE__)
make_function(expr::Expr) = @RuntimeGeneratedFunction(:(x -> @inbounds begin $expr end))
make_function(vex::Vector{Expr}) = make_function.(vex)
fn_o∇ = make_function(ex_o∇)
fn_o∇² = make_function(ex_o∇²)
fn_c∇s = make_function.(ex_c∇s)
fn_c∇²s = make_function.(ex_c∇²s)

@generated function _eval_function!(output::MMatrix{1,B,T}, func::F, X::SMatrix{N,B,T}) where {F,N,B,T}
    quote
        @assert size(output, 2) == size(X, 2)
        @inbounds for (o, x) in zip(eachcol(output), eachcol(X))
            o[] = func(x)
        end
        return nothing
    end
end
function _eval_function!(output, func, X)
    N, B = size(X)
    T = eltype(X)
    return _eval_function!(output, func, SMatrix{N,B,T}(X))
end
function eval_function!(output::MMatrix{1,B,T}, func::F, X::AbstractMatrix) where {F,B,T}
    N = size(X, 1)
    Xs = SMatrix{N,B,T}(X)
    _eval_function!(output, func, Xs)
    return nothing
end

@generated function _eval_function(func::RuntimeGeneratedFunction, X::SMatrix{N,B,T}) where {N,B,T}
    quote
        output = MMatrix{1,B,T}(zeros(T,1,B)) # FIXME: conversion API?
        @inbounds for (o, x) in zip(eachcol(output), eachcol(X))
            o[] = func(x)
        end
        return output
    end
end
function eval_function(func::F, X::AbstractMatrix) where F
    N,B = size(X)
    T = eltype(X)
    return _eval_function(func, SMatrix{N,B,T}(X))
end

batch_size = 64
w = rand(rng, T, n_assets, batch_size)

@time eval_function(fn_o∇[1], w)
@time eval_function(fn_o∇[1], w)
@time eval_function(fn_o∇[1], w)


all_rgf = RuntimeGeneratedFunction[]
for func in fn_o∇
    push!(all_rgf, func)
end
for func in fn_o∇²
    push!(all_rgf, func)
end
for c in fn_c∇s
    for func in c
        push!(all_rgf, func)
    end
end
for c in fn_c∇²s
    for func in c
        push!(all_rgf, func)
    end
end

@time eval_function.(all_rgf, Ref(w));


using DynamicExpressions
const DE = DynamicExpressions
_DE_OPERATORS = OperatorEnum(
    ( +, -, *, ^, /, ifelse, atan, min, max ), # binary operators
    ( +, -, abs, sign, sqrt, cbrt, abs2, inv, log, log10, log2, log1p, exp, exp2, expm1, sin, cos, tan, sec, csc, cot, sind, cosd, tand, secd, cscd, cotd, asin, acos, atan, asec ) # unary operators
)
names(X::AbstractMatrix) = ["x$i" for i in 1:size(X, 1)]
convert_index_to_name(expr::Expr) = begin
    expr.head == :ref && return Symbol(string( expr.args[1]) * string(expr.args[2]))
    return Expr(expr.head, convert_index_to_name.(expr.args)...)
end
convert_index_to_name(expr) = expr
_eval_function(func::DE.Expression, X) = func(X)
_eval_function!(output, func::DE.Expression, X) = begin
    output .= func(X)
    return nothing
end
to_de(expr) = DE.combine_operators(DE.parse_expression(convert_index_to_name(expr); operators=_DE_OPERATORS, variable_names=names(w)))
to_de(vex::Vector{Expr}) = to_de.(vex)
dex_o∇ = to_de(ex_o∇)
dex_o∇² = to_de(ex_o∇²)
dex_c∇s = to_de.(ex_c∇s)
dex_c∇²s = to_de.(ex_c∇²s)

all_de = DE.Expression[]
for dex in dex_o∇
    push!(all_de, dex)
end
for dex in dex_o∇²
    push!(all_de, dex)
end
for c in dex_c∇s
    for dex in c
        push!(all_de, dex)
    end
end
for c in dex_c∇²s
    for dex in c
        push!(all_de, dex)
    end
end

using BenchmarkTools
@info "MOINS"
@assert evaluator.backend isa MOINS.Evaluator
@benchmark begin
    MOINS._evaluate!.(Ref($evaluator.backend), eachcol($w))
end samples=100
println("\n\n")

@info "BOI RGF"
@benchmark begin
    eval_function.($all_rgf, Ref($w))
end samples=100
println("\n\n")

@info "BOI DE"
out = eval_function.(all_de, Ref(w));
fill!(out, zero(eltype(out)));
@benchmark begin
    _eval_function.($all_de, Ref($w))
end samples=100
println("\n\n")

include("old/eval_expr.jl")

@info "BOI old"
exprs = build_typed_expressions(evaluator.backend);
n_var = length(evaluator.backend.x);
n_cons = length(evaluator.backend.constraints);
exev = ExprEvaluator(exprs, n_var, n_cons, backend=CPU(), batch_size=batch_size);
@benchmark begin
    evaluate_expressions_batch(exprs, $w, expr_evaluator=$exev)
end samples=100

# Correctness Tests
@info "Running correctness tests..."

# Test points for validation
test_points = [
    # rand(rng, T, n_assets),
    # rand(rng, T, n_assets),
    # rand(rng, T, n_assets)
    rand(rng, T, n_assets) for _ in 1:batch_size
]

# Convert test points to batch format for BOI old
test_batch = hcat(test_points...)  # n_assets x 3 matrix

# Evaluate using BOI old
exprs = build_typed_expressions(evaluator.backend);
exev = ExprEvaluator(exprs, n_var, n_cons, backend=CPU(), batch_size=length(test_points));
vals_boi = evaluate_expressions_batch(exprs, test_batch, expr_evaluator=exev)

# Test objective gradient
@info "Testing objective gradient correctness..."
for (i, x_test) in enumerate(test_points)
    # MOI reference
    og_moi = zeros(T, n_assets)
    MOI.eval_objective_gradient(evaluator, og_moi, x_test)
    
    # BOI old implementation
    og_boi = vals_boi.objective_gradient[:, i]
    
    @assert all(isapprox.(og_boi, og_moi)) "BOI old objective gradient test $i failed"
end

# Test constraint evaluation
@info "Testing constraint evaluation correctness..."
for (i, x_test) in enumerate(test_points)
    # MOI reference
    c_moi = zeros(T, length(rows))
    MOI.eval_constraint(evaluator, c_moi, x_test)
    
    # BOI old implementation
    c_boi = vals_boi.constraints[:, i]
    
    @assert all(isapprox.(c_boi, c_moi, atol=1e-10)) "BOI old constraint evaluation test $i failed"
end

# Test constraint jacobian
@info "Testing constraint jacobian correctness..."
for (i, x_test) in enumerate(test_points)
    # MOI reference
    cg_moi = zeros(T, length(MOI.jacobian_structure(evaluator)))
    MOI.eval_constraint_jacobian(evaluator, cg_moi, x_test)
    
    # BOI old implementation
    cg_boi = vals_boi.constraint_jacobian[:, i]
    
    @assert all(isapprox.(cg_boi, cg_moi, atol=1e-10)) "BOI old constraint jacobian test $i failed"
end

# Test hessian lagrangian
@info "Testing hessian lagrangian correctness..."
σ_test = 1.0  # objective scaling
μ_test = zeros(T, length(rows))  # constraint multipliers
for (i, x_test) in enumerate(test_points)
    # MOI reference
    H_moi = zeros(T, length(MOI.hessian_lagrangian_structure(evaluator)))
    MOI.eval_hessian_lagrangian(evaluator, H_moi, x_test, σ_test, μ_test)
    
    # BOI old implementation
    H_boi = vals_boi.hessian_lagrangian[:, i]
    
    @assert all(isapprox.(H_boi, H_moi, atol=1e-10)) "BOI old hessian lagrangian test $i failed"
end

@info "All correctness tests passed! ✓"
println()
