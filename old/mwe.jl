using JuMP, Ipopt, LinearAlgebra, Random

function create_markowitz_problem(n_assets::Int; risk_aversion::Float64=1.0, seed::Int=42)
    Random.seed!(seed)  # For reproducible results
    
    # Generate realistic market data
    μ = 0.05 .+ 0.15 * rand(n_assets)  # Expected returns between 5% and 20%
    
    # Generate positive definite covariance matrix
    A = randn(n_assets, n_assets)
    Σ = (A' * A) / n_assets + 0.01 * I  # Scaled covariance matrix
    
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    
    @variable(m, 0.0 <= w[1:n_assets] <= 1.0)  # Portfolio weights with bounds
    
    # Budget constraint: weights sum to 1
    @constraint(m, budget, sum(w) == 1)
    
    # Add some nonlinear constraints for complexity (transaction costs, etc.)
    # Nonlinear transaction cost constraint
    @constraint(m, transaction_cost, sum(w[i]^1.5 for i in 1:n_assets) <= 2.0)
    
    # Risk budget constraint (nonlinear)
    @constraint(m, risk_budget, sum(w[i] * exp(0.1 * w[i]) for i in 1:n_assets) <= n_assets * 0.5)
    
    # Nonlinear objective: maximize utility with nonlinear risk penalty
    # Utility = expected return - risk_aversion * (quadratic risk + nonlinear penalty)
    portfolio_return = sum(μ[i] * w[i] for i in 1:n_assets)
    quadratic_risk = sum(w[i] * Σ[i,j] * w[j] for i in 1:n_assets, j in 1:n_assets)
    nonlinear_penalty = sum(w[i]^2 * log(1 + w[i]) for i in 1:n_assets)
    
    @objective(m, Max, portfolio_return - risk_aversion * (quadratic_risk + 0.1 * nonlinear_penalty))
    
    return m, μ, Σ
end
function create_nlp_model(model::JuMP.Model)
    rows = Vector{ConstraintRef}(undef, 0)
    nlp = MOI.Nonlinear.Model()
    for (F, S) in list_of_constraint_types(model)
        if F <: VariableRef && (S <: MOI.Parameter{Float64})
            continue  # Skip variable bounds
        end
        for ci in all_constraints(model, F, S)
            push!(rows, ci)
            object = constraint_object(ci)
            MOI.Nonlinear.add_constraint(nlp, object.func, object.set)
        end
    end
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    return nlp, rows
end

function create_evaluator(model::JuMP.Model; x=all_variables(model), mode=:symbolic)
    nlp, rows = create_nlp_model(model)
    backend = if mode == :reverse
        MOI.Nonlinear.SparseReverseMode()
    elseif mode == :symbolic
        MOI.Nonlinear.SymbolicMode()
    else
        error("Unknown evaluator backend mode")
    end
    evaluator = MOI.Nonlinear.Evaluator(nlp, backend, vcat(index.(x)...))
    MOI.initialize(evaluator, [:Grad, :Jac, :Hess])
    return evaluator, rows, nlp
end
# Create the problem with configurable size
n_assets = 10  # Change this to adjust problem size
m, μ, Σ = create_markowitz_problem(n_assets, risk_aversion=2.0)

optimize!(m)
evaluator = first(create_evaluator(m))

@info "Created Markowitz problem with $n_assets assets"
@info "Optimal portfolio weights: $(value.(m[:w]))"
@info "Expected return: $(sum(μ[i] * value(m[:w][i]) for i in 1:n_assets))"
@info "Portfolio risk: $(sqrt(sum(value(m[:w][i]) * Σ[i,j] * value(m[:w][j]) for i in 1:n_assets, j in 1:n_assets)))"

include("eval_expr.jl")

# cpu broadcasting batch
exprs = build_typed_expressions(
    evaluator.backend,
)
vals = evaluate_expressions_batch(
    exprs,
    Matrix{Float64}([3.0 10.0; 1.0 2.0;])',
    ones(Float64, 2),
    zeros(Float64, 2, 2)
)

og = zeros(2); MOI.eval_objective_gradient(evaluator, og, [3.0, 10.0])
@assert all(isapprox.(vals.objective_gradient[:, 1],og))

og = zeros(2); MOI.eval_objective_gradient(evaluator, og, [1.0, 2.0])
@assert all(isapprox.(vals.objective_gradient[:, 2],og))

cg = zeros(length(evaluator.backend.constraints)); MOI.eval_constraint(evaluator, cg, [3.0, 10.0])
@assert all(isapprox.(vals.constraints[:, 1],cg))

cg = zeros(length(evaluator.backend.constraints)); MOI.eval_constraint(evaluator, cg, [1.0, 2.0])
@assert all(isapprox.(vals.constraints[:, 2],cg))

J = zeros(length(MOI.jacobian_structure(evaluator))); MOI.eval_constraint_jacobian(evaluator, J, [3.0, 10.0])
@assert all(isapprox.(vals.constraint_jacobian[:, 1],J))

J = zeros(length(MOI.jacobian_structure(evaluator))); MOI.eval_constraint_jacobian(evaluator, J, [1.0, 2.0])
@assert all(isapprox.(vals.constraint_jacobian[:, 2],J))

H = zeros(length(MOI.hessian_lagrangian_structure(evaluator))); MOI.eval_hessian_lagrangian(evaluator, H, [3.0, 10.0], 1.0, zeros(length(evaluator.backend.constraints)))
@assert all(isapprox.(vals.hessian_lagrangian[:, 1],H))

H = zeros(length(MOI.hessian_lagrangian_structure(evaluator))); MOI.eval_hessian_lagrangian(evaluator, H, [1.0, 2.0], 1.0, zeros(length(evaluator.backend.constraints)))
@assert all(isapprox.(vals.hessian_lagrangian[:, 2],H))

@info "CPU pass"

using Metal
# metal kernel batch
exprs = build_typed_expressions(
    evaluator.backend,
    array_type=MtlVector,number_type=Float32
)
x = MtlMatrix{Float32}([3.0f0 8.0f0; 2.0f0 -1.0f0])'
σ = Metal.rand(2)
μ = Metal.rand(2, 2)

xcpu_1 = [3.0, 8.0]
xcpu_2 = [2.0, -1.0]
σcpu = Vector{Float64}(σ)
σcpu_1 = σcpu[1]
σcpu_2 = σcpu[2]
μcpu = Matrix{Float64}(μ)
μcpu_1 = μcpu[:, 1]
μcpu_2 = μcpu[:, 2]
vals = evaluate_expressions_batch(
    exprs,
    x,
    σ,
    μ,
    backend=MetalBackend()
)

og = zeros(2); MOI.eval_objective_gradient(evaluator, og, xcpu_1)
@assert all(isapprox.(convert(Vector{Float32},vals.objective_gradient[:, 1]),convert(Vector{Float32},og)))

og = zeros(2); MOI.eval_objective_gradient(evaluator, og, xcpu_2)
@assert all(isapprox.(convert(Vector{Float32},vals.objective_gradient[:, 2]),convert(Vector{Float32},og)))

cg = zeros(length(evaluator.backend.constraints)); MOI.eval_constraint(evaluator, cg, xcpu_1)
@assert all(isapprox.(convert(Vector{Float32},vals.constraints[:, 1]),convert(Vector{Float32},cg)))

cg = zeros(length(evaluator.backend.constraints)); MOI.eval_constraint(evaluator, cg, xcpu_2)
@assert all(isapprox.(convert(Vector{Float32},vals.constraints[:, 2]),convert(Vector{Float32},cg)))

J = zeros(length(MOI.jacobian_structure(evaluator))); MOI.eval_constraint_jacobian(evaluator, J, xcpu_1)
@assert all(isapprox.(convert(Vector{Float32},vals.constraint_jacobian[:, 1]),convert(Vector{Float32},J)))

J = zeros(length(MOI.jacobian_structure(evaluator))); MOI.eval_constraint_jacobian(evaluator, J, xcpu_2)
@assert all(isapprox.(convert(Vector{Float32},vals.constraint_jacobian[:, 2]),convert(Vector{Float32},J)))

H = zeros(length(MOI.hessian_lagrangian_structure(evaluator))); MOI.eval_hessian_lagrangian(evaluator, H, xcpu_1, σcpu_1, μcpu_1)
@assert all(isapprox.(convert(Vector{Float32},vals.hessian_lagrangian[:, 1]),convert(Vector{Float32},H)))

H = zeros(length(MOI.hessian_lagrangian_structure(evaluator))); MOI.eval_hessian_lagrangian(evaluator, H, xcpu_2, σcpu_2, μcpu_2)
@assert all(isapprox.(convert(Vector{Float32},vals.hessian_lagrangian[:, 2]),convert(Vector{Float32},H)))

@info "Metal pass"
println("\n\n")

using BenchmarkTools

batch_size = 128
n_var = length(evaluator.backend.x)
n_cons = length(evaluator.backend.constraints)

@info "Batch size: $batch_size\nNumber of variables: $n_var\nNumber of constraints: $n_cons"
println("\n")

x_batch = rand(Float64, n_var, batch_size)
σ_batch = rand(Float64, batch_size)
μ_batch = rand(Float64, n_cons, batch_size)

exprs_cpu = build_typed_expressions(evaluator.backend)
exev = ExprEvaluator(exprs_cpu, n_var, n_cons, backend=CPU(), batch_size=batch_size)
cpu_backend = CPU()
b = @benchmark begin
    evaluate_expressions_batch(
        $exprs_cpu,
        $x_batch,
        $σ_batch,
        $μ_batch,
        backend=$cpu_backend,
        expr_evaluator=$exev
    )
end samples=100

@info "CPU"
show(stdout, MIME("text/plain"), b)
println("\n\n")

backend = MetalBackend()
x_batch_metal = MtlMatrix{Float32}(x_batch)
σ_batch_metal = MtlVector{Float32}(σ_batch)
μ_batch_metal = MtlMatrix{Float32}(μ_batch)

metal_exprs = build_typed_expressions(evaluator.backend, array_type=MtlVector, number_type=Float32)
exev = ExprEvaluator(metal_exprs, n_var, n_cons, backend=backend, batch_size=batch_size)
bm = @benchmark begin
    evaluate_expressions_batch(
        $metal_exprs,
        $x_batch_metal,
        $σ_batch_metal,
        $μ_batch_metal,
        backend=$backend,
        expr_evaluator=$exev
    ); KernelAbstractions.synchronize($backend)
end samples=10

@info "Metal"
show(stdout, MIME("text/plain"), bm)
println("\n\n")

grad = zeros(n_var, 1, batch_size)
bmoi = @benchmark begin
    for j in 1:$batch_size
        MOI.eval_objective_gradient($evaluator, @view($grad[:, :, j]), $x_batch[:, j])
    end
end samples=100

@info "MOI Loop"
show(stdout, MIME("text/plain"), bmoi)
println("\n\n")

grad = zeros(n_var, 1, batch_size)
bmoib = @benchmark begin
    MOI.eval_objective_gradient.(Ref($evaluator), eachslice($grad, dims=3), eachslice($x_batch, dims=2))
end samples=100

@info "MOI Broadcast"
show(stdout, MIME("text/plain"), bmoib)
println("\n\n")

@info "Speedup over broadcasted MOI: $(median(bmoib).time / median(b).time)"