using JuMP, Ipopt
m = Model(Ipopt.Optimizer)
set_silent(m)
@variable m x
@variable m y
@constraint m c1 2x + sin(y) == 3
@constraint m c2 3*x*y^4 + cos(x) == 0
@objective m Max x^3 + sin(x*y)
optimize!(m)
evaluator = MOI.Nonlinear.Evaluator(unsafe_backend(m).nlp_model, MOI.Nonlinear.SymbolicAD.Evaluator(unsafe_backend(m).nlp_model, index.(all_variables(m))))
MOI.initialize(evaluator, [:Hess])

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





using BenchmarkTools


batch_size = 640
# @info "Running timing test with batch size: $batch_size"

# # Generate random batch data
x_batch = rand(Float64, 2, batch_size)
σ_batch = rand(Float64, batch_size)
μ_batch = rand(Float64, length(evaluator.backend.constraints), batch_size)

# # CPU timing
# @info "CPU timing..."
exprs_cpu = build_typed_expressions(
    evaluator.backend
)
exev = ExprEvaluator(exprs_cpu, 2, 2, backend=CPU(), batch_size=batch_size)
# for i in 1:10
#     evaluate_expressions_batch(
#         exprs_cpu,
#         x_batch,
#         σ_batch,
#         μ_batch,
#         backend=CPU(),
#         expr_evaluator=exev
#     )
# end
b = @benchmark begin
    evaluate_expressions_batch(
        exprs_cpu,
        x_batch,
        σ_batch,
        μ_batch,
        backend=CPU(),
        expr_evaluator=exev
    )
end samples=10

@info "Metal timing..."
metal_exprs = build_typed_expressions(
    evaluator.backend,
    array_type=MtlVector,
    number_type=Float32
)
backend = MetalBackend()
x_batch_metal = MtlMatrix{Float32}(x_batch)
σ_batch_metal = MtlVector{Float32}(σ_batch)
μ_batch_metal = MtlMatrix{Float32}(μ_batch)

exev = ExprEvaluator(metal_exprs, 2, 2, backend=backend, batch_size=batch_size)

# Warm up
for i in 1:10
    evaluate_expressions_batch(
        metal_exprs,
        x_batch_metal,
        σ_batch_metal,
        μ_batch_metal,
        backend=backend,
        expr_evaluator=exev
    )
end

bm = @benchmark begin
    evaluate_expressions_batch(
        metal_exprs,
        x_batch_metal,
        σ_batch_metal,
        μ_batch_metal,
        backend=backend,
        expr_evaluator=exev
    ); KernelAbstractions.synchronize(backend)
end samples=10 

grad = zeros(2, 1, batch_size)
bmoi = @benchmark begin
    for j in 1:batch_size
        MOI.eval_objective_gradient(evaluator, @view(grad[:, :, j]), x_batch[:, j])
    end
end samples=10

bmoib = @benchmark begin
    MOI.eval_objective_gradient.(Ref(evaluator), eachslice(grad, dims=3), eachslice(x_batch, dims=2))
end samples=10
