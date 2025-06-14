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

evaluator, rows, nlp = create_evaluator(m)

obj = objective_function(m)
x = filter(!is_parameter, all_variables(m))

gradient_and_hessian(co::ConstraintRef) = gradient_and_hessian(constraint_object(co).func)
gradient_and_hessian(co::AbstractJuMPScalar) = gradient_and_hessian(moi_function(co))
gradient_and_hessian(co::MOI.AbstractScalarFunction) = MOINS.gradient_and_hessian(vi -> !is_parameter(VariableRef(m, vi)), co)
batch_gradient_and_hessian(cos) = zip(gradient_and_hessian.(cos)...)
_, moifn_o∇, oH, moifn_o∇² = gradient_and_hessian(obj)
_, moifn_c∇s, cH, moifn_c∇²s = batch_gradient_and_hessian(rows)

_drop_empty(vec) = filter(!isempty, vec)
moifn_c∇s = _drop_empty(moifn_c∇s)
cH = _drop_empty(cH)
moifn_c∇²s = _drop_empty(moifn_c∇²s)

batch_parse_expression(expr) = MOIN.parse_expression.(Ref(nlp), expr)
moiex_o∇ = batch_parse_expression(moifn_o∇)
moiex_o∇² = batch_parse_expression(moifn_o∇²)
moiex_c∇s = batch_parse_expression.(moifn_c∇s)
moiex_c∇²s = batch_parse_expression.(moifn_c∇²s)

T = Float32
convert_to_expr(expr) = MOIN.convert_to_expr.(Ref(evaluator), expr, moi_output_format=true)
ex_o∇ = convert_to_expr(moiex_o∇)
ex_o∇² = convert_to_expr(moiex_o∇²)
ex_c∇s = convert_to_expr.(moiex_c∇s) 
ex_c∇²s = convert_to_expr.(moiex_c∇²s)

convert_constants(expr::Expr, T) = Expr(expr.head, convert_constants.(expr.args, T)...)
convert_constants(expr, ::Any) = expr
convert_constants(expr::Real, T) = T(expr)
convert_constants(expr::Integer, ::Any) = expr
convert_constants(vex::Vector{Expr}, T) = convert_constants.(vex, T)
ex_o∇ = convert_constants(ex_o∇, T)
ex_o∇² = convert_constants(ex_o∇², T)
ex_c∇s = convert_constants.(ex_c∇s, T)
ex_c∇²s = convert_constants.(ex_c∇²s, T)

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
