struct BatchTerminationStatus <: MOI.AbstractModelAttribute
    result_index::Int
    BatchTerminationStatus(result_index::Int = 1) = new(result_index)
end

MOI.attribute_value_type(::BatchTerminationStatus) = MOI.TerminationStatusCode
MOI.is_set_by_optimize(::BatchTerminationStatus) = true

struct CopyScenarioSolver end
# TODO: rename to IndividualScenarioSolver since it creates an individual new solver per scenario
# TODO: make a ParametricScenarioSolver that uses parameters like usual (no POI)

mutable struct Optimizer{T,F,S} <: MOI.AbstractOptimizer
    optimizer_constructor::F
    scenario_solver::S
    cache::MOIU.UniversalFallback{MOIU.Model{T}}
    scenarios::Vector{Any}
    with_bridge_type::Union{Nothing,Type}
    with_cache_type::Union{Nothing,Type}
    evaluate_duals::Bool
    save_original_objective_and_constraints::Bool
end

function Optimizer{T}(
    optimizer_constructor;
    scenario_solver = CopyScenarioSolver(),
    with_bridge_type = nothing,
    with_cache_type = nothing,
    evaluate_duals::Bool = true,
    save_original_objective_and_constraints::Bool = true,
) where {T}
    cache = MOIU.UniversalFallback(MOIU.Model{T}())
    return Optimizer{
        T,
        typeof(optimizer_constructor),
        typeof(scenario_solver),
    }(
        optimizer_constructor,
        scenario_solver,
        cache,
        Any[],
        with_bridge_type,
        with_cache_type,
        evaluate_duals,
        save_original_objective_and_constraints,
    )
end

function Optimizer(optimizer_constructor; kwargs...)
    return Optimizer{Float64}(optimizer_constructor; kwargs...)
end

function _new_inner_optimizer(model::Optimizer{T}) where {T}
    with_cache_type = model.with_cache_type
    if model.with_bridge_type === nothing && with_cache_type === nothing
        with_cache_type = T
    end
    return MOI.instantiate(
        model.optimizer_constructor;
        with_bridge_type = model.with_bridge_type,
        with_cache_type,
    )
end

_invalidate!(model::Optimizer) = (empty!(model.scenarios); nothing)

MOI.supports_incremental_interface(::Optimizer) = true
MOI.is_empty(model::Optimizer) = MOI.is_empty(model.cache)

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.cache)
    empty!(model.scenarios)
    return
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOIU.default_copy_to(dest, src)
end

function MOI.add_variable(model::Optimizer)
    _invalidate!(model)
    return MOI.add_variable(model.cache)
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet,
)
    _invalidate!(model)
    return MOI.add_constraint(model.cache, func, set)
end

function MOI.delete(model::Optimizer, index::MOI.Index)
    _invalidate!(model)
    MOI.delete(model.cache, index)
    return
end

function MOI.modify(
    model::Optimizer,
    ci::MOI.ConstraintIndex,
    change::MOI.AbstractFunctionModification,
)
    _invalidate!(model)
    MOI.modify(model.cache, ci, change)
    return
end

function MOI.set(
    model::Optimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
    value,
)
    _invalidate!(model)
    MOI.set(model.cache, attr, value)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    _invalidate!(model)
    MOI.set(model.cache, attr, vi, value)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value,
)
    _invalidate!(model)
    MOI.set(model.cache, attr, ci, value)
    return
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "BatchOptInterface"
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
)
    return MOI.get(model.cache, attr)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
)
    return MOI.get(model.cache, attr, vi)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model.cache, attr, ci)
end

function MOI.supports(
    model::Optimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
)
    return MOI.supports(model.cache, attr)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(model.cache, attr, MOI.VariableIndex)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    index_type::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.cache, attr, index_type)
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.cache, F, S)
end

MOI.is_valid(model::Optimizer, index::MOI.Index) = MOI.is_valid(model.cache, index)

mutable struct _ScenarioOptimizer{OT} <: MOI.ModelLike
    optimizer::OT
    scenario::Int
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}
end

function _ScenarioOptimizer(optimizer, scenario::Int)
    return _ScenarioOptimizer(
        optimizer,
        scenario,
        Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}(),
    )
end

struct _OptimizerScenarioResult
    optimizer::MOI.ModelLike
    index_map::MOIU.IndexMap
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}
end

struct _SnapshotScenarioResult
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    objective_value::Any
    dual_objective_value::Any
    variable_primal::Dict{MOI.VariableIndex,Any}
    constraint_primal::Dict{MOI.ConstraintIndex,Any}
    constraint_dual::Dict{MOI.ConstraintIndex,Any}
end

function _inner_constraint_index(
    result::_OptimizerScenarioResult,
    ci::MOI.ConstraintIndex,
)
    mapped_ci = result.index_map[ci]
    return get(result.constraint_map, mapped_ci, mapped_ci)
end

MOI.supports_incremental_interface(model::_ScenarioOptimizer) =
    MOI.supports_incremental_interface(model.optimizer)
MOI.is_empty(model::_ScenarioOptimizer) = MOI.is_empty(model.optimizer)
MOI.empty!(model::_ScenarioOptimizer) = MOI.empty!(model.optimizer)

function MOI.copy_to(dest::_ScenarioOptimizer, src::MOI.ModelLike)
    return MOIU.default_copy_to(dest, src)
end

function MOI.add_variable(model::_ScenarioOptimizer)
    return MOI.add_variable(model.optimizer)
end

function MOI.add_variables(model::_ScenarioOptimizer, n::Int)
    return MOI.add_variables(model.optimizer, n)
end

function MOI.add_constraint(
    model::_ScenarioOptimizer,
    func::F,
    set::Batched{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractScalarSet}
    inner_set = batch_set(set, model.scenario)
    inner_ci = MOI.add_constraint(model.optimizer, func, inner_set)
    outer_ci = MOI.ConstraintIndex{F,typeof(set)}(inner_ci.value)
    model.constraint_map[outer_ci] = inner_ci
    return outer_ci
end

function MOI.add_constraint(
    model::_ScenarioOptimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet,
)
    return MOI.add_constraint(model.optimizer, func, set)
end

function MOI.add_constrained_variable(
    model::_ScenarioOptimizer,
    set::Batched{S},
) where {S<:MOI.AbstractScalarSet}
    inner_set = batch_set(set, model.scenario)
    vi, inner_ci = MOI.add_constrained_variable(model.optimizer, inner_set)
    outer_ci = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(inner_ci.value)
    model.constraint_map[outer_ci] = inner_ci
    return vi, outer_ci
end

function MOI.add_constrained_variable(
    model::_ScenarioOptimizer,
    set::MOI.AbstractScalarSet,
)
    return MOI.add_constrained_variable(model.optimizer, set)
end

function MOI.supports_add_constrained_variable(
    model::_ScenarioOptimizer,
    ::Type{<:Batched{S}},
) where {S<:MOI.AbstractScalarSet}
    return MOI.supports_add_constrained_variable(model.optimizer, S)
end

function MOI.supports_add_constrained_variable(
    model::_ScenarioOptimizer,
    S::Type{<:MOI.AbstractScalarSet},
)
    return MOI.supports_add_constrained_variable(model.optimizer, S)
end

function MOI.supports_add_constrained_variables(
    model::_ScenarioOptimizer,
    S::Type{<:MOI.AbstractVectorSet},
)
    return MOI.supports_add_constrained_variables(model.optimizer, S)
end

function MOI.supports_constraint(
    model::_ScenarioOptimizer,
    F::Type{<:MOI.AbstractFunction},
    ::Type{<:Batched{S}},
) where {S<:MOI.AbstractScalarSet}
    return MOI.supports_constraint(model.optimizer, F, S)
end

function MOI.supports_constraint(
    model::_ScenarioOptimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.optimizer, F, S)
end

function MOI.supports(
    model::_ScenarioOptimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
)
    return MOI.supports(model.optimizer, attr)
end

function MOI.supports(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(model.optimizer, attr, MOI.VariableIndex)
end

function MOI.supports(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,Batched{S}}},
) where {F,S<:MOI.AbstractScalarSet}
    return MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{F,S})
end

function MOI.supports(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    index_type::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.optimizer, attr, index_type)
end

function MOI.set(
    model::_ScenarioOptimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
    value,
)
    MOI.set(model.optimizer, attr, value)
    return
end

function MOI.set(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    MOI.set(model.optimizer, attr, vi, value)
    return
end

function MOI.set(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.set(
        model.optimizer,
        attr,
        get(model.constraint_map, ci, ci),
        value,
    )
    return
end

function MOI.get(
    model::_ScenarioOptimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
)
    return MOI.get(model.optimizer, attr, vi)
end

function MOI.get(
    model::_ScenarioOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model.optimizer, attr, get(model.constraint_map, ci, ci))
end

function _batched_constraint_sizes(model::Optimizer)
    sizes = Int[]
    for (F, S) in MOI.get(model.cache, MOI.ListOfConstraintTypesPresent())
        if S <: Batched
            for ci in MOI.get(model.cache, MOI.ListOfConstraintIndices{F,S}())
                set = MOI.get(model.cache, MOI.ConstraintSet(), ci)
                push!(sizes, batch_size(set))
            end
        end
    end
    return sizes
end

function batch_size(model::Optimizer)
    sizes = _batched_constraint_sizes(model)
    if isempty(sizes)
        return 1
    end
    first_size = first(sizes)
    if any(!=(first_size), sizes)
        throw(
            DimensionMismatch(
                "All Batched sets in a model must have the same batch size.",
            ),
        )
    end
    return first_size
end

function _apply_optimizer_attributes!(inner, model::Optimizer)
    for attr in MOI.get(model.cache, MOI.ListOfOptimizerAttributesSet())
        value = MOI.get(model.cache, attr)
        if value !== nothing
            MOI.set(inner, attr, value)
        end
    end
    return
end

function _optimize!(model::Optimizer, ::CopyScenarioSolver)
    for i in 1:batch_size(model)
        inner = _new_inner_optimizer(model)
        _apply_optimizer_attributes!(inner, model)
        scenario = _ScenarioOptimizer(inner, i)
        index_map = MOI.copy_to(scenario, model.cache)
        MOI.optimize!(inner)
        push!(
            model.scenarios,
            _OptimizerScenarioResult(
                inner,
                index_map,
                copy(scenario.constraint_map),
            ),
        )
    end
    return
end

function MOI.optimize!(model::Optimizer)
    empty!(model.scenarios)
    return _optimize!(model, model.scenario_solver)
end

function _check_scenario_index(model::Optimizer, attr)
    MOI.check_result_index_bounds(model, attr)
    return model.scenarios[attr.result_index]
end

function _try_get(default, model, attr, args...)
    try
        return MOI.get(model, attr, args...)
    catch
        return default
    end
end

function _snapshot_result(
    model::Optimizer,
    inner::MOI.ModelLike,
    index_map::MOIU.IndexMap,
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex},
)
    result = _OptimizerScenarioResult(inner, index_map, constraint_map)
    result_count = MOI.get(inner, MOI.ResultCount())
    has_result = result_count >= 1
    primal_status =
        has_result ? MOI.get(inner, MOI.PrimalStatus(1)) : MOI.NO_SOLUTION
    dual_status =
        has_result ? MOI.get(inner, MOI.DualStatus(1)) : MOI.NO_SOLUTION
    variable_primal = Dict{MOI.VariableIndex,Any}()
    if primal_status != MOI.NO_SOLUTION
        for vi in MOI.get(model.cache, MOI.ListOfVariableIndices())
            value = _try_get(nothing, inner, MOI.VariablePrimal(1), index_map[vi])
            if value !== nothing
                variable_primal[vi] = value
            end
        end
    end
    constraint_primal = Dict{MOI.ConstraintIndex,Any}()
    constraint_dual = Dict{MOI.ConstraintIndex,Any}()
    for (F, S) in MOI.get(model.cache, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model.cache, MOI.ListOfConstraintIndices{F,S}())
            inner_ci = _inner_constraint_index(result, ci)
            primal = _try_get(nothing, inner, MOI.ConstraintPrimal(1), inner_ci)
            if primal !== nothing
                constraint_primal[ci] = primal
            end
            dual = _try_get(nothing, inner, MOI.ConstraintDual(1), inner_ci)
            if dual !== nothing
                constraint_dual[ci] = dual
            end
        end
    end
    return _SnapshotScenarioResult(
        MOI.get(inner, MOI.TerminationStatus()),
        primal_status,
        dual_status,
        _try_get(nothing, inner, MOI.ObjectiveValue(1)),
        _try_get(nothing, inner, MOI.DualObjectiveValue(1)),
        variable_primal,
        constraint_primal,
        constraint_dual,
    )
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return length(model.scenarios)
end

function _termination_status(result::_OptimizerScenarioResult)
    return MOI.get(result.optimizer, MOI.TerminationStatus())
end

_termination_status(result::_SnapshotScenarioResult) = result.termination_status

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if isempty(model.scenarios)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    statuses = _termination_status.(model.scenarios)
    return all(==(first(statuses)), statuses) ? first(statuses) : MOI.OTHER_ERROR
end

function MOI.get(model::Optimizer, attr::BatchTerminationStatus)
    result = _check_scenario_index(model, attr)
    return _termination_status(result)
end

function termination_status(model::Optimizer, result_index::Int)
    return MOI.get(model, BatchTerminationStatus(result_index))
end

function _primal_status(result::_OptimizerScenarioResult)
    return MOI.get(result.optimizer, MOI.PrimalStatus(1))
end

_primal_status(result::_SnapshotScenarioResult) = result.primal_status

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    return _primal_status(model.scenarios[attr.result_index])
end

function _dual_status(result::_OptimizerScenarioResult)
    return MOI.get(result.optimizer, MOI.DualStatus(1))
end

_dual_status(result::_SnapshotScenarioResult) = result.dual_status

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    return _dual_status(model.scenarios[attr.result_index])
end

function _objective_value(result::_OptimizerScenarioResult)
    return MOI.get(result.optimizer, MOI.ObjectiveValue(1))
end

function _objective_value(result::_SnapshotScenarioResult)
    if result.objective_value === nothing
        throw(MOI.GetAttributeNotAllowed(MOI.ObjectiveValue(1)))
    end
    return result.objective_value
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    result = _check_scenario_index(model, attr)
    return _objective_value(result)
end

function _dual_objective_value(result::_OptimizerScenarioResult)
    return MOI.get(result.optimizer, MOI.DualObjectiveValue(1))
end

function _dual_objective_value(result::_SnapshotScenarioResult)
    if result.dual_objective_value === nothing
        throw(MOI.GetAttributeNotAllowed(MOI.DualObjectiveValue(1)))
    end
    return result.dual_objective_value
end

function MOI.get(model::Optimizer, attr::MOI.DualObjectiveValue)
    result = _check_scenario_index(model, attr)
    return _dual_objective_value(result)
end

function _variable_primal(
    result::_OptimizerScenarioResult,
    vi::MOI.VariableIndex,
)
    return MOI.get(result.optimizer, MOI.VariablePrimal(1), result.index_map[vi])
end

function _variable_primal(
    result::_SnapshotScenarioResult,
    vi::MOI.VariableIndex,
)
    if !haskey(result.variable_primal, vi)
        throw(MOI.GetAttributeNotAllowed(MOI.VariablePrimal(1)))
    end
    return result.variable_primal[vi]
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    result = _check_scenario_index(model, attr)
    return _variable_primal(result, vi)
end

function _constraint_primal(
    result::_OptimizerScenarioResult,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(
        result.optimizer,
        MOI.ConstraintPrimal(1),
        _inner_constraint_index(result, ci),
    )
end

function _constraint_primal(
    result::_SnapshotScenarioResult,
    ci::MOI.ConstraintIndex,
)
    if !haskey(result.constraint_primal, ci)
        throw(MOI.GetAttributeNotAllowed(MOI.ConstraintPrimal(1)))
    end
    return result.constraint_primal[ci]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    result = _check_scenario_index(model, attr)
    return _constraint_primal(result, ci)
end

function _constraint_dual(
    result::_OptimizerScenarioResult,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(
        result.optimizer,
        MOI.ConstraintDual(1),
        _inner_constraint_index(result, ci),
    )
end

function _constraint_dual(result::_SnapshotScenarioResult, ci::MOI.ConstraintIndex)
    if !haskey(result.constraint_dual, ci)
        throw(MOI.GetAttributeNotAllowed(MOI.ConstraintDual(1)))
    end
    return result.constraint_dual[ci]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
)
    result = _check_scenario_index(model, attr)
    return _constraint_dual(result, ci)
end

function scenario_optimizer(model::Optimizer, result_index::Int)
    result = _check_scenario_index(model, BatchTerminationStatus(result_index))
    if result isa _SnapshotScenarioResult
        error(
            "Scenario optimizers are not retained by this scenario solver. " *
            "Query results through MOI or JuMP instead.",
        )
    end
    return result.optimizer
end
