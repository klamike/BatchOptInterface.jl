module BatchOptInterfacePOIExt

import BatchOptInterface as BOI
import MathOptInterface as MOI
import ParametricOptInterface as POI

struct _POIScenarioSolver
    evaluate_duals::Bool
    save_original_objective_and_constraints::Bool
end

function BOI.POIScenarioSolver(;
    evaluate_duals::Bool = true,
    save_original_objective_and_constraints::Bool = true,
)
    return _POIScenarioSolver(
        evaluate_duals,
        save_original_objective_and_constraints,
    )
end

function _new_poi_optimizer(
    model::BOI.Optimizer{T},
    solver::_POIScenarioSolver,
) where {T}
    return POI.Optimizer{T}(
        model.optimizer_constructor;
        with_bridge_type = model.with_bridge_type,
        with_cache_type = model.with_cache_type,
        evaluate_duals = solver.evaluate_duals,
        save_original_objective_and_constraints =
            solver.save_original_objective_and_constraints,
    )
end

function _update_batched_sets!(
    inner::MOI.ModelLike,
    cache::MOI.ModelLike,
    result::BOI._OptimizerScenarioResult,
    scenario::Int,
)
    for (F, S) in MOI.get(cache, MOI.ListOfConstraintTypesPresent())
        if !(S <: BOI.Batched)
            continue
        end
        for ci in MOI.get(cache, MOI.ListOfConstraintIndices{F,S}())
            set = MOI.get(cache, MOI.ConstraintSet(), ci)
            MOI.set(
                inner,
                MOI.ConstraintSet(),
                BOI._inner_constraint_index(result, ci),
                BOI.batch_set(set, scenario),
            )
        end
    end
    return
end

function BOI._optimize!(
    model::BOI.Optimizer,
    solver::_POIScenarioSolver,
)
    inner = _new_poi_optimizer(model, solver)
    BOI._apply_optimizer_attributes!(inner, model)
    scenario = BOI._ScenarioOptimizer(inner, 1)
    index_map = MOI.copy_to(scenario, model.cache)
    result = BOI._OptimizerScenarioResult(
        inner,
        index_map,
        copy(scenario.constraint_map),
    )
    for i in 1:BOI.batch_size(model)
        _update_batched_sets!(inner, model.cache, result, i)
        MOI.optimize!(inner)
        push!(
            model.scenarios,
            BOI._snapshot_result(
                model,
                inner,
                index_map,
                copy(scenario.constraint_map),
            ),
        )
    end
    return
end

end
