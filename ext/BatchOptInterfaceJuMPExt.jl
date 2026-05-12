module BatchOptInterfaceJuMPExt

import BatchOptInterface as BOI
import JuMP
import MathOptInterface as MOI

function JuMP.moi_set(parameter::BOI.BatchedParameter{T}) where {T}
    return BOI.Batched(MOI.Parameter{T}.(parameter.values))
end

function JuMP.build_variable(
    ::Function,
    variable::JuMP.AbstractVariable,
    parameter::BOI.BatchedParameter,
)
    return JuMP.VariableConstrainedOnCreation(variable, JuMP.moi_set(parameter))
end

function JuMP.build_variable(
    error_fn::Function,
    variables::AbstractArray{<:JuMP.AbstractVariable},
    parameter::BOI.BatchedParameter,
)
    return JuMP.build_variable.(error_fn, variables, Ref(parameter))
end

function JuMP.build_variable(
    error_fn::Function,
    variables::AbstractArray{<:JuMP.AbstractVariable},
    parameters::AbstractArray{<:BOI.BatchedParameter},
)
    if length(variables) != length(parameters)
        return error_fn(
            "Dimensions must match. Got a vector of scalar variables with " *
            "$(length(variables)) elements and a vector of BatchedParameter " *
            "sets with $(length(parameters)) elements.",
        )
    end
    return JuMP.build_variable.(error_fn, variables, parameters)
end

function _batched_parameter_index(x::JuMP.GenericVariableRef)
    T = JuMP.value_type(typeof(x))
    F, S = MOI.VariableIndex, BOI.Batched{MOI.Parameter{T}}
    return MOI.ConstraintIndex{F,S}(JuMP.index(x).value)
end

function BOI.is_batched_parameter(x::JuMP.GenericVariableRef)
    return MOI.is_valid(
        JuMP.backend(JuMP.owner_model(x)),
        _batched_parameter_index(x),
    )::Bool
end

function BOI.BatchedParameterRef(x::JuMP.GenericVariableRef)
    if !BOI.is_batched_parameter(x)
        error("Variable $x is not a batched parameter.")
    end
    return JuMP.ConstraintRef(
        JuMP.owner_model(x),
        _batched_parameter_index(x),
        JuMP.ScalarShape(),
    )
end

function BOI.batch_parameter_values(x::JuMP.GenericVariableRef)
    T = JuMP.value_type(typeof(x))
    set = MOI.get(
        JuMP.owner_model(x),
        MOI.ConstraintSet(),
        BOI.BatchedParameterRef(x),
    )::BOI.Batched{MOI.Parameter{T}}
    return [parameter.value for parameter in set.sets]
end

function BOI.set_batch_parameter_values(x::JuMP.GenericVariableRef, values)
    old_values = BOI.batch_parameter_values(x)
    new_values = collect(values)
    if length(new_values) != length(old_values)
        throw(
            DimensionMismatch(
                "Cannot change batch size from $(length(old_values)) to " *
                "$(length(new_values)).",
            ),
        )
    end
    model = JuMP.owner_model(x)
    T = JuMP.value_type(typeof(x))
    model.is_model_dirty = true
    set = BOI.Batched(MOI.Parameter{T}.(T.(new_values)))
    MOI.set(model, MOI.ConstraintSet(), BOI.BatchedParameterRef(x), set)
    return
end

function BOI.termination_status(model::JuMP.GenericModel, result_index::Int)
    return MOI.get(JuMP.backend(model), BOI.BatchTerminationStatus(result_index))
end

function BOI.batch_size(model::JuMP.GenericModel)
    return BOI.batch_size(JuMP.unsafe_backend(model))
end

function BOI.scenario_optimizer(model::JuMP.GenericModel, result_index::Int)
    return BOI.scenario_optimizer(JuMP.unsafe_backend(model), result_index)
end

end
