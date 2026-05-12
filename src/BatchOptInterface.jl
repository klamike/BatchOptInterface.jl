module BatchOptInterface

import MathOptInterface as MOI

const MOIU = MOI.Utilities

export Batch,
    Batched,
    BatchedParameter,
    BatchedParameterRef,
    BatchTerminationStatus,
    CopyScenarioSolver,
    FlexPOIScenarioSolver,
    Optimizer,
    POIScenarioSolver,
    batch_parameter_values,
    batch_set,
    batch_size,
    is_batched_parameter,
    scenario_optimizer,
    set_batch_parameter_values

function BatchedParameterRef end
function batch_parameter_values end
function is_batched_parameter end
function FlexPOIScenarioSolver end
function POIScenarioSolver end
function set_batch_parameter_values end
function termination_status end

include("batched_set.jl")
include("optimizer.jl")

end
