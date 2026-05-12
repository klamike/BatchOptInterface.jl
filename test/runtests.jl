using BatchOptInterface
using FlexPOI
using HiGHS
using JuMP
using MathOptInterface
using ParametricOptInterface
using Test

const BOI = BatchOptInterface
const MOI = MathOptInterface
const POI = ParametricOptInterface

@testset "Batched set" begin
    set = BOI.Batched(MOI.Parameter.(Float64[1, 2, 3]))
    @test BOI.batch_size(set) == 3
    @test BOI.batch_set(set, 2) == MOI.Parameter(2.0)
    @test_throws ArgumentError BOI.Batched(MOI.Parameter{Float64}[])
    @test_throws ArgumentError BOI.BatchedParameter(Float64[])
    shifted = MOI.Utilities.shift_constant(set, 1.0)
    @test BOI.batch_parameter_values(
        let
            model = Model()
            @variable(model, p in BOI.BatchedParameter([1, 2, 3]))
            p
        end,
    ) == [1.0, 2.0, 3.0]
    @test BOI.batch_set(shifted, 1) == MOI.Parameter(2.0)
end

@testset "JuMP extension" begin
    model = Model()
    @variable(model, p in BOI.BatchedParameter(1:3))
    @variable(model, x >= 0)
    @constraint(model, x * p >= 1)
    @constraint(model, 2x in BOI.Batched(MOI.LessThan.([2.0, 4.0, 6.0])))
    @test BOI.is_batched_parameter(p)
    @test !BOI.is_batched_parameter(x)
    @test BOI.batch_parameter_values(p) == [1.0, 2.0, 3.0]
    BOI.set_batch_parameter_values(p, [3, 4, 5])
    @test BOI.batch_parameter_values(p) == [3.0, 4.0, 5.0]
    @test_throws DimensionMismatch BOI.set_batch_parameter_values(p, [1, 2])
end

@testset "MOI optimizer cache" begin
    model = BOI.Optimizer(() -> MOI.Utilities.MockOptimizer())
    x = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(
        model,
        BOI.Batched(MOI.Parameter.(Float64[2, 3])),
    )
    f = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, x, p)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.add_constraint(model, f, MOI.GreaterThan(1.0))
    @test BOI.batch_size(model) == 2
end

@testset "Scenario optimize" begin
    function factory()
        mock = MOI.Utilities.MockOptimizer()
        mock.optimize! = m -> MOI.Utilities.mock_optimize!(m, MOI.OPTIMAL)
        return mock
    end
    model = Model(() -> BOI.Optimizer(factory))
    @variable(model, x >= 0)
    @variable(model, p in BOI.BatchedParameter([2.0, 3.0]))
    @constraint(model, x * p >= 1)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test result_count(model) == 2
    @test BOI.termination_status(model, 1) == MOI.OPTIMAL
    @test BOI.termination_status(model, 2) == MOI.OPTIMAL
end

@testset "HiGHS with POI scenario solver" begin
    model = Model(
        () -> BOI.Optimizer(
            HiGHS.Optimizer;
            scenario_solver = BOI.POIScenarioSolver(),
            with_bridge_type = Float64,
        ),
    )
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in BOI.BatchedParameter([1.0, 2.0, 4.0]))
    @objective(model, Min, x)
    @constraint(model, x * p >= 1)
    @constraint(model, x in BOI.Batched(MOI.LessThan.([2.0, 1.0, 0.3])))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test result_count(model) == 3
    @test BOI.termination_status(model, 1) == MOI.OPTIMAL
    @test BOI.termination_status(model, 2) == MOI.OPTIMAL
    @test BOI.termination_status(model, 3) == MOI.OPTIMAL
    @test [value(x; result = i) for i in 1:3] ≈ [1.0, 0.5, 0.25]
    @test [value(p; result = i) for i in 1:3] == [1.0, 2.0, 4.0]
end

@testset "HiGHS with FlexPOI scenario solver" begin
    model = Model(
        () -> BOI.Optimizer(
            HiGHS.Optimizer;
            scenario_solver = BOI.FlexPOIScenarioSolver(),
            with_bridge_type = Float64,
        ),
    )
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in BOI.BatchedParameter([1.0, 2.0, 4.0]))
    @objective(model, Min, x)
    @constraint(model, x * p >= 1)
    @constraint(model, x in BOI.Batched(MOI.LessThan.([2.0, 1.0, 0.3])))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test result_count(model) == 3
    @test BOI.termination_status(model, 1) == MOI.OPTIMAL
    @test BOI.termination_status(model, 2) == MOI.OPTIMAL
    @test BOI.termination_status(model, 3) == MOI.OPTIMAL
    @test [value(x; result = i) for i in 1:3] ≈ [1.0, 0.5, 0.25]
    @test [value(p; result = i) for i in 1:3] == [1.0, 2.0, 4.0]
end

@testset "FlexPOI simplifies nonlinear parameter expressions" begin
    model = Model(
        () -> BOI.Optimizer(
            HiGHS.Optimizer;
            scenario_solver = BOI.FlexPOIScenarioSolver(),
            with_bridge_type = Float64,
        ),
    )
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, p in BOI.BatchedParameter([pi / 2, pi / 6]))
    @objective(model, Max, x - x * cos(p))
    @constraint(model, x * sin(p) <= 1)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test result_count(model) == 2
    @test BOI.termination_status(model, 1) == MOI.OPTIMAL
    @test BOI.termination_status(model, 2) == MOI.OPTIMAL
    @test [value(x; result = i) for i in 1:2] ≈ [1.0, 2.0] atol = 1e-6
end

@testset "Unsupported inner constraints" begin
    model = Model(
        () -> BOI.Optimizer(
            HiGHS.Optimizer;
            scenario_solver = BOI.POIScenarioSolver(),
            with_bridge_type = Float64,
        ),
    )
    set_silent(model)
    @variable(model, x)
    @constraint(model, sin(x) >= 0.5)
    @test_throws MOI.UnsupportedConstraint{
        MOI.ScalarNonlinearFunction,
        MOI.GreaterThan{Float64},
    } optimize!(model)
end

@testset "Mismatched batch sizes" begin
    model = Model(() -> BOI.Optimizer(HiGHS.Optimizer; with_bridge_type = Float64))
    @variable(model, x)
    @variable(model, p in BOI.BatchedParameter([1.0, 2.0]))
    @constraint(model, x >= p)
    @constraint(model, x in BOI.Batched(MOI.LessThan.([1.0, 2.0, 3.0])))
    @test_throws DimensionMismatch optimize!(model)
end
