struct EvaluatorExpressions{E,CE,V,CV}
    objective::E
    objective_p::V

    objective_gradient::CE
    objective_gradient_p::V

    constraints::CE
    constraint_p::CV

    constraint_jacobian::CE
    constraint_jacobian_p::CV
    jacobian_structure::Vector{Tuple{Int,Int}}

    hessian_lagrangian::CE
    hessian_lagrangian_p::CV
    hessian_structure::Vector{Tuple{Int,Int}}


    variable_mapping::Dict{UInt64, Dict{Int,Int}}
    objective_H_structure::Vector{Tuple{Int,Int}}
    constraint_H_structures::Vector{Vector{Tuple{Int,Int}}}
    objective_hash::Union{Nothing, UInt64}
    constraint_hashes::Vector{UInt64}
end

function EvaluatorExpressions(;
    objective::E, objective_p::V,
    objective_gradient::CE, objective_gradient_p::V,
    constraints::CE, constraint_p::CV,
    constraint_jacobian::CE, constraint_jacobian_p::CV, jacobian_structure::Vector{Tuple{Int,Int}},
    hessian_lagrangian::CE, hessian_lagrangian_p::CV, hessian_structure::Vector{Tuple{Int,Int}},
    variable_mapping::Dict{UInt64, Dict{Int,Int}},
    objective_H_structure::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
    constraint_H_structures::Vector{Vector{Tuple{Int,Int}}} = Vector{Tuple{Int,Int}}[],
    objective_hash::Union{Nothing, UInt64} = nothing,
    constraint_hashes::Vector{UInt64} = UInt64[]
) where {E,CE,V,CV}
    return EvaluatorExpressions{E,CE,V,CV}(
        objective, objective_p,
        objective_gradient, objective_gradient_p,
        constraints, constraint_p,
        constraint_jacobian, constraint_jacobian_p, jacobian_structure,
        hessian_lagrangian, hessian_lagrangian_p, hessian_structure,
        variable_mapping,
        objective_H_structure, constraint_H_structures, objective_hash, constraint_hashes
    )
end

function Base.convert(::Type{EvaluatorExpressions{E,CE,V,CV}}, ee::EvaluatorExpressions{E1,CE1,V1,CV1}) where {E,CE,V,CV,E1,CE1,V1,CV1}
    return EvaluatorExpressions{E,CE,V,CV}(
        ee.objective, ee.objective_p,
        ee.objective_gradient, ee.objective_gradient_p,
        ee.constraints, ee.constraint_p,
        ee.constraint_jacobian, ee.constraint_jacobian_p, ee.jacobian_structure,
        ee.hessian_lagrangian, ee.hessian_lagrangian_p, ee.hessian_structure,
        ee.variable_mapping,
        ee.objective_H_structure, ee.constraint_H_structures, ee.objective_hash, ee.constraint_hashes
    )
end

function _evaluator_to_expressions(evaluator::SymbolicAD.Evaluator)
    objective_expr = nothing
    objective_gradient_exprs = nothing
    objective_p = nothing
    objective_gradient_p = nothing
    constraint_exprs = Expr[]
    constraint_p = Vector{Vector{Float64}}()
    jacobian_exprs = Expr[]
    jacobian_p = Vector{Vector{Float64}}()
    jacobian_structure = Tuple{Int,Int}[]
    hessian_exprs = Expr[]
    hessian_p = Vector{Vector{Float64}}()
    hessian_structure = Tuple{Int,Int}[]
    variable_mapping = Dict{UInt64, Dict{Int,Int}}()
    
    # New fields for Hessian structure information
    objective_H_structure = Tuple{Int,Int}[]
    constraint_H_structures = Vector{Tuple{Int,Int}}[]
    objective_hash = nothing
    constraint_hashes = UInt64[]
    
    # Extract objective
    if !isnothing(evaluator.objective)
        o = evaluator.objective
        dag = evaluator.dag[o.hash]
        dag_exprs = dag_to_expressions(dag)
        
        variable_mapping[o.hash] = Dict(i => o.x[i] for i in eachindex(o.x))
        objective_expr = dag_exprs[1]
        objective_p = copy(o.p)
        objective_hash = o.hash
        objective_H_structure = copy(evaluator.H[o.hash])
        
        if length(dag_exprs) > 1
            objective_gradient_exprs = dag_exprs[2:(1 + length(o.x))]
            objective_gradient_p = copy(o.p)
        end
    end
    
    # Extract constraints
    processed_hashes = Set{UInt64}()
    for (i, c) in enumerate(evaluator.constraints)
        dag = evaluator.dag[c.hash]
        
        if !(c.hash in processed_hashes)
            push!(processed_hashes, c.hash)
            variable_mapping[c.hash] = Dict(j => c.x[j] for j in eachindex(c.x))
        end
        
        push!(constraint_hashes, c.hash)
        push!(constraint_H_structures, copy(evaluator.H[c.hash]))
        
        dag_exprs = dag_to_expressions(dag)
        push!(constraint_exprs, dag_exprs[1])
        push!(constraint_p, copy(c.p))
        
        # Jacobian entries
        if length(dag_exprs) > 1
            for (j, xj) in enumerate(c.x)
                push!(jacobian_exprs, dag_exprs[1 + j])
                push!(jacobian_p, copy(c.p))
                push!(jacobian_structure, (i, xj))
            end
        end
    end
    
    # Objective hessian terms first
    if !isnothing(evaluator.objective)
        o = evaluator.objective
        dag = evaluator.dag[o.hash]
        dag_exprs = dag_to_expressions(dag)
        H_structure = evaluator.H[o.hash]
        
        hess_start = 2 + length(o.x)
        for (k, (hi, hj)) in enumerate(H_structure)
            if hess_start + k - 1 <= length(dag_exprs)
                push!(hessian_exprs, dag_exprs[hess_start + k - 1])
                push!(hessian_p, copy(o.p))
                push!(hessian_structure, (o.x[hi], o.x[hj]))
            end
        end
    end
    
    # Constraint hessian terms in order
    for c in evaluator.constraints
        dag = evaluator.dag[c.hash]
        dag_exprs = dag_to_expressions(dag)
        H_structure = evaluator.H[c.hash]
        
        hess_start = 2 + length(c.x)
        for (k, (hi, hj)) in enumerate(H_structure)
            if hess_start + k - 1 <= length(dag_exprs)
                push!(hessian_exprs, dag_exprs[hess_start + k - 1])
                push!(hessian_p, copy(c.p))
                push!(hessian_structure, (c.x[hi], c.x[hj]))
            end
        end
    end

    return EvaluatorExpressions(
        objective = objective_expr,
        objective_p = objective_p,
        objective_gradient = objective_gradient_exprs,
        objective_gradient_p = objective_gradient_p,
        constraints = constraint_exprs,
        constraint_p = constraint_p,
        constraint_jacobian = jacobian_exprs,
        constraint_jacobian_p = jacobian_p,
        jacobian_structure = jacobian_structure,
        hessian_lagrangian = hessian_exprs,
        hessian_lagrangian_p = hessian_p,
        hessian_structure = hessian_structure,
        variable_mapping = variable_mapping,
        objective_H_structure = objective_H_structure,
        constraint_H_structures = constraint_H_structures,
        objective_hash = objective_hash,
        constraint_hashes = constraint_hashes
    )
end