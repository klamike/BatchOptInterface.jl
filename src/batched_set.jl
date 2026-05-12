struct Batched{S<:MOI.AbstractScalarSet} <: MOI.AbstractScalarSet
    sets::Vector{S}
    function Batched(sets::AbstractVector{S}) where {S<:MOI.AbstractScalarSet}
        if isempty(sets)
            throw(ArgumentError("Batched sets must contain at least one set."))
        end
        return new{S}(collect(sets))
    end
end

const Batch = Batched

function Batched(sets::S...) where {S<:MOI.AbstractScalarSet}
    return Batched(collect(sets))
end

Base.copy(set::Batched) = Batched(copy.(set.sets))
Base.broadcastable(set::Batched) = Ref(set)

batch_size(set::Batched) = length(set.sets)
batch_set(set::Batched, index::Integer) = set.sets[index]

function Base.show(io::IO, set::Batched)
    print(io, "Batched(")
    show(io, set.sets)
    return print(io, ")")
end

function MOIU.supports_shift_constant(::Type{<:Batched{S}}) where {S}
    return MOIU.supports_shift_constant(S)
end

function MOIU.shift_constant(set::Batched, offset)
    return Batched([MOIU.shift_constant(s, offset) for s in set.sets])
end

struct BatchedParameter{T}
    values::Vector{T}
    function BatchedParameter(values)
        values_vector = collect(values)
        if isempty(values_vector)
            throw(
                ArgumentError(
                    "BatchedParameter must contain at least one value.",
                ),
            )
        end
        T = promote_type(map(typeof, values_vector)...)
        return new{T}(T.(values_vector))
    end
end

Base.broadcastable(parameter::BatchedParameter) = Ref(parameter)

batch_size(parameter::BatchedParameter) = length(parameter.values)

function Base.show(io::IO, parameter::BatchedParameter)
    print(io, "BatchedParameter(")
    show(io, parameter.values)
    return print(io, ")")
end
