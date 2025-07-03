function _obj_batch!(
    bm::BatchModel,
    objbuffers,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)

    _obj_batch(bm.model.ext.backend, objbuffers, bm.model.objs, X, Θ)
    return vec(sum(objbuffers, dims=1))
end

function _obj_batch(backend, objbuffers, obj, X, Θ)
    if !isempty(obj.itr)
        batch_size = size(X, 2)
        kerf_batch(backend)(objbuffers, obj.f, obj.itr, X, Θ; ndrange = (length(obj.itr), batch_size))
    end
    _obj_batch(backend, objbuffers, obj.inner, X, Θ)
    synchronize(backend)
end
function _obj_batch(backend, objbuffers, f::ExaModels.ObjectiveNull, X, Θ) end


"""
    obj_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate objective function for a batch of points.
"""
function obj_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck batch_size eachcol(X) eachcol(Θ)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)

    _check_buffer_available(bm.objbuffer, "objbuffer", "obj")
    objbuffers = view(bm.objbuffer, :, 1:batch_size)
    
    if !isempty(objbuffers)
        fill!(objbuffers, zero(eltype(objbuffers)))
        results = _obj_batch!(bm, objbuffers, X, Θ)
        return results
    else
        return zeros(eltype(bm.objbuffer), batch_size)
    end
end


"""
    obj_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate objective function for a batch of points.
"""
function obj_batch!(bm::BatchModel, X::AbstractMatrix)
    B = size(X, 2)
    @assert B <= bm.batch_size "Input batch size ($B) exceeds model batch size ($(bm.batch_size))"
    Θ = repeat(bm.model.θ, 1, B)  # FIXME: better way to do this?
    return obj_batch!(bm, X, Θ)
end