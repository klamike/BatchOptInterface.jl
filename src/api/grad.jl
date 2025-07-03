function _grad_batch!(backend, gradbuffers, objs, X, Θ)
    sgradient_batch!(backend, gradbuffers, objs, X, Θ, one(eltype(gradbuffers)))
    _grad_batch!(backend, gradbuffers, objs.inner, X, Θ)
    synchronize(backend)
end
function _grad_batch!(backend, gradbuffers, objs::ExaModels.ObjectiveNull, X, Θ) end

function sgradient_batch!(
    backend::B,
    Y,
    f,
    X,
    Θ,
    adj,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        batch_size = size(X, 2)
        kerg_batch(backend)(Y, f.f, f.itr, X, Θ, adj; ndrange = (length(f.itr), batch_size))
    end
end

"""
    grad_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, G::AbstractMatrix)

Evaluate gradients for a batch of points with different parameters.
"""
function grad_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    G::AbstractMatrix,
)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(G)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(G)
    @lencheck length(bm.model.θ) eachrow(Θ)  # FIXME
    
    _check_buffer_available(bm.gradbuffer, "gradbuffer", "grad")
    gradbuffers = view(bm.gradbuffer, :, 1:batch_size)
    
    if !isempty(gradbuffers)
        fill!(gradbuffers, zero(eltype(gradbuffers)))
        _grad_batch!(bm.model.ext.backend, gradbuffers, bm.model.objs, X, Θ)
        
        fill!(G, zero(eltype(G)))
        batch_size = size(X, 2)
        compress_to_dense_batch(bm.model.ext.backend)(
            G,
            gradbuffers,
            bm.model.ext.gptr,
            bm.model.ext.gsparsity;
            ndrange = (length(bm.model.ext.gptr) - 1, batch_size),
        )
        synchronize(bm.model.ext.backend)
    end
    
    return G
end

"""
    grad_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate objective gradient for a batch of points.
"""
function grad_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    B = size(X, 2)
    @assert B <= bm.batch_size "Input batch size ($B) exceeds model batch size ($(bm.batch_size))"
    _check_buffer_available(bm.gradout, "gradout", "grad")
    G = view(bm.gradout, :, 1:B)
    grad_batch!(bm, X, Θ, G)
    return G
end

"""
    grad_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate objective gradient for a batch of points.
"""
function grad_batch!(bm::BatchModel, X::AbstractMatrix)
    B = size(X, 2)
    @assert B <= bm.batch_size "Input batch size ($B) exceeds model batch size ($(bm.batch_size))"
    Θ = repeat(bm.model.θ, 1, B)  # FIXME: better way to do this?
    return grad_batch!(bm, X, Θ)
end