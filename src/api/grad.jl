"""
    grad_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate objective gradient for a batch of points.
"""
function grad_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    G = _maybe_view(bm, :grad_out, X)
    grad_batch!(bm, X, Θ, G)
    return G
end

"""
    grad_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate objective gradient for a batch of points.
"""
function grad_batch!(bm::BatchModel, X::AbstractMatrix)
    Θ = _repeat_params(bm, X)
    grad_batch!(bm, X, Θ)
end

function _grad_batch!(backend, grad_work, objs, X, Θ)
    sgradient_batch!(backend, grad_work, objs, X, Θ, one(eltype(grad_work)))
    _grad_batch!(backend, grad_work, objs.inner, X, Θ)
    synchronize(backend)
end
function _grad_batch!(backend, grad_work, objs::ExaModels.ObjectiveNull, X, Θ) end

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
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(G)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(G)
    @lencheck length(bm.model.θ) eachrow(Θ)  # FIXME
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)
    
    grad_work = _maybe_view(bm, :grad_work, X)
    
    if !isempty(grad_work)
        fill!(grad_work, zero(eltype(grad_work)))

        _grad_batch!(backend, grad_work, bm.model.objs, X, Θ)
        
        fill!(G, zero(eltype(G)))
        compress_to_dense_batch(backend)(
            G,
            grad_work,
            bm.model.ext.gptr,
            bm.model.ext.gsparsity;
            ndrange = (length(bm.model.ext.gptr) - 1, batch_size),
        )
        synchronize(backend)
    end
    
    return G
end
