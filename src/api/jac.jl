"""
    jac_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate Jacobian coordinates for a batch of points.
"""
function jac_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    J_view = _maybe_view(bm, :jprod_work, X)
    jac_coord_batch!(bm, X, Θ, J_view)
    return J_view
end

"""
    jac_coord_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate Jacobian coordinates for a batch of points.
"""
function jac_coord_batch!(bm::BatchModel, X::AbstractMatrix)
    Θ = _repeat_params(bm, X)
    jac_coord_batch!(bm, X, Θ)
end

function jac_coord_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    J::AbstractMatrix,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(J)
    @lencheck length(bm.model.meta.x0) eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.nnzj eachrow(J)
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)
    
    fill!(J, zero(eltype(J)))
    _jac_coord_batch!(backend, J, bm.model.cons, X, Θ)
    return J
end

function _jac_coord_batch!(backend, J, cons, X, Θ)
    sjacobian_batch!(backend, J, nothing, cons, X, Θ, one(eltype(J)))
    _jac_coord_batch!(backend, J, cons.inner, X, Θ)
    synchronize(backend)
end
function _jac_coord_batch!(backend, J, cons::ExaModels.ConstraintNull, X, Θ) end

function sjacobian_batch!(
    backend::B,
    y1,
    y2,
    f,
    X,
    Θ,
    adj,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        batch_size = size(X, 2)
        kerj_batch(backend)(y1, y2, f.f, f.itr, X, Θ, adj; ndrange = (length(f.itr), batch_size))
    end
end
