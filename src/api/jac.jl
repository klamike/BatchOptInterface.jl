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
    
    fill!(J, zero(eltype(J)))
    _jac_coord_batch!(bm.model.ext.backend, J, bm.model.cons, X, Θ)
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

"""
    jac_coord_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate Jacobian coordinates for a batch of points.
"""
function jac_coord_batch!(bm::BatchModel, X::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck bm.model.meta.nvar eachrow(X)
    
    Θ = repeat(bm.model.θ, 1, batch_size)  # FIXME: better way to do this?    
    _check_buffer_available(bm.jacbuffer, "jacbuffer", "jac")
    J_view = view(bm.jacbuffer, :, 1:batch_size)
    
    jac_coord_batch!(bm, X, Θ, J_view)
    
    return J_view
end

"""
    jac_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate Jacobian coordinates for a batch of points.
"""
function jac_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck bm.model.meta.nvar eachrow(X)
    
    _check_buffer_available(bm.jacbuffer, "jacbuffer", "jac")
    J_view = view(bm.jacbuffer, :, 1:batch_size)
    
    jac_coord_batch!(bm, X, Θ, J_view)
    
    return J_view
end