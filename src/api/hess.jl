"""
    hess_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix, H::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian coordinates for a batch of points with different parameters.
"""
function hess_coord_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    Y::AbstractMatrix,
    H::AbstractMatrix;
    obj_weight=1.0,
)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @assert size(Θ, 2) == batch_size "X and Θ must have the same number of columns (batch size)"
    @assert size(Y, 2) == batch_size "Y must have the same number of columns as X and Θ"
    @assert size(H, 2) == batch_size "H must have the same number of columns as X and Θ"
    
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(Y) eachcol(H)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Y)
    @lencheck bm.model.meta.nnzh eachrow(H)
    
    fill!(H, zero(eltype(H)))
    _obj_hess_coord_batch!(bm.model.ext.backend, H, bm.model.objs, X, Θ, obj_weight)
    _con_hess_coord_batch!(bm.model.ext.backend, H, bm.model.cons, X, Θ, Y)
    return H
end

function _obj_hess_coord_batch!(backend, H, objs, X, Θ, obj_weight)
    shessian_batch!(backend, H, nothing, objs, X, Θ, obj_weight, zero(eltype(H)))
    _obj_hess_coord_batch!(backend, H, objs.inner, X, Θ, obj_weight)
    synchronize(backend)
end
function _obj_hess_coord_batch!(backend, H, objs::ExaModels.ObjectiveNull, X, Θ, obj_weight) end

function _con_hess_coord_batch!(backend, H, cons, X, Θ, Y)
    shessian_batch!(backend, H, nothing, cons, X, Θ, Y, zero(eltype(H)))
    _con_hess_coord_batch!(backend, H, cons.inner, X, Θ, Y)
    synchronize(backend)
end
function _con_hess_coord_batch!(backend, H, cons::ExaModels.ConstraintNull, X, Θ, Y) end

function shessian_batch!(
    backend::B,
    y1,
    y2,
    f,
    X,
    Θ,
    adj,
    adj2,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        batch_size = size(X, 2)
        kerh_batch(backend)(y1, y2, f.f, f.itr, X, Θ, adj, adj2; ndrange = (length(f.itr), batch_size))
    end
end

function shessian_batch!(
    backend::B,
    y1,
    y2,
    f,
    X,
    Θ,
    adj::AbstractMatrix,
    adj2,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        batch_size = size(X, 2)
        kerh2_batch(backend)(y1, y2, f.f, f.itr, X, Θ, adj, adj2; ndrange = (length(f.itr), batch_size))
    end
end

"""
    hess_coord_batch!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian coordinates for a batch of points.
"""
function hess_coord_batch!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck batch_size eachcol(X) eachcol(Y)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck bm.model.meta.ncon eachrow(Y)
    
    # Use the underlying model's θ parameter for all evaluations
    Θ = repeat(bm.model.θ, 1, batch_size)
    _check_buffer_available(bm.hessbuffer, "hessbuffer", "hess")
    H_view = view(bm.hessbuffer, :, 1:batch_size)
    
    # Call the batch evaluation function
    hess_coord_batch!(bm, X, Θ, Y, H_view; obj_weight=obj_weight)
    
    return H_view
end

"""
    hess_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian coordinates for a batch of points.
"""
function hess_coord_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(Y)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Y)
    
    _check_buffer_available(bm.hessbuffer, "hessbuffer", "hess")
    H_view = view(bm.hessbuffer, :, 1:batch_size)
    
    # Call the batch evaluation function
    hess_coord_batch!(bm, X, Θ, Y, H_view; obj_weight=obj_weight)
    
    return H_view
end