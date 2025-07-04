"""
    jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-vector products for a batch of points.
"""
function jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)
    Jv = _maybe_view(bm, :jprod_out, X)
    jprod_nln_batch!(bm, X, Θ, V, Jv)
    return Jv
end

"""
    jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-vector products for a batch of points.
"""
function jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)
    Θ = _repeat_params(bm, X)
    jprod_nln_batch!(bm, X, Θ, V)
end


"""
    jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix, Jv::AbstractMatrix)

Evaluate Jacobian-vector products for a batch of points.
"""
function jprod_nln_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    V::AbstractMatrix,
    Jv::AbstractMatrix,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(V) eachcol(Jv)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(V)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Jv)
    _assert_batch_size(batch_size, bm.batch_size)
    ph = _get_prodhelper(bm.model)

    J_batch = _maybe_view(bm, :jprod_work, X)

    jac_coord_batch!(bm, X, Θ, J_batch)
    
    fill!(Jv, zero(eltype(Jv)))
    kerspmv_batch(bm.model.ext.backend)(
        Jv,
        V,
        ph.jacsparsityi,
        J_batch,
        ph.jacptri;
        ndrange = (length(ph.jacptri) - 1, batch_size),
    )
    synchronize(bm.model.ext.backend)
    
    return Jv
end


"""
    jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-transpose-vector products for a batch of points.
"""
function jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)
    Jtv = _maybe_view(bm, :jtprod_out, X)
    jtprod_nln_batch!(bm, X, Θ, V, Jtv)
    return Jtv
end

"""
    jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-transpose-vector products for a batch of points.
"""
function jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)
    Θ = _repeat_params(bm, X)
    jtprod_nln_batch!(bm, X, Θ, V)
end

"""
    jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix, Jtv::AbstractMatrix)

Evaluate Jacobian-transpose-vector products for a batch of points.
"""
function jtprod_nln_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    V::AbstractMatrix,
    Jtv::AbstractMatrix,
)
    batch_size = size(X, 2)
    
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(V) eachcol(Jtv)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(Jtv)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(V)
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)
    ph = _get_prodhelper(bm.model)
    
    J_batch = _maybe_view(bm, :jprod_work, X)

    jac_coord_batch!(bm, X, Θ, J_batch)
    
    fill!(Jtv, zero(eltype(Jtv)))
    kerspmv2_batch(backend)(
        Jtv,
        V,
        ph.jacsparsityj,
        J_batch,
        ph.jacptrj;
        ndrange = (length(ph.jacptrj) - 1, batch_size),
    )
    synchronize(backend)
    
    return Jtv
end
