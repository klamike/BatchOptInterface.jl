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
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @assert !isnothing(bm.jprod_buffer) "ExaModel was not created with prod=true. Matrix-vector products not supported."
    @assert !isnothing(bm.model.ext.prodhelper) "ExaModel was not created with prod=true. Matrix-vector products not supported."
    
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(V) eachcol(Jv)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(V)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Jv)
    
    _check_buffer_available(bm.jacbuffer, "jacbuffer", "jac")
    J_batch = view(bm.jacbuffer, :, 1:batch_size)
    jac_coord_batch!(bm, X, Θ, J_batch)
    
    fill!(Jv, zero(eltype(Jv)))
    kerspmv_batch(bm.model.ext.backend)(
        Jv,
        V,
        bm.model.ext.prodhelper.jacsparsityi,
        J_batch,
        bm.model.ext.prodhelper.jacptri;
        ndrange = (length(bm.model.ext.prodhelper.jacptri) - 1, batch_size),
    )
    synchronize(bm.model.ext.backend)
    
    return Jv
end

"""
    jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-vector products for a batch of points.
"""
function jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    
    _check_buffer_available(bm.jprod_buffer, "jprod_buffer", "jprod")
    Jv = view(bm.jprod_buffer, :, 1:batch_size)
    Θ = repeat(bm.model.θ, 1, batch_size)
    jprod_nln_batch!(bm, X, Θ, V, Jv)
    return Jv
end

"""
    jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-vector products for a batch of points.
"""
function jprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    
    _check_buffer_available(bm.jprod_buffer, "jprod_buffer", "jprod")
    Jv = view(bm.jprod_buffer, :, 1:batch_size)
    jprod_nln_batch!(bm, X, Θ, V, Jv)
    return Jv
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
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @assert !isnothing(bm.jtprod_buffer) "ExaModel was not created with prod=true. Matrix-vector products not supported."
    @assert !isnothing(bm.model.ext.prodhelper) "ExaModel was not created with prod=true. Matrix-vector products not supported."
    
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(V) eachcol(Jtv)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(Jtv)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(V)
    
    _check_buffer_available(bm.jacbuffer, "jacbuffer", "jac")
    J_batch = view(bm.jacbuffer, :, 1:batch_size)
    jac_coord_batch!(bm, X, Θ, J_batch)
    
    fill!(Jtv, zero(eltype(Jtv)))
    kerspmv2_batch(bm.model.ext.backend)(
        Jtv,
        V,
        bm.model.ext.prodhelper.jacsparsityj,
        J_batch,
        bm.model.ext.prodhelper.jacptrj;
        ndrange = (length(bm.model.ext.prodhelper.jacptrj) - 1, batch_size),
    )
    synchronize(bm.model.ext.backend)
    
    return Jtv
end

"""
    jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-transpose-vector products for a batch of points.
"""
function jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, V::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    
    _check_buffer_available(bm.jtprod_buffer, "jtprod_buffer", "jtprod")
    Jtv = view(bm.jtprod_buffer, :, 1:batch_size)
    Θ = repeat(bm.model.θ, 1, batch_size)
    jtprod_nln_batch!(bm, X, Θ, V, Jtv)
    return Jtv
end

"""
    jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)

Evaluate Jacobian-transpose-vector products for a batch of points.
"""
function jtprod_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, V::AbstractMatrix)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    
    _check_buffer_available(bm.jtprod_buffer, "jtprod_buffer", "jtprod")
    Jtv = view(bm.jtprod_buffer, :, 1:batch_size)
    jtprod_nln_batch!(bm, X, Θ, V, Jtv)
    return Jtv
end 