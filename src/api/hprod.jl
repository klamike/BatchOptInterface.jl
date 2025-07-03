"""
    hprod_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix, Hv::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian-vector products for a batch of points.
"""
function hprod_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    Y::AbstractMatrix,
    V::AbstractMatrix,
    Hv::AbstractMatrix;
    obj_weight=1.0,
)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @assert !isnothing(bm.hprod_buffer) "ExaModel was not created with prod=true. Matrix-vector products not supported."
    @assert !isnothing(bm.model.ext.prodhelper) "ExaModel was not created with prod=true. Matrix-vector products not supported."
    
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(Y) eachcol(V) eachcol(Hv)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(V) eachrow(Hv)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Y)
    
    _check_buffer_available(bm.hessbuffer, "hessbuffer", "hess")
    H_batch = view(bm.hessbuffer, :, 1:batch_size)
    hess_coord_batch!(bm, X, Θ, Y, H_batch; obj_weight=obj_weight)
    
    fill!(Hv, zero(eltype(Hv)))
    kersyspmv_batch(bm.model.ext.backend)(
        Hv,
        V,
        bm.model.ext.prodhelper.hesssparsityi,
        H_batch,
        bm.model.ext.prodhelper.hessptri;
        ndrange = (length(bm.model.ext.prodhelper.hessptri) - 1, batch_size),
    )
    synchronize(bm.model.ext.backend)
    
    kersyspmv2_batch(bm.model.ext.backend)(
        Hv,
        V,
        bm.model.ext.prodhelper.hesssparsityj,
        H_batch,
        bm.model.ext.prodhelper.hessptrj;
        ndrange = (length(bm.model.ext.prodhelper.hessptrj) - 1, batch_size),
    )
    synchronize(bm.model.ext.backend)
    
    return Hv
end

"""
    hprod_batch!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian-vector products for a batch of points.
"""
function hprod_batch!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    
    _check_buffer_available(bm.hprod_buffer, "hprod_buffer", "hprod")
    Hv = view(bm.hprod_buffer, :, 1:batch_size)
    Θ = repeat(bm.model.θ, 1, batch_size)
    hprod_batch!(bm, X, Θ, Y, V, Hv; obj_weight=obj_weight)
    return Hv
end

"""
    hprod_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian-vector products for a batch of points.
"""
function hprod_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    
    _check_buffer_available(bm.hprod_buffer, "hprod_buffer", "hprod")
    Hv = view(bm.hprod_buffer, :, 1:batch_size)
    hprod_batch!(bm, X, Θ, Y, V, Hv; obj_weight=obj_weight)
    return Hv
end 