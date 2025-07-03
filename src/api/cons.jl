function cons_nln_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    C::AbstractMatrix,
)
    batch_size = size(X, 2)
    @assert batch_size <= bm.batch_size "Input batch size ($batch_size) exceeds model batch size ($(bm.batch_size))"
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(C)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(C)

    _cons_nln_batch!(bm.model.ext.backend, C, bm.model.cons, X, Θ)

    _check_buffer_available(bm.consbuffer, "consbuffer", "cons")
    conbuffers_batch = view(bm.consbuffer, :, 1:batch_size)
    _conaugs_batch!(bm.model.ext.backend, conbuffers_batch, bm.model.cons, X, Θ)
    
    if length(bm.model.ext.conaugptr) > 1
        compress_to_dense_batch(bm.model.ext.backend)(
            C,
            conbuffers_batch,
            bm.model.ext.conaugptr,
            bm.model.ext.conaugsparsity;
            ndrange = (length(bm.model.ext.conaugptr) - 1, batch_size),
        )
        synchronize(bm.model.ext.backend)
    end
    return C
end

function _cons_nln_batch!(backend, C, con::ExaModels.Constraint, X, Θ)
    if !isempty(con.itr)
        batch_size = size(X, 2)
        kerf_batch(backend)(C, con.f, con.itr, X, Θ; ndrange = (length(con.itr), batch_size))
    end
    _cons_nln_batch!(backend, C, con.inner, X, Θ)
    synchronize(backend)
end
function _cons_nln_batch!(backend, C, con::ExaModels.ConstraintNull, X, Θ) end
function _cons_nln_batch!(backend, C, con::ExaModels.ConstraintAug, X, Θ)
    _cons_nln_batch!(backend, C, con.inner, X, Θ)
end

function _conaugs_batch!(backend, conbuffers, con::ExaModels.ConstraintAug, X, Θ)
    if !isempty(con.itr)
        batch_size = size(X, 2)
        kerf2_batch(backend)(conbuffers, con.f, con.itr, X, Θ, con.oa; ndrange = (length(con.itr), batch_size))
    end
    _conaugs_batch!(backend, conbuffers, con.inner, X, Θ)
    synchronize(backend)
end
function _conaugs_batch!(backend, conbuffers, con::ExaModels.Constraint, X, Θ)
    _conaugs_batch!(backend, conbuffers, con.inner, X, Θ)
end
function _conaugs_batch!(backend, conbuffers, con::ExaModels.ConstraintNull, X, Θ) end


"""
    cons_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate constraints for a batch of points.
"""
function cons_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    B = size(X, 2)
    @assert B <= bm.batch_size "Input batch size ($B) exceeds model batch size ($(bm.batch_size))"
    _check_buffer_available(bm.consout, "consout", "cons")
    C = view(bm.consout, :, 1:B)
    cons_nln_batch!(bm, X, Θ, C)
    return C
end

"""
    cons_nln_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate constraints for a batch of points.
"""
function cons_nln_batch!(bm::BatchModel, X::AbstractMatrix)
    B = size(X, 2)
    @assert B <= bm.batch_size "Input batch size ($B) exceeds model batch size ($(bm.batch_size))"
    Θ = repeat(bm.model.θ, 1, B)  # FIXME: better way to do this?
    return cons_nln_batch!(bm, X, Θ)
end
