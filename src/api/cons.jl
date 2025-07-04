"""
    cons_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate constraints for a batch of solutions and parameters.
"""
function cons_nln_batch!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    C = _maybe_view(bm, :cons_out, X)
    cons_nln_batch!(bm, X, Θ, C)
    return C
end

"""
    cons_nln_batch!(bm::BatchModel, X::AbstractMatrix)

Evaluate constraints for a batch of solutions.
"""
function cons_nln_batch!(bm::BatchModel, X::AbstractMatrix)
    Θ = _repeat_params(bm, X)
    cons_nln_batch!(bm, X, Θ)
end


function cons_nln_batch!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    C::AbstractMatrix,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(C)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(C)
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)

    _cons_nln_batch!(backend, C, bm.model.cons, X, Θ)

    conbuffers_batch = _maybe_view(bm, :cons_work, X)

    _conaugs_batch!(backend, conbuffers_batch, bm.model.cons, X, Θ)
    
    if length(bm.model.ext.conaugptr) > 1
        compress_to_dense_batch(backend)(
            C,
            conbuffers_batch,
            bm.model.ext.conaugptr,
            bm.model.ext.conaugsparsity;
            ndrange = (length(bm.model.ext.conaugptr) - 1, batch_size),
        )
        synchronize(backend)
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
