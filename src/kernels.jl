@kernel function kerf_batch(Y, @Const(f), @Const(itr), @Const(X), @Const(Θ))
    I, batch_idx = @index(Global, NTuple)
    @inbounds Y[ExaModels.offset0(f, itr, I), batch_idx] = f.f(itr[I], view(X, :, batch_idx), view(Θ, :, batch_idx))
end

@kernel function kerf2_batch(Y, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(oa))
    I, batch_idx = @index(Global, NTuple)
    @inbounds Y[oa+I, batch_idx] = f.f(itr[I], view(X, :, batch_idx), view(Θ, :, batch_idx))
end

@kernel function kerg_batch(Y, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adj))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.grpass(
        f.f(itr[I], ExaModels.AdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp1,
        view(Y, :, batch_idx),
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end

@kernel function kerj_batch(Y1, Y2, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adj))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.jrpass(
        f.f(itr[I], ExaModels.AdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp1,
        ExaModels.offset0(f, itr, I),
        isnothing(Y1) ? nothing : view(Y1, :, batch_idx),
        isnothing(Y2) ? nothing : view(Y2, :, batch_idx),
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end

@kernel function kerh_batch(Y1, Y2, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adj1), @Const(adj2))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.hrpass0(
        f.f(itr[I], ExaModels.SecondAdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp2,
        isnothing(Y1) ? nothing : view(Y1, :, batch_idx),
        isnothing(Y2) ? nothing : view(Y2, :, batch_idx),
        ExaModels.offset2(f, I),
        0,
        adj1,
        adj2,
    )
end

@kernel function kerh2_batch(Y1, Y2, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adjs1), @Const(adj2))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.hrpass0(
        f.f(itr[I], ExaModels.SecondAdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp2,
        isnothing(Y1) ? nothing : view(Y1, :, batch_idx),
        isnothing(Y2) ? nothing : view(Y2, :, batch_idx),
        ExaModels.offset2(f, I),
        0,
        adjs1[ExaModels.offset0(f, itr, I), batch_idx],
        adj2,
    )
end

@kernel function compress_to_dense_batch(Y, @Const(Y0), @Const(ptr), @Const(sparsity))
    I, batch_idx = @index(Global, NTuple)
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        (k, l) = sparsity[j]
        Y[k, batch_idx] += Y0[l, batch_idx]
    end
end

@kernel function kerspmv_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        Y[i, batch_idx] += V[ind, batch_idx] * X[j, batch_idx]
    end
end

@kernel function kerspmv2_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        Y[j, batch_idx] += V[ind, batch_idx] * X[i, batch_idx]
    end
end

@kernel function kersyspmv_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        Y[i, batch_idx] += V[ind, batch_idx] * X[j, batch_idx]
    end
end

@kernel function kersyspmv2_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        if i != j
            Y[j, batch_idx] += V[ind, batch_idx] * X[i, batch_idx]
        end
    end
end