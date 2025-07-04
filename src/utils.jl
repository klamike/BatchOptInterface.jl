# https://groups.google.com/forum/?fromgroups=#!topic/julia-users/b6RbQ2amKzg
macro lencheck(l, vars...)
    exprs = Expr[]
    for var in vars
      varname = string(var)
      push!(exprs, :(
        if length($(esc(var))) != $(esc(l))
          throw(DimensionMismatch(string("Dimension mismatch: ", $varname, " (", $(esc(l)), ") != ", length($(esc(var))))))
        end
      ))
    end
    Expr(:block, exprs...)
end

function _assert_batch_size(b, bmax)
    @assert b <= bmax "Batch size $b exceeds maximum batch size $bmax"
end

function _maybe_view(bm, buffer_name, X)
    batch_size = size(X, 2)
    _assert_batch_size(batch_size, bm.batch_size)
    _check_buffer_available(getfield(bm, buffer_name), buffer_name)
    return view(getfield(bm, buffer_name), :, 1:batch_size)
end

function _repeat_params(bm, X)
    Θ = repeat(bm.model.θ, 1, size(X, 2))
    return Θ
end

_get_prodhelper(bm::BatchModel) = _get_prodhelper(bm.model)
_get_prodhelper(model::ExaModels.ExaModel) = model.ext.prodhelper

_get_backend(bm::BatchModel) = _get_backend(bm.model)
_get_backend(model::ExaModels.ExaModel) = model.ext.backend

function _check_buffer_available(buffer, buffer_name::Symbol)
    if isnothing(buffer)
        config_field_str = split(string(buffer_name), "_")[1]
        throw(ArgumentError("$buffer_name is not available. Set $config_field_str = true in BatchModelConfig when creating BatchModel to enable."))
    end
    return buffer
end