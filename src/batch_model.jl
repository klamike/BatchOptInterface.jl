"""
    BatchModelConfig

Configuration struct for controlling which buffers are allocated in a BatchModel.

## Fields
- `obj::Bool`: Allocate objective buffer (default: true)
- `cons::Bool`: Allocate constraint buffers (default: true)
- `grad::Bool`: Allocate gradient buffers (default: false)
- `jac::Bool`: Allocate jacobian buffer (default: false)
- `hess::Bool`: Allocate hessian buffer (default: false)
- `jprod::Bool`: Allocate jacobian-vector product buffer (default: false)
- `jtprod::Bool`: Allocate jacobian transpose-vector product buffer (default: false)
- `hprod::Bool`: Allocate hessian-vector product buffer (default: false)
"""
struct BatchModelConfig
    obj::Bool
    cons::Bool
    grad::Bool
    jac::Bool
    hess::Bool
    jprod::Bool
    jtprod::Bool
    hprod::Bool
end

"""
    BatchModelConfig(; obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false)

Create a BatchModelConfig with specified buffer allocations.
"""
function BatchModelConfig(; obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false)
    return BatchModelConfig(obj, cons, grad, jac, hess, jprod, jtprod, hprod)
end

"""
    BatchModelConfig(:minimal)

Minimal configuration with only objective and constraint buffers.
"""
BatchModelConfig(::Val{:minimal}) = BatchModelConfig(obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false)

"""
    BatchModelConfig(:full)

Full configuration with all buffers allocated.
"""
BatchModelConfig(::Val{:full}) = BatchModelConfig(obj=true, cons=true, grad=true, jac=true, hess=true, jprod=true, jtprod=true, hprod=true)

BatchModelConfig(s::Symbol) = BatchModelConfig(Val(s))

"""
    BatchModel{MT,E}

A wrapper around ExaModel that pre-initializes buffers for batch operations.
Allows efficient evaluation of multiple points simultaneously.

## Fields
- `model::ExaModel`: The underlying ExaModel
- `batch_size::Int`: Number of points to evaluate simultaneously
- `objbuffer::Union{MT,Nothing}`: Batch objective values (nobj × batch_size)
- `consbuffer::Union{MT,Nothing}`: Batch constraint values (nconaug × batch_size)
- `consout::Union{MT,Nothing}`: Dense constraint output buffer (ncon × batch_size)
- `gradbuffer::Union{MT,Nothing}`: Batch gradient values (nnzg × batch_size)
- `gradout::Union{MT,Nothing}`: Dense gradient output buffer (nvar × batch_size)
- `jacbuffer::Union{MT,Nothing}`: Batch jacobian values (nnzj × batch_size)
- `hessbuffer::Union{MT,Nothing}`: Batch hessian values (nnzh × batch_size)
- `jprod_buffer::Union{MT,Nothing}`: Batch jacobian-vector product buffer (ncon × batch_size)
- `jtprod_buffer::Union{MT,Nothing}`: Batch jacobian transpose-vector product buffer (nvar × batch_size)
- `hprod_buffer::Union{MT,Nothing}`: Batch hessian-vector product buffer (nvar × batch_size)
"""
struct BatchModel{MT,E}
    model::E
    batch_size::Int

    objbuffer::Union{MT,Nothing}
    consbuffer::Union{MT,Nothing}
    consout::Union{MT,Nothing}
    gradbuffer::Union{MT,Nothing}
    gradout::Union{MT,Nothing}
    jacbuffer::Union{MT,Nothing}
    hessbuffer::Union{MT,Nothing}
    jprod_buffer::Union{MT,Nothing}
    jtprod_buffer::Union{MT,Nothing}
    hprod_buffer::Union{MT,Nothing}
end

"""
    BatchModel(model::ExaModel, batch_size::Int; config=BatchModelConfig())

Create a BatchModel wrapper around an ExaModel with pre-allocated buffers
for batch operations. Use `config` to specify which buffers to allocate.
"""
function BatchModel(model::ExaModels.ExaModel{T,VT,E,O,C}, batch_size::Int; config=BatchModelConfig(:minimal)) where {T,VT,E,O,C}
    @assert batch_size > 0 "Batch size must be positive"
    @assert E <: KAExtension "ExaModel must be created with a KernelAbstractions backend"
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nnzj = model.meta.nnzj
    nnzh = model.meta.nnzh

    nobj = length(model.ext.objbuffer)
    nnzg = length(model.ext.gradbuffer) 
    nconaug = length(model.ext.conbuffer)

    has_prodhelper = !isnothing(model.ext.prodhelper)
    if (config.jprod || config.jtprod || config.hprod) && !has_prodhelper
        error("Matrix-vector operations (jprod, jtprod, hprod) require ExaModel to be initialized with prod = true")
    end
    if (config.jprod || config.jtprod) && !config.jac
        error("jprod and jtprod operations require jac = true in BatchModelConfig")
    end
    if config.hprod && !config.hess
        error("hprod operation requires hess = true in BatchModelConfig")
    end

    o = model.ext.objbuffer
    objbuffer = config.obj ? similar(o, nobj, batch_size) : nothing
    consbuffer = config.cons ? similar(o, nconaug, batch_size) : nothing
    consout = config.cons ? similar(o, ncon, batch_size) : nothing
    gradbuffer = config.grad ? similar(o, nnzg, batch_size) : nothing
    gradout = config.grad ? similar(o, nvar, batch_size) : nothing
    jacbuffer = config.jac ? similar(o, nnzj, batch_size) : nothing
    hessbuffer = config.hess ? similar(o, nnzh, batch_size) : nothing
    
    jprod_buffer = config.jprod ? similar(o, ncon, batch_size) : nothing
    jtprod_buffer = config.jtprod ? similar(o, nvar, batch_size) : nothing
    hprod_buffer = config.hprod ? similar(o, nvar, batch_size) : nothing
    
    return BatchModel(
        model,
        batch_size,
        objbuffer,
        consbuffer,
        consout,
        gradbuffer,
        gradout,
        jacbuffer,
        hessbuffer,
        jprod_buffer,
        jtprod_buffer,
        hprod_buffer,
    )
end

# Convenience accessors
Base.getproperty(bm::BatchModel, sym::Symbol) = 
    sym in fieldnames(BatchModel) ? getfield(bm, sym) : getproperty(bm.model, sym)

# Helper function to check buffer availability and provide informative errors
function _check_buffer_available(buffer, buffer_name::String, config_field::String)
    if isnothing(buffer)
        throw(ArgumentError("$buffer_name is not available. Set $config_field = true in BatchModelConfig when creating BatchModel to enable."))
    end
    return buffer
end

function Base.show(io::IO, bm::BatchModel{T,VT}) where {T,VT}
    println(io, "BatchModel{$T, $VT, ...} with max batch size = $(bm.batch_size)")
    println(io, "Wrapping:")
    Base.show(io, bm.model)
end