module BatchNLPKernels

using ExaModels
using KernelAbstractions

const ExaKA = Base.get_extension(ExaModels, :ExaModelsKernelAbstractions)
const KAExtension = ExaKA.KAExtension

include("batch_model.jl")

const BOI = BatchNLPKernels
export BOI, BatchModel, BatchModelConfig
export obj_batch!, grad_batch!, cons_nln_batch!, jac_coord_batch!, hess_coord_batch!
export jprod_nln_batch!, jtprod_nln_batch!, hprod_batch!

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

include("kernels.jl")
include("api/cons.jl")
include("api/grad.jl")
include("api/hess.jl")
include("api/jac.jl")
include("api/obj.jl")
include("api/jprod.jl")
include("api/hprod.jl")

end # module BatchNLPKernels