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

include("utils.jl")
include("kernels.jl")
include("api/cons.jl")
include("api/grad.jl")
include("api/hess.jl")
include("api/jac.jl")
include("api/obj.jl")
include("api/jprod.jl")
include("api/hprod.jl")

end # module BatchNLPKernels