module BNKChainRulesCore

using BatchNLPKernels
using ChainRulesCore

function ChainRulesCore.rrule(::typeof(BatchNLPKernels.obj_batch!), bm::BatchModel, X, Θ)
    y = BatchNLPKernels.obj_batch!(bm, X, Θ)
    
    function obj_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        gradients = BatchNLPKernels.grad_batch!(bm, X, Θ)
        
        X̄ = gradients .* Ȳ'
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄, ChainRulesCore.NoTangent()
    end
    
    return y, obj_batch_pullback
end
function ChainRulesCore.rrule(::typeof(BatchNLPKernels.obj_batch!), bm::BatchModel, X)
    y = BatchNLPKernels.obj_batch!(bm, X)
    
    function obj_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        gradients = BatchNLPKernels.grad_batch!(bm, X)

        X̄ = gradients .* Ȳ'
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄
    end
    
    return y, obj_batch_pullback
end


function ChainRulesCore.rrule(::typeof(BatchNLPKernels.cons_nln_batch!), bm::BatchModel, X, Θ)
    y = BatchNLPKernels.cons_nln_batch!(bm, X, Θ)
    
    function cons_nln_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        X̄ = BatchNLPKernels.jtprod_nln_batch!(bm, X, Θ, Ȳ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄, ChainRulesCore.NoTangent()
    end
    
    return y, cons_nln_batch_pullback
end
function ChainRulesCore.rrule(::typeof(BatchNLPKernels.cons_nln_batch!), bm::BatchModel, X)
    y = BatchNLPKernels.cons_nln_batch!(bm, X)
    
    function cons_nln_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        X̄ = BatchNLPKernels.jtprod_nln_batch!(bm, X, Ȳ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄
    end
    
    return y, cons_nln_batch_pullback
end

end # module BNKChainRulesCore 