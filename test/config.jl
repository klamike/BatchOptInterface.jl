@testset "Config Errors" begin
    model = create_luksan_vlcek_model(5; M = 1)
    batch_size = 2
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X = OpenCL.randn(nvar, batch_size)
    Θ = OpenCL.randn(nθ, batch_size)
    
    @testset "Minimal" begin
        bm_minimal = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:minimal))
        @test_throws ArgumentError BOI.grad_batch!(bm_minimal, X, Θ)
        @test_throws ArgumentError BOI.jac_coord_batch!(bm_minimal, X, Θ)
        if ncon > 0
            Y = OpenCL.randn(ncon, batch_size)
            @test_throws ArgumentError BOI.hess_coord_batch!(bm_minimal, X, Θ, Y)
        end
    end
    
    @testset "Mat-vec" begin
        model_with_prod = create_luksan_vlcek_model(5; M = 1, prod = true)
        bm_partial = BOI.BatchModel(model_with_prod, batch_size, config=BOI.BatchModelConfig(obj=true, cons=true, grad=true, jac=true, hess=true, jprod=false, jtprod=false, hprod=false))
        
        if ncon > 0
            V = OpenCL.randn(nvar, batch_size)
            @test_throws ArgumentError BOI.jprod_nln_batch!(bm_partial, X, Θ, V)
            
            V_t = OpenCL.randn(ncon, batch_size)
            @test_throws ArgumentError BOI.jtprod_nln_batch!(bm_partial, X, Θ, V_t)
        end
        
        V_h = OpenCL.randn(nvar, batch_size)
        if ncon > 0
            Y = OpenCL.randn(ncon, batch_size)
            @test_throws ArgumentError BOI.hprod_batch!(bm_partial, X, Θ, Y, V_h)
        else
            Y = OpenCL.zeros(ncon, batch_size)
            @test_throws ArgumentError BOI.hprod_batch!(bm_partial, X, Θ, Y, V_h)
        end
    end
    
    @testset "Output" begin
        bm_no_gradout = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false))
        G = OpenCL.randn(nvar, batch_size)
        @test_throws ArgumentError BOI.grad_batch!(bm_no_gradout, X, Θ, G)
        @test_throws ArgumentError BOI.grad_batch!(bm_no_gradout, X, Θ)
        
        bm_no_cons = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(obj=true, cons=false, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false))
        @test_throws ArgumentError BOI.cons_nln_batch!(bm_no_cons, X, Θ)
    end
end
