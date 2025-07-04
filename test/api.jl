function test_batch_model(model::ExaModel, batch_size::Int; 
                                   atol::Float64=1e-10, rtol::Float64=1e-10)
    
    bm = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:full))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X = OpenCL.randn(nvar, batch_size)
    Θ = OpenCL.randn(nθ, batch_size)
    
    @testset "Model Info: $(nvar) vars, $(ncon) cons, $(nθ) params" begin
        @testset "Objective" begin
            obj_vals = BOI.obj_batch!(bm, X, Θ)
            @test length(obj_vals) == batch_size
            @test all(isfinite, obj_vals)
            for i in 1:batch_size
                OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                OpenCL.@allowscalar @test obj_vals[i] ≈ ExaModels.obj(model, X[:, i]) atol=atol rtol=rtol
            end
        end
        
        @testset "Constraint" begin
            if ncon > 0
                cons_vals = BOI.cons_nln_batch!(bm, X, Θ)
                @test size(cons_vals) == (ncon, batch_size)
                @test all(isfinite, cons_vals)
                for i in 1:batch_size
                    OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    cons_single = CLVector{Float64}(undef, ncon)
                    OpenCL.@allowscalar ExaModels.cons_nln!(model, X[:, i], cons_single)
                    OpenCL.@allowscalar @test cons_vals[:, i] ≈ cons_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Gradient" begin
            grad_vals = BOI.grad_batch!(bm, X, Θ)
            @test size(grad_vals) == (nvar, batch_size)
            @test all(isfinite, grad_vals)
            for i in 1:batch_size
                OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                grad_single = CLVector{Float64}(undef, nvar)
                OpenCL.@allowscalar ExaModels.grad!(model, X[:, i], grad_single)
                OpenCL.@allowscalar @test grad_vals[:, i] ≈ grad_single atol=atol rtol=rtol
            end
        end
        
        @testset "Jacobian-Vector Product" begin
            if ncon > 0
                V = OpenCL.randn(nvar, batch_size)
                jprod_vals = BOI.jprod_nln_batch!(bm, X, Θ, V)
                @test size(jprod_vals) == (ncon, batch_size)
                @test all(isfinite, jprod_vals)
                for i in 1:batch_size
                    OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    jprod_single = CLVector{Float64}(undef, ncon)
                    OpenCL.@allowscalar ExaModels.jprod_nln!(model, X[:, i], V[:, i], jprod_single)
                    OpenCL.@allowscalar @test jprod_vals[:, i] ≈ jprod_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Jacobian-Transpose-Vector Product" begin
            if ncon > 0
                V = OpenCL.randn(ncon, batch_size)
                jtprod_vals = BOI.jtprod_nln_batch!(bm, X, Θ, V)
                @test size(jtprod_vals) == (nvar, batch_size)
                @test all(isfinite, jtprod_vals)
                for i in 1:batch_size
                    OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    jtprod_single = CLVector{Float64}(undef, nvar)
                    OpenCL.@allowscalar ExaModels.jtprod_nln!(model, X[:, i], V[:, i], jtprod_single)
                    OpenCL.@allowscalar @test jtprod_vals[:, i] ≈ jtprod_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Hessian-Vector Product" begin
            V = OpenCL.randn(nvar, batch_size)
            if ncon > 0
                Y = OpenCL.randn(ncon, batch_size)
                hprod_vals = BOI.hprod_batch!(bm, X, Θ, Y, V)
                @test size(hprod_vals) == (nvar, batch_size)
                @test all(isfinite, hprod_vals)
                for i in 1:batch_size
                    OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    hprod_single = CLVector{Float64}(undef, nvar)
                    OpenCL.@allowscalar ExaModels.hprod!(model, X[:, i], Y[:, i], V[:, i], hprod_single)
                    OpenCL.@allowscalar @test hprod_vals[:, i] ≈ hprod_single atol=atol rtol=rtol
                end
            else
                Y = OpenCL.zeros(ncon, batch_size)
                hprod_vals = BOI.hprod_batch!(bm, X, Θ, Y, V)
                @test size(hprod_vals) == (nvar, batch_size)
                @test all(isfinite, hprod_vals)
                for i in 1:batch_size
                    OpenCL.@allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    hprod_single = CLVector{Float64}(undef, nvar)
                    OpenCL.@allowscalar ExaModels.hprod!(model, X[:, i], Y[:, i], V[:, i], hprod_single)
                    OpenCL.@allowscalar @test hprod_vals[:, i] ≈ hprod_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Batch Size Validation" begin
            X_large = OpenCL.randn(nvar, batch_size + 1)
            @test_throws AssertionError BOI.obj_batch!(bm, X_large)
            
            if ncon > 0
                @test_throws AssertionError BOI.cons_nln_batch!(bm, X_large)
            end
            
            @test_throws AssertionError BOI.grad_batch!(bm, X_large)
            
            if ncon > 0
                V_jprod = OpenCL.randn(nvar, batch_size + 1)
                @test_throws AssertionError BOI.jprod_nln_batch!(bm, X_large, V_jprod)
                
                V_jtprod = OpenCL.randn(ncon, batch_size + 1)
                @test_throws AssertionError BOI.jtprod_nln_batch!(bm, X_large, V_jtprod)
            end
            
            V_hprod = OpenCL.randn(nvar, batch_size + 1)
            if ncon > 0
                Y_large = OpenCL.randn(ncon, batch_size + 1)
                @test_throws AssertionError BOI.hprod_batch!(bm, X_large, Y_large, V_hprod)
            else
                Y_large = OpenCL.zeros(ncon, batch_size + 1)
                @test_throws AssertionError BOI.hprod_batch!(bm, X_large, Y_large, V_hprod)
            end
        end
        
        @testset "Dimension Validation" begin
            X_wrong = OpenCL.randn(nvar + 1, batch_size)
            @test_throws DimensionMismatch BOI.obj_batch!(bm, X_wrong)

            if nθ > 0
                Θ_wrong = OpenCL.randn(nθ + 1, batch_size)
                @test_throws DimensionMismatch BOI.obj_batch!(bm, X, Θ_wrong)
            end
            
            if ncon > 0
                V_jprod_wrong = OpenCL.randn(nvar + 1, batch_size)
                @test_throws DimensionMismatch BOI.jprod_nln_batch!(bm, X, V_jprod_wrong)
                
                V_jtprod_wrong = OpenCL.randn(ncon + 1, batch_size)
                @test_throws DimensionMismatch BOI.jtprod_nln_batch!(bm, X, V_jtprod_wrong)
                
                Y_wrong = OpenCL.randn(ncon + 1, batch_size)
                V_hprod = OpenCL.randn(nvar, batch_size)
                @test_throws DimensionMismatch BOI.hprod_batch!(bm, X, Y_wrong, V_hprod)
            end

            V_hprod_wrong = OpenCL.randn(nvar + 1, batch_size)
            if ncon > 0
                Y = OpenCL.randn(ncon, batch_size)
                @test_throws DimensionMismatch BOI.hprod_batch!(bm, X, Y, V_hprod_wrong)
            else
                Y = OpenCL.zeros(ncon, batch_size)
                @test_throws DimensionMismatch BOI.hprod_batch!(bm, X, Y, V_hprod_wrong)
            end
        end
    end
end

@testset "API" begin
    models, names = create_luksan_models()
    
    for (name, model) in zip(names, models)
        @testset "$name Model" begin
            for batch_size in [1, 2, 4]
                @testset "Batch Size $batch_size" begin
                    test_batch_model(model, batch_size, atol=1e-5, rtol=1e-5)
                end
            end
        end
    end
end
