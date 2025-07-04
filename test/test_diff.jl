using DifferentiationInterface
const DI = DifferentiationInterface

import Zygote
import FiniteDifferences


function test_diff_gpu(model::ExaModel, batch_size::Int)
    bm = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:full))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X_cpu = randn(nvar, batch_size)
    Θ_cpu = randn(nθ, batch_size)
    
    X_gpu = CLArray(X_cpu)
    Θ_gpu = CLArray(Θ_cpu)
    
    @testset "obj_batch! CLArray" begin
        y = BOI.obj_batch!(bm, X_gpu, Θ_gpu)
        @test y isa CLArray
        @test size(y) == (batch_size,)

        function f_gpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BOI.obj_batch!(bm, X, Θ))
        end
        
        params = vcat(X_gpu, Θ_gpu)
        grad = DI.gradient(f_gpu, AutoZygote(), params)
        @test grad isa AbstractMatrix && grad isa CLArray
        @test size(grad) == size(params)
    end
    
    ncon == 0 && return

    @testset "cons_nln_batch! CLArray" begin
        y = BOI.cons_nln_batch!(bm, X_gpu, Θ_gpu)
        @test y isa CLArray
        @test size(y) == (ncon, batch_size)

        function f_gpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BOI.cons_nln_batch!(bm, X, Θ))
        end
        
        params = vcat(X_gpu, Θ_gpu)
        grad = DI.gradient(f_gpu, AutoZygote(), params)
        @test grad isa AbstractMatrix && grad isa CLArray
        @test size(grad) == size(params)
    end
end

function test_diff_cpu(model::ExaModel, batch_size::Int)
    bm = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:full))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X_cpu = randn(nvar, batch_size)
    Θ_cpu = randn(nθ, batch_size)
    
    @testset "obj_batch! CPU" begin
        y = BOI.obj_batch!(bm, X_cpu, Θ_cpu)
        @test size(y) == (batch_size,)

        function f_cpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BOI.obj_batch!(bm, X, Θ))
        end
        
        params = vcat(X_cpu, Θ_cpu)
        grad = DI.gradient(f_cpu, AutoZygote(), params)
        @test grad isa AbstractMatrix
        @test size(grad) == size(params)

        @testset "FiniteDifferences obj_batch!" begin
            gradfd = DI.gradient(f_cpu, AutoFiniteDifferences(fdm=FiniteDifferences.central_fdm(3,1)), params)
            @test gradfd[1:nvar,:] ≈ grad[1:nvar,:] atol=1e-4 rtol=1e-4
        end
    end

    ncon == 0 && return
    
    @testset "cons_nln_batch! CPU" begin
        y = BOI.cons_nln_batch!(bm, X_cpu, Θ_cpu)
        @test size(y) == (ncon, batch_size)

        function f_cpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BOI.cons_nln_batch!(bm, X, Θ))
        end
        
        params = vcat(X_cpu, Θ_cpu)
        grad = DI.gradient(f_cpu, AutoZygote(), params)
        @test grad isa AbstractMatrix
        @test size(grad) == size(params)

        @testset "FiniteDifferences cons_nln_batch!" begin
            gradfd = DI.gradient(f_cpu, AutoFiniteDifferences(fdm=FiniteDifferences.central_fdm(3,1)), params)
            @test gradfd[1:nvar,:] ≈ grad[1:nvar,:] atol=1e-4 rtol=1e-4
        end
    end
end


@testset "AD rules" begin
    cpu_models, names = create_luksan_models(CPU())
    gpu_models, _ = create_luksan_models(OpenCLBackend())
    
    for (name, (cpu_model, gpu_model)) in zip(names, zip(cpu_models, gpu_models))
        @testset "$name Model" begin
            for batch_size in [1, 4]
                @testset "Batch Size $batch_size" begin
                    @testset "CPU Diff" begin
                        test_diff_cpu(cpu_model, batch_size)
                    end
                    @testset "GPU Diff" begin
                        test_diff_gpu(gpu_model, batch_size)
                    end
                end
            end
        end
    end
end
