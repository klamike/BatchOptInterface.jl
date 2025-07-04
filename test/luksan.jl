luksan_vlcek_obj(x, i, j) = 100 * (x[i-1, j]^2 - x[i, j])^2 + (x[i-1, j] - 1)^2
luksan_vlcek_con1(x, i, j) = 3x[i+1, j]^3 + 2 * x[i+2, j] - 5
luksan_vlcek_con2(x, i, j) = sin(x[i+1, j] - x[i+2, j])sin(x[i+1, j] + x[i+2, j]) + 4x[i+1, j] - x[i, j]exp(x[i, j] - x[i+1, j]) - 3
luksan_vlcek_x0(i) = mod(i, 2) == 1 ? -1.2 : 1.0
function create_luksan_vlcek_model(N; M = 1, prod = true, backend = OpenCLBackend())
    # c = ExaCore(backend = KernelAbstractions.CPU())
    c = ExaCore(backend = backend)
    x = variable(c, N, M; start = [luksan_vlcek_x0(i) for i = 1:N, j = 1:M])
    s = constraint(c, luksan_vlcek_con1(x, i, j) for i = 1:(N-2), j = 1:M)
    constraint!(c, s, (i, j) => luksan_vlcek_con2(x, i, j) for i = 1:(N-2), j = 1:M)
    objective(c, luksan_vlcek_obj(x, i, j) for i = 2:N, j = 1:M)
    return ExaModel(c; prod = prod)
end

luksan_vlcek_obj_param(x, θ, i, j) = θ[1] * (x[i-1, j]^2 - x[i, j])^2 + (x[i-1, j] - θ[2])^2
luksan_vlcek_con1_param(x, θ, i, j) = θ[3] * x[i+1, j]^3 + θ[4] * x[i+2, j] - θ[5]
luksan_vlcek_con2_param(x, θ, i, j) = sin(x[i+1, j] - x[i+2, j])sin(x[i+1, j] + x[i+2, j]) + θ[6] * x[i+1, j] - x[i, j]exp(x[i, j] - x[i+1, j]) - θ[7]
function create_luksan_vlcek_parametric_model(N; M = 1, prod = true, backend = OpenCLBackend())
    # c = ExaCore(backend = KernelAbstractions.CPU())
    c = ExaCore(backend = backend)
    x = variable(c, N, M; start = [luksan_vlcek_x0(i) for i = 1:N, j = 1:M])
    θ = parameter(c, [100.0, 1.0, 3.0, 2.0, 5.0, 4.0, 3.0])
    
    s = constraint(c, luksan_vlcek_con1_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
    constraint!(c, s, (i, j) => luksan_vlcek_con2_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
    objective(c, luksan_vlcek_obj_param(x, θ, i, j) for i = 2:N, j = 1:M)
    
    return ExaModel(c; prod = prod)
end

function create_luksan_models(backend = OpenCLBackend())
    models = ExaModel[]
    push!(models, create_luksan_vlcek_model(5; M = 1, backend = backend))
    push!(models, create_luksan_vlcek_model(8; M = 1, backend = backend))
    push!(models, create_luksan_vlcek_parametric_model(5; M = 1, backend = backend))
    return models, ["Luksan-Vlcek 5", "Luksan-Vlcek 8", "Parametric Luksan-Vlcek 5"]
end