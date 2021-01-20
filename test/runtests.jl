using DamagedShearBand; const DSB = DamagedShearBand
using Tensors
using Test

# Data :
S, σ₃ = 5, -1e6
r = DSB.Rheology()
D = r.D₀
σᵢⱼ = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])

@testset "get_plane_strain_oop_stress" begin
    atol=1e-12
    σᵢⱼ_cor = DSB.set_plane_strain_oop_stress(σᵢⱼ, r, D ; abstol=atol, maxiter=100, σoop_guess=nothing)
    ϵoop = DSB.compute_ϵ_oop(r,D,σᵢⱼ_cor)
    @test abs(ϵoop)<=atol
end

@testset "get_damage_onset" begin
    Sc = DSB.get_damage_onset(r,σ₃)

    # KI at damage onset
    σᵢⱼ_onset = DSB.build_principal_stress_tensor(r,Sc,σ₃)
    KI_onset = DSB.compute_KI(r, σᵢⱼ_onset, r.D₀)
    @test KI_onset < 1e-10

    # KI after damage onset
    σᵢⱼ_up = DSB.build_principal_stress_tensor(r,Sc+0.1,σ₃)
    KI_up = DSB.compute_KI(r, σᵢⱼ_up, r.D₀)
    @test KI_up > 0

    # KI before damage onset
    σᵢⱼ_down = DSB.build_principal_stress_tensor(r,Sc-0.1,σ₃)
    KI_down = DSB.compute_KI(r, σᵢⱼ_down, r.D₀)
    @test KI_down < 0
end

@testset "band_coords" begin
    σ0 = DSB.band_coords(σᵢⱼ,0) # plane is vertical, 
    @test σ0[1,1] == σᵢⱼ[2,2]
    σm90 = DSB.band_coords(σᵢⱼ,-90) # plane is horizontal, 
    @test all(σm90 .== σᵢⱼ)
    σ90 = DSB.band_coords(σᵢⱼ,90) # plane is horizontal, 
    @test all(σ90 .== σᵢⱼ)
    σ90 = DSB.band_coords(σᵢⱼ,90+360) # plane is horizontal, 
    @test all(σ90 .== σᵢⱼ)
    σ_rotated = DSB.band_coords(σᵢⱼ,30)
    σ_back = DSB.principal_coords(σ_rotated,30)
    @test σ_back ≈ σᵢⱼ
end

@testset "solving using given ϵ̇11. D=D₀=0 test vs elastic rheology" begin
    tspan = (0.0,1.0)
    Δt = 0.1
    ϵ̇11 = -1e-5
    r = DSB.Rheology(D₀=0.0) # object containing elastic moduli and damage parameters.
    D_i = r.D₀
    σ₃ = -1e6
    S = 1
    
    n_iter_saved = 5000
    n_iter_printed = 10
    op = DSB.OutputParams(save_frequency = n_iter_saved/(tspan[2]-tspan[1]),
                        save_period = nothing,
                        print_frequency = n_iter_printed/(tspan[2]-tspan[1]),
                        print_period = nothing)

    sp = DSB.SolverParams(newton_abstol = 1e-12,
                        newton_maxiter = 100,
                        time_maxiter = nothing,
                        e₀ = (D=1e-1, σ=100.0, ϵ=1e-3)) 

    p = DSB.Params(op,sp)

    σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint
    ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)
    t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec = DSB.time_integration(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan)

    Ce = DSB.get_elastic_stiffness_tensor(r)
    σᵢⱼ_vec_elast = Ref(Ce) .⊡ ϵᵢⱼ_vec

    @test all([σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[2,2] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[2,2] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[1,2] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,2] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[1,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,3] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[2,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[2,3] for σᵢⱼ in σᵢⱼ_vec_elast])

    
    # check tangent modulus along xx (σxx/ϵxx)
    Emod = DSB.E_from_Gν(r.G,r.ν)
    tan_mod_elast_model = (σᵢⱼ_vec_elast[end][1,1]-σᵢⱼ_vec_elast[1][1,1]) / (ϵᵢⱼ_vec[end][1,1]-ϵᵢⱼ_vec[1][1,1])
    tan_mod_Dat0_model = (σᵢⱼ_vec[end][1,1]-σᵢⱼ_vec_elast[1][1,1]) / (ϵᵢⱼ_vec[end][1,1]-ϵᵢⱼ_vec[1][1,1])
    tan_mod_theoretical = Emod/(1-r.ν^2)
    @test abs(tan_mod_Dat0_model-tan_mod_theoretical) < 1e-2
    @test abs(tan_mod_elast_model-tan_mod_theoretical) < 1e-2
end

@testset "from principal stress rate to band orientation" begin
    θ = 60.0
    Ṡ = 1e-2
    Δt = 1
    σᵢⱼ_principal = DSB.build_principal_stress_tensor(r,S,σ₃,D ; abstol=1e-15)
    σᵢⱼ1 = DSB.compute_rotated_stress_rate_from_principal_coords(r,Ṡ,σ₃,σᵢⱼ_principal,D,Δt,θ ; damaged_allowed=true)
    σᵢⱼ2 = DSB.compute_rotated_stress_rate_from_band_coords(r,Ṡ,σ₃,DSB.band_coords(σᵢⱼ_principal,θ),D,Δt,θ; damaged_allowed=true)
    @test all(σᵢⱼ1 .≈ σᵢⱼ2)
end

@testset "get_stress_deviation_from_far_field" begin
    σᵢⱼ = DSB.build_principal_stress_tensor(r,S,σ₃,D)
    σᵢⱼ_rotated_90 = DSB.insert_into(σᵢⱼ, (σᵢⱼ[1,1],σᵢⱼ[2,2]), ((2,2),(1,1))) # swap two principals in plane stresses
    σᵢⱼ_rotated_30 = DSB.band_coords(σᵢⱼ,60)
    angle_90 = DSB.get_stress_deviation_from_far_field(σᵢⱼ_rotated_90 ; offdiagtol=1e-5, compression_axis=:x)
    angle_30 = DSB.get_stress_deviation_from_far_field(σᵢⱼ_rotated_30 ; offdiagtol=1e-5, compression_axis=:x)
    @test abs(angle_90) ≈ 90
    @test abs(angle_30) ≈ 30
end

@testset "get_KI_minimizer_D" begin
    r = DSB.Rheology(D₀=0.25)
    Dc = DSB.get_KI_minimizer_D(r,S,σ₃)
    σᵢⱼ_Dc = DSB.build_principal_stress_tensor(r,S,σ₃,Dc)
    KI_Dc = DSB.compute_KI(r, σᵢⱼ_Dc, Dc)

    # KI after damage onset
    KI_up = DSB.KI_from_external_load(r, S, σ₃, Dc+0.01)
    @test KI_up > KI_Dc

    # KI before damage onset
    KI_down = DSB.KI_from_external_load(r, S, σ₃, Dc-0.01)
    @test KI_down > KI_Dc
end

@testset "get_D_and_S_that_minimize_KI" begin
    σ₃ = -1e6
    r = DSB.Rheology(D₀=0.175)
    Dc, Sc = DSB.get_KI_minimizer_D_S(r,σ₃)
    σᵢⱼ_min = DSB.build_principal_stress_tensor(r,Sc,σ₃,Dc)
    KI_min = DSB.compute_KI(r,σᵢⱼ_min,Dc)
    @test abs(KI_min) < 1e-8

    eps = 1e-6
    S_more = Sc + eps
    S_less = Sc - eps
    D_more = Dc + eps
    D_less = Dc - eps
    KI_S_more = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,S_more,σ₃,Dc),Dc)
    KI_S_less = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,S_less,σ₃,Dc),Dc)
    KI_D_more = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,Sc,σ₃,D_more),D_more)
    KI_D_less = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,Sc,σ₃,D_less),D_less)

    @test abs(KI_S_more) > abs(KI_min)
    @test abs(KI_S_less) > abs(KI_min)
    @test abs(KI_D_more) > abs(KI_min)
    @test abs(KI_S_less) > abs(KI_min)
end