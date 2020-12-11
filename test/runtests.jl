using DamagedShearBand; const DSB = DamagedShearBand
using Test

# Data :
S, σ₃ = 5, -1e6
r = DSB.Rheology()
D = r.D₀
σᵢⱼ = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])

@testset "get_plane_strain_oop_stress" begin
    atol=1e-12
    σᵢⱼ_cor = set_plane_strain_oop_stress(σᵢⱼ, r, D ; abstol=atol, maxiter=100, σoop_guess=nothing)
    ϵoop = DSB.compute_ϵ_oop(r,D,σᵢⱼ_cor)
    @test abs(ϵoop)<=atol
end

@testset "get_damage_onset" begin
    Sc = get_damage_onset(r,σ₃,r.D₀)

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

@testset "change_coords" begin
    σ0 = DSB.change_coords(σᵢⱼ,0) # plane is vertical, 
    @test σ0[1,1] == σᵢⱼ[2,2]
    σm90 = DSB.change_coords(σᵢⱼ,-90) # plane is vertical, 
    @test all(σm90 .== σᵢⱼ)
    σ90 = DSB.change_coords(σᵢⱼ,90) # plane is horizontal, 
    @test all(σ90 .== σᵢⱼ)
    σ90 = DSB.change_coords(σᵢⱼ,90+360) # plane is horizontal, 
    @test all(σ90 .== σᵢⱼ)
end

@testset "get_KI_minimizer_D" begin
    r = DSB.Rheology(D₀=0.25)
    Dc = DSB.get_KI_mininizer_D(r,S,σ₃)
    σᵢⱼ_Dc = DSB.build_principal_stress_tensor(r,S,σ₃,Dc)
    KI_Dc = DSB.compute_KI(r, σᵢⱼ_Dc, Dc)

    # KI after damage onset
    KI_up = DSB.KI_from_external_load(r, S, σ₃, Dc+0.01)
    @test KI_up > KI_Dc

    # KI before damage onset
    KI_down = DSB.KI_from_external_load(r, S, σ₃, Dc-0.01)
    @test KI_down > KI_Dc
end

@testset "solving using given ϵ̇11. D=D₀=0 test vs elastic rheology" begin
    tspan = (0.0,1.0)
    Δt = 0.1
    ϵ̇11 = -1e-5
    r = DSB.Rheology(D₀=0.0) # object containing elastic moduli and damage parameters.
    D_i = r.D₀
    σ₃ = -1e6
    S = 1
    
    σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint
    ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)
    t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec = DSB.time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan ; abstol=1e-12, maxiter=100)

    Ce = DSB.get_elastic_stiffness_tensor(r::Rheology)
    σᵢⱼ_vec_elast = Ref(Ce) .⊡ ϵᵢⱼ_vec

    @test all([σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[2,2] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[2,2] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[1,2] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,2] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[1,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,3] for σᵢⱼ in σᵢⱼ_vec_elast])
    @test all([σᵢⱼ[2,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[2,3] for σᵢⱼ in σᵢⱼ_vec_elast])
end