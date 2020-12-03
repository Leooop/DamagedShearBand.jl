using DamagedShearBand; const DSB = DamagedShearBand
using Test

# Data :
S, σ₃ = 3, -1e6
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
    Sc = get_damage_onset(r,σ₃)

    # KI at damage onset
    σᵢⱼ_onset = DSB.build_elastic_stress_tensor(r,Sc,σ₃)
    KI_onset = DSB.compute_KI(r, σᵢⱼ_onset, r.D₀)
    @test KI_onset ≈ 0

    # KI after damage onset
    σᵢⱼ_up = DSB.build_elastic_stress_tensor(r,Sc+0.1,σ₃)
    KI_up = DSB.compute_KI(r, σᵢⱼ_up, r.D₀)
    @test KI_up > 0

    # KI before damage onset
    σᵢⱼ_down = DSB.build_elastic_stress_tensor(r,Sc-0.1,σ₃)
    KI_down = DSB.compute_KI(r, σᵢⱼ_down, r.D₀)
    @test KI_down < 0
end
