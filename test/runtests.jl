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
