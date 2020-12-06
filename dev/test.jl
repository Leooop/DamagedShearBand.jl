using DamagedShearBand ; const DSB = DamagedShearBand
using Plots; plotlyjs()

σ₃ = -1e6
θ = 60.0
r = DSB.Rheology(D₀=0.2)
Sc = get_damage_onset(r,σ₃)
σᵢⱼ = DSB.build_principal_stress_tensor(r,5,σ₃,Dc)

σᵢⱼ_r = DSB.change_coords(σᵢⱼ,θ)

Dc = DSB.get_KI_mininizer_D(r)
σᵢⱼ_Dc = DSB.build_principal_stress_tensor(r,100,σ₃,Dc)
KI_Dc = DSB.compute_KI(r, σᵢⱼ_Dc, Dc)


σᵢⱼ_Dc_vec = [DSB.build_principal_stress_tensor(r,S,σ₃,Dc) for S in S_vec]
τ_vec = [DSB.get_τ(dev(σᵢⱼ)) for σᵢⱼ in σᵢⱼ_Dc_vec]
p_vec = [-1/3*tr(σᵢⱼ) for σᵢⱼ in σᵢⱼ_Dc_vec]
τ_vec./p_vec
A,B = DSB.compute_AB(r,Dc)
A/B
KI_vec = [DSB.compute_KI(r, σᵢⱼ, Dc) for σᵢⱼ in σᵢⱼ_Dc_vec]

KI_vec = [DSB.compute_KI(r, σ,τ, Dc) for (σ,τ) in zip(-p_vec,τ_vec)]
-A.*p_vec .+ B.*τ_vec

##
r = DSB.Rheology(D₀=0.24)
D_out_i = Dc = DSB.get_KI_mininizer_D(r)
D_in_i = 0.9
S_vec = 1:0.1:10
D_vec = r.D₀:0.01:0.99
KI_mat = [DSB.compute_KI(r, DSB.build_principal_stress_tensor(r,S,σ₃,D), D) for S in S_vec, D in D_vec]
DSB.compute_KI(r, DSB.build_principal_stress_tensor(r,10,σ₃,Dc), Dc) 
Sc = get_damage_onset(r,σ₃,D_in_i)

heatmap(D_vec,S_vec,KI_mat./r.K₁c,clim=(0,0.1))
xlabel!("D")
ylabel!("S")



solve(r,σᵢⱼ_in,ϵᵢⱼ_in,knowns,unknowns,D,Δt ; abstol=1e-12, maxiter=100)