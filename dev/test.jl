@time (using DamagedShearBand ; const DSB = DamagedShearBand)
ENV["JULIA_DEBUG"] = DamagedShearBand
using Plots; plotlyjs()

# σ₃ = -1e6
# θ = 60.0
# r = DSB.Rheology(D₀=0.2)
# Sc = get_damage_onset(r,σ₃,D)
# σᵢⱼ = DSB.build_principal_stress_tensor(r,5,σ₃,Dc)

# σᵢⱼ_r = DSB.change_coords(σᵢⱼ,θ)

# Dc = DSB.get_KI_mininizer_D(r)
# σᵢⱼ_Dc = DSB.build_principal_stress_tensor(r,100,σ₃,Dc)
# KI_Dc = DSB.compute_KI(r, σᵢⱼ_Dc, Dc)


# σᵢⱼ_Dc_vec = [DSB.build_principal_stress_tensor(r,S,σ₃,Dc) for S in S_vec]
# τ_vec = [DSB.get_τ(dev(σᵢⱼ)) for σᵢⱼ in σᵢⱼ_Dc_vec]
# p_vec = [-1/3*tr(σᵢⱼ) for σᵢⱼ in σᵢⱼ_Dc_vec]
# τ_vec./p_vec
# A,B = DSB.compute_AB(r,Dc)
# A/B
# KI_vec = [DSB.compute_KI(r, σᵢⱼ, Dc) for σᵢⱼ in σᵢⱼ_Dc_vec]

# KI_vec = [DSB.compute_KI(r, σ,τ, Dc) for (σ,τ) in zip(-p_vec,τ_vec)]
# -A.*p_vec .+ B.*τ_vec

# ##
# r = DSB.Rheology(D₀=0.24)
# D_out_i = Dc = DSB.get_KI_mininizer_D(r)
# D_in_i = 0.9
# S_vec = 1:0.1:10
# D_vec = r.D₀:0.01:0.99
# KI_mat = [DSB.compute_KI(r, DSB.build_principal_stress_tensor(r,S,σ₃,D), D) for S in S_vec, D in D_vec]
# DSB.compute_KI(r, DSB.build_principal_stress_tensor(r,10,σ₃,Dc), Dc) 
# Sc = get_damage_onset(r,σ₃,D_in_i)

# heatmap(D_vec,S_vec,KI_mat./r.K₁c,clim=(0,0.1))
# xlabel!("D")
# ylabel!("S")

## Test solve
# σ₃ = -1e6
# θ = 60
# r = DSB.Rheology(D₀=0.0)
# # set damage inside the band
# D_in_i = r.D₀
# # find S so that damage starts frowing inside the band
# S = get_damage_onset(r,σ₃,D_in_i)
# # set damage outside so that it is the most unlikely to grow with increase of S
# D_out_i = D_in_i#DSB.get_KI_mininizer_D(r,S,σ₃)
# # build stress tensors and rotate them in the band coordinates system
# σᵢⱼ_in = DSB.build_principal_stress_tensor(r,S,σ₃,D_in_i)
# σᵢⱼ_out = DSB.build_principal_stress_tensor(r,S,σ₃,D_out_i)
# σᵢⱼ_in_ξη = DSB.change_coords(σᵢⱼ_in,θ)
# σᵢⱼ_out_ξη = DSB.change_coords(σᵢⱼ_out,θ)
# # get strains
# ϵᵢⱼ_in_ξη = DSB.compute_ϵij(r,D_in_i,σᵢⱼ_in_ξη)
# ϵᵢⱼ_out_ξη = DSB.compute_ϵij(r,D_out_i,σᵢⱼ_out_ξη)

# # They are not equal on the ξξ componant => two strategies :
# # 1 - initialize with the same initial damage (anything) in and out and only allow the inner damage to grow
# # 2 - initialize with the same initial damage (the lower KI D) and increase D_in just a bit using a modified solve function 
# #     so that the deviatoric stress increase will only affect the inside of the band (puts some constraint on ϵ̇)
# Δt = 1
# Ṡ = 1e-6
# σᵢⱼ_in_next = DSB.build_principal_stress_tensor(r,S+Ṡ*Δt,σ₃,D_in_i)
# Δσ = DSB.change_coords(σᵢⱼ_in_next,θ) - σᵢⱼ_in_ξη 

# ϵᵢⱼ_in_ξη_next = DSB.compute_ϵij(r,D_in_i,DSB.change_coords(σᵢⱼ_in_next,θ))
# Δϵ = ϵᵢⱼ_in_ξη_next - ϵᵢⱼ_in_ξη

# #C = DSB.compute_damaged_stiffness_tensor(r,(ϵᵢⱼ_in_ξη_next+ϵᵢⱼ_in_ξη)/2,D_in_i)
# #Δσ2 = C ⊡ Δϵ

# #knowns = Δσηη , Δσξη, Δϵξξ
# #unknowns = Δσξξ, Δϵηη, Δϵξη
# knowns = Vec(Δσ[2,2] , Δσ[1,2], Δϵ[1,1])
# unknowns = Vec(Δσ[1,1], Δϵ[2,2], Δϵ[1,2])
# #res = DSB.residual(r, ϵᵢⱼ_in_ξη, D_out_i, knowns, unknowns)

# σᵢⱼ_in_next, ϵᵢⱼ_in_next, knowns, u, D = DSB.solve(r,σᵢⱼ_in_ξη,ϵᵢⱼ_in_ξη,knowns,unknowns,D_out_i,Δt ; abstol=1e-12, maxiter=100)

# KI_vec = [DSB.KI_to_minimize(r,S,σ₃,D) for D in r.D₀:0.01:0.99]
# plot(r.D₀:0.01:0.99,KI_vec)
# Dc = DSB.get_KI_mininizer_D(r,S,σ₃)

## Second strategy
using DamagedShearBand ; const DSB = DamagedShearBand

tspan = (0.0,1000)
Δt = 0.1
ϵ̇11 = -1e-4
r = DSB.Rheology(D₀=0.0) # object containing elastic moduli and damage parameters.
σ₃ = -1e6
S = 1
D_i = r.D₀
σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint
ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)
u_i = [σᵢⱼ_i[1,1], σᵢⱼ_i[3,3], ϵᵢⱼ_i[2,2]] # initialization of unknown vector (3,3) component is out of plane

# σ = 1/3 *tr(σᵢⱼ_i)
# I = SymmetricTensor{2,3}(DSB.δ)
# sij = σᵢⱼ_i - σ*I
# -DSB.get_τ(sij)/σ

# DSB.compute_KI(r,σᵢⱼ_i,D_i)
#σᵢⱼnext, ϵᵢⱼnext, D, u = DSB.solve_2(r, σᵢⱼ_i, ϵᵢⱼ_i, D_i, ϵ̇11, Δt ; abstol=1e-12, maxiter=10)



# ϵᵢⱼnext_test = DSB.insert_into(ϵᵢⱼ_i, (ϵᵢⱼ_i[1,1] + ϵ̇11*Δt), (1,1))
# σᵢⱼnext_test = DSB.compute_σij(r,D_i,ϵᵢⱼnext_test)

Ce = DSB.get_elastic_stiffness_tensor(r::Rheology)
σᵢⱼnext_e = Ce ⊡ ϵᵢⱼ_vec[end]
σᵢⱼ_vec_elast = Ref(Ce) .⊡ ϵᵢⱼ_vec_a
pente = (σᵢⱼ_vec_elast[end][1,1]-σᵢⱼ_vec_elast[2][1,1]) / (ϵᵢⱼ_vec_a[end][1,1]-ϵᵢⱼ_vec_a[2][1,1])
E = DSB.E_from_Gν(r.G,r.ν)
tan_modulus = E/(1-r.ν^2)
# ϵ̇ij_ana, Ḋ = DSB.compute_ϵ̇ij(r,D_i,σᵢⱼ_i,σᵢⱼnext,Δt)

t_vec_a, σᵢⱼ_vec_a, ϵᵢⱼ_vec_a, D_vec_a = DSB.time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan ; abstol=1e-12, maxiter=100)

#t_vec_a, σᵢⱼ_vec_a, ϵᵢⱼ_vec_a, D_vec_a = DSB.adaptative_time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan ; abstol=1e-15, time_maxiter=100, newton_maxiter=10, e₀=(D=1e-10, σ=1e-4, ϵ=1e-10))

##
plot(t_vec,D_vec)
  ylims!((D_vec[1],D_vec[end]))
plot(t_vec,[σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec])

  #plot!(t_vec,[σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec_elast])
#plot(t_vec,[σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec])
plot(t_vec_a,D_vec_a)
plot(diff(t_vec_a))
plot(t_vec_a,D_vec_a)
D_vec_a[end]-D_vec_a[1]
plot(t_vec_a,[σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec_a])
plot(t_vec_a,[ϵᵢⱼ[1,1] for ϵᵢⱼ in ϵᵢⱼ_vec_a])
plot([-ϵᵢⱼ[1,1] for ϵᵢⱼ in ϵᵢⱼ_vec_a],[-σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec_a])

plot!(t_vec_a[2:end],-diff(t_vec_a).*3e8)