@time (using DamagedShearBand ; const DSB = DamagedShearBand)
using Plots
#ENV["JULIA_DEBUG"] = DamagedShearBand


# σ₃ = -1e6
# θ = 60.0
r = DSB.Rheology(D₀=0.2,ν=0.5)
# Sc = get_damage_onset(r,σ₃,D)
# σᵢⱼ = DSB.build_principal_stress_tensor(r,5,σ₃,Dc)

# σᵢⱼ_r = DSB.band_coords(σᵢⱼ,θ)

# Dc = DSB.get_KI_mininizer_D(r)
# σᵢⱼ_Dc = DSB.build_principal_stress_tensor(r,100,σ₃,Dc)
# KI_Dc = DSB.compute_KI(r, σᵢⱼ_Dc, Dc)


# σᵢⱼ_Dc_vec = [DSB.build_principal_stress_tensor(r,S,σ₃,Dc) for S in S_vec]
# τ_vec = [DSB.get_τ(dev(σᵢⱼ)) for σᵢⱼ in σᵢⱼ_Dc_vec]
# p_vec = [-1/3*tr(σᵢⱼ) for σᵢⱼ in σᵢⱼ_Dc_vec]
# τ_vec./p_vec
# A,B = DSB.compute_AB(r,Dc)
# A/B
σ_vec = -1e6:-10e6:-10e8
D_vec = r.D₀:0.01:0.999
  KI_vec = [DSB.compute_KI(r,σ,-2* r.μ*σ,D) for σ in σ_vec, D in D_vec]
  KI_pos = KI_vec.>=0
  heatmap(string.(D_vec),string.(σ_vec./1e6),KI_pos,aspect_ratio=:equal,xlabel="D",ylabel="σ (MPa)")

KI_vec = [DSB.compute_KI(r, σ,τ, Dc) for (σ,τ) in zip(-p_vec,τ_vec)]
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
# σᵢⱼ_in_ξη = DSB.band_coords(σᵢⱼ_in,θ)
# σᵢⱼ_out_ξη = DSB.band_coords(σᵢⱼ_out,θ)
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
# Δσ = DSB.band_coords(σᵢⱼ_in_next,θ) - σᵢⱼ_in_ξη 

# ϵᵢⱼ_in_ξη_next = DSB.compute_ϵij(r,D_in_i,DSB.band_coords(σᵢⱼ_in_next,θ))
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
σ₂₂
## Second strategy
using DamagedShearBand ; const DSB = DamagedShearBand
using Plots
plotlyjs()

tspan = (0.0,24)
Δt = 10
ϵ̇11 = -1e-4
r = DSB.Rheology(D₀=0.25) # object containing elastic moduli and damage parameters.
σ₃ = -1e6
S = 1
D_i = r.D₀
σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint
ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)
#σᵢⱼ_i2,ϵᵢⱼ_i2,D_i2 = σᵢⱼ_vec_a[end], ϵᵢⱼ_vec_a[end], D_vec_a[end]
#σᵢⱼ_i,ϵᵢⱼ_i,D_i = σᵢⱼ_i2,ϵᵢⱼ_i2,D_i2

#t_vec_a, σᵢⱼ_vec_a, ϵᵢⱼ_vec_a, D_vec_a = DSB.time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan ; abstol=1e-12, maxiter=100)
t_vec_a, σᵢⱼ_vec_a, ϵᵢⱼ_vec_a, D_vec_a = DSB.adaptative_time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan ; abstol=1e-12, time_maxiter=nothing, newton_maxiter=100, e₀=(D=1e-4, σ=1.0, ϵ=1e-6))


# σ = 1/3 *tr(σᵢⱼ_i)
# I = SymmetricTensor{2,3}(DSB.δ)
# sij = σᵢⱼ_i - σ*I
# -DSB.get_τ(sij)/σ

# DSB.compute_KI(r,σᵢⱼ_i,D_i)
#σᵢⱼnext, ϵᵢⱼnext, D, u = DSB.solve_2(r, σᵢⱼ_i, ϵᵢⱼ_i, D_i, ϵ̇11, Δt ; abstol=1e-12, maxiter=10)



# ϵᵢⱼnext_test = DSB.insert_into(ϵᵢⱼ_i, (ϵᵢⱼ_i[1,1] + ϵ̇11*Δt), (1,1))
# σᵢⱼnext_test = DSB.compute_σij(r,D_i,ϵᵢⱼnext_test)
##
Ce = DSB.get_elastic_stiffness_tensor(r::Rheology)
σᵢⱼnext_e = Ce ⊡ ϵᵢⱼ_vec[end]
σᵢⱼ_vec_elast = Ref(Ce) .⊡ ϵᵢⱼ_vec_a
pente_model = (σᵢⱼ_vec_a[end][1,1]-σᵢⱼ_vec_a[2][1,1]) / (ϵᵢⱼ_vec_a[end][1,1]-ϵᵢⱼ_vec_a[2][1,1])
pente_elast = (σᵢⱼ_vec_elast[end][1,1]-σᵢⱼ_vec_elast[2][1,1]) / (ϵᵢⱼ_vec_a[end][1,1]-ϵᵢⱼ_vec_a[2][1,1])
E = DSB.E_from_Gν(r.G,r.ν)
tan_modulus = E/(1-r.ν^2)
# ϵ̇ij_ana, Ḋ = DSB.compute_ϵ̇ij(r,D_i,σᵢⱼ_i,σᵢⱼnext,Δt)




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

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
  append!(sp1.series_list, sp2.series_list)
  Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
  Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
  Plots.expand_extrema!(sp1[:zaxis], zlims(sp2))
  return sp1
end

function merge_series!(plt, plts...)
  for (i, sp) in enumerate(plt.subplots)
      for other_plt in plts
          if i in eachindex(other_plt.subplots)
              merge_series!(sp, other_plt[i])
          end
      end
  end
  return plt
end
##
layout = @layout [a ; b]
l2 = @layout [a ; b]

p1 = plot(-ϵ₁₁_vec,-σ₁₁_vec)
#plot(p1,-ϵ₁₁_vec,-σ₁₁_vec)
title!(p1,"D0 = $(r.D₀) & epsdot = $(ϵ̇₁₁) & p=$(-σ₂₂)")
ylabel!("-σ₁₁")

p2 = plot(-ϵ₁₁_vec,D_vec,c=:red)
#plot(p2,-ϵ₁₁_vec,D_vec,c=:red)
ylabel!("D")
xlabel!("-ϵ₁₁")

pl1 = plot(p1,p2,layout=layout)
##
p3 = plot(-ϵ₁₁_vec.+0.0001,-σ₁₁_vec)
#plot(p1,-ϵ₁₁_vec,-σ₁₁_vec)
title!(p1,"D0 = $(r.D₀) & epsdot = $(ϵ̇₁₁) & p=$(-σ₂₂)")
ylabel!("-σ₁₁")

p4 = plot(-ϵ₁₁_vec,D_vec,c=:red)
#plot(p2,-ϵ₁₁_vec,D_vec,c=:red)
ylabel!("D")
xlabel!("-ϵ₁₁")

pl2 = plot(p3,p4,layout=l2)

##
function test_func(args...) 
  return sum([args...])
end

test_func(1,2,3) 
nt = (a=1,b=2)
@time isdefined(nt,:a)

D_vec = 0.6495327404332681:0.000001:0.6495527404332681
KI_vec = [DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,22.207207207207208,σ₃,D ; abstol=1e-15),D) for D in D_vec]
_,ind = findmin(KI_vec)
D_vec[ind]

op = DSB.OutputParams(save_frequency = n_iter_saved/(tspan[2]-tspan[1]),
                      save_period = 1000,
                      print_frequency = n_iter_printed/(tspan[2]-tspan[1]),
                      print_period = 1000)

sp = DSB.SolverParams(newton_abstol = 1e-12,
                      newton_maxiter = 100,
                      time_maxiter = nothing,
                      e₀ = 1e-4, #(D=1e-2, S=1e-2, σ=100.0, ϵ=1e-2))
                      adaptative_maxrecursions = 50)
p = DSB.Params(op,sp)
