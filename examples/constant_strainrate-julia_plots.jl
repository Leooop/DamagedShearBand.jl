
@time using DamagedShearBand ; const DSB = DamagedShearBand
using Plots # add Plots before using it
pyplot(size=(1000,400)) # slow but interactive backend for Plots, add PlotlyJS package first

# imposed values
ϵ̇₁₁ = -1e-6
σ₂₂ = -1e6 # negative in compression

# initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
S = 1.0

# Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
r = DSB.Rheology(D₀=0.4,n=34.0) 

# initial damage
D_i = r.D₀

# time integration parameters :
ϵ₁₁_goal = -0.002
tspan = (0.0,ϵ₁₁_goal/ϵ̇₁₁) # initial and final simulation time
Δt_i = 0.0001 # initial timestep

# Build stress and strain tensors 
σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₂₂,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)

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
# Integrate over time, by solving for the unknowns [σ₁₁next, σ₃₃next, ϵ₂₂next] at each timestep
# -- e₀ is the absolute tolerance used to adapt time stepping, may be a scalar or a NamedTuple with keys (D,σ,ϵ)
# -- abstol is the absolute tolerance on the unknowns in the Newton solver 
t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec = DSB.adaptative_time_integration(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇₁₁,Δt_i,tspan)

# extract σ₁₁ and ϵ₁₁
σ₁₁_vec = [σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec]
ϵ₁₁_vec = [ϵᵢⱼ[1,1] for ϵᵢⱼ in ϵᵢⱼ_vec]
ϵkk_vec = [tr(ϵᵢⱼ) for ϵᵢⱼ in ϵᵢⱼ_vec]
KI_vec = [DSB.compute_KI(r,σᵢⱼ_vec[i], D_vec[i]) for i in eachindex(σᵢⱼ_vec)]
free_energy_vec = [DSB.compute_free_energy(r,D_vec[i],ϵᵢⱼ_vec[i]) for i in eachindex(t_vec)]
## plot time series
pp1 = plot(t_vec,ϵkk_vec)
pp2 = plot(t_vec,σ₁₁_vec)
plot!(pp1,pp2)
plot(t_vec,ϵ₁₁_vec)
plot(t_vec,KI_vec./r.K₁c)
plot!(t_vec,D_vec)
plot(t_vec,strain_energy_vec)

## plot stress-strain along x with options on plot
layout = @layout [a ; b]

p1 = plot(-ϵ₁₁_vec,-σ₁₁_vec)
#plot(p1,-ϵ₁₁_vec,-σ₁₁_vec)
title!(p1,"D0 = $(r.D₀) & epsdot = $(ϵ̇₁₁) & p=$(-σ₂₂)")
ylabel!("-σ₁₁")

p2 = plot(-ϵ₁₁_vec,D_vec,c=:red)
#plot(p2,-ϵ₁₁_vec,D_vec,c=:red)
ylabel!("D")
xlabel!("-ϵ₁₁")

pl = plot(p1,p2,layout=layout)
#xlims!((0,0.005))

#savefig(p1,"./examples/singlepoint_D0=$(r.D₀)_var_epsdot.pdf")
