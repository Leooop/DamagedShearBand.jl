using DamagedShearBand ; const DSB = DamagedShearBand
using Plots # add Plots before using it
plotlyjs() # slow but interactive backend for Plots, add PlotlyJS package first

# imposed values
ϵ̇₁₁ = -1e-4
σ₂₂ = -1e-6 # negative in compression

# initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
S = 1.0

# Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
r = DSB.Rheology(D₀=0.3) 

# initial damage
D_i = r.D₀

# time integration parameters :
tspan = (0.0,25) # initial and final simulation time
Δt_i = 0.001 # initial timestep

# Build stress and strain tensors 
σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₂₂,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)


# Integrate over time, by solving for the unknowns [σ₁₁next, σ₃₃next, ϵ₂₂next] at each timestep
# -- e₀ is the absolute tolerance used to adapt time stepping, may be a scalar or a NamedTuple with keys (D,σ,ϵ)
# -- abstol is the absolute tolerance on the unknowns in the Newton solver 
t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec = DSB.adaptative_time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇₁₁,Δt_i,tspan ; 
                                                                 abstol=1e-12, 
                                                                 time_maxiter=nothing, 
                                                                 newton_maxiter=100, 
                                                                 e₀=(D=1e-4, σ=1.0, ϵ=1e-5),
                                                                 print_frequency=10000)

# extract σ₁₁ and ϵ₁₁
σ₁₁_vec = [σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec]
ϵ₁₁_vec = [ϵᵢⱼ[1,1] for ϵᵢⱼ in ϵᵢⱼ_vec]

## plot time series
plot(t_vec,σ₁₁_vec)
plot(t_vec,ϵ₁₁_vec)
plot(t_vec,D_vec)

## plot variable Δt
plot(diff(t_vec))

## plot stress-strain along x with options on plot
plot(-ϵ₁₁_vec,-σ₁₁_vec)
title!("-σ₁₁ = f(-ϵ₁₁)")
xlabel!("-ϵ₁₁")
ylabel!("-σ₁₁ (MPa)")
xlims!((0,maximum(-ϵ₁₁_vec)))
ylims!((0,maximum(-σ₁₁_vec)))

