@time using DamagedShearBand
const DSB = DamagedShearBand
@time using Plots
using Printf
##
# imposed values
ϵ̇₁₁ = -1e-5
ϵ̇ⁱξη = 1e-5
σ₃ = -5e7 # negative in compression

# initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
S_i = 1.0
θ = 60.0

# Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
r = DSB.Rheology(D₀=0.2,n=5.0) 

# initial damage
#D_i = r.D₀
Dᵒ, _ = DSB.get_KI_mininizer_D_on_S_range(r,S_i,100,σ₃)
Dⁱ = min(Dᵒ+0.15, 0.9)

# time integration parameters :
ϵⁱξη_goal = 0.01
#tspan = (0.0,ϵⁱξη_goal/ϵ̇ⁱξη) # initial and final simulation time
tspan = (0.0,500)
Δt_i = 0.01 # initial timestep

# Build stress and strain tensors 
# σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S_i,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
# ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)

n_iter_saved = 20000
n_iter_printed = 10
op = DSB.OutputParams(save_frequency = n_iter_saved/(tspan[2]-tspan[1]),
                      save_period = 1000,
                      print_frequency = n_iter_printed/(tspan[2]-tspan[1]),
                      print_period = 1000)

sp = DSB.SolverParams(newton_abstol = 1e-12,
                      newton_maxiter = 100,
                      time_maxiter = nothing,
                      e₀ = (D=1e-2, S=1e-2, σ=100.0, ϵ=1e-2))
# sp = DSB.SolverParams(newton_abstol = 1e-12,
#                       newton_maxiter = 100,
#                       time_maxiter = nothing,
#                       e₀ = (D=1e-3, S=1e-3, σ=1.0, ϵ=1e-5))

p = DSB.Params(op,sp)
# Integrate over time, by solving for the unknowns [σ₁₁next, σ₃₃next, ϵ₂₂next] at each timestep
# -- e₀ is the absolute tolerance used to adapt time stepping, may be a scalar or a NamedTuple with keys (D,σ,ϵ)
# -- abstol is the absolute tolerance on the unknowns in the Newton solver 

t_vec, S_vec, σⁱᵢⱼ_vec, σᵒᵢⱼ_vec, ϵⁱᵢⱼ_vec, Dⁱ_vec, Dᵒ_vec = DSB.adaptative_time_integration_2_points(r,p,S_i,σ₃,Dⁱ,Dᵒ,ϵ̇₁₁,ϵ̇ⁱξη,Δt_i,θ,tspan);

#
σᵒ_p_vec = DSB.principal_coords.(σᵒᵢⱼ_vec,Ref(θ))
σⁱ_p_vec = DSB.principal_coords.(σⁱᵢⱼ_vec,Ref(θ))
σᵒ₁₁_p_vec = [σᵢⱼ[1,1] for σᵢⱼ in σᵒ_p_vec]
σᵒ₂₂_p_vec = [σᵢⱼ[2,2] for σᵢⱼ in σᵒ_p_vec]
σᵒ₃₃_p_vec = [σᵢⱼ[3,3] for σᵢⱼ in σᵒ_p_vec]

σᵒ₁₁_vec = [σᵢⱼ[1,1] for σᵢⱼ in σᵒᵢⱼ_vec]
σⁱ₁₁_vec = [σᵢⱼ[1,1] for σᵢⱼ in σⁱᵢⱼ_vec]
σᵒ₁₂_vec = [σᵢⱼ[1,2] for σᵢⱼ in σᵒᵢⱼ_vec]
σⁱ₁₂_vec = [σᵢⱼ[1,2] for σᵢⱼ in σⁱᵢⱼ_vec]
σᵒ₂₁_vec = [σᵢⱼ[2,1] for σᵢⱼ in σᵒᵢⱼ_vec]
σⁱ₂₁_vec = [σᵢⱼ[2,1] for σᵢⱼ in σⁱᵢⱼ_vec]
σᵒ₂₂_vec = [σᵢⱼ[2,2] for σᵢⱼ in σᵒᵢⱼ_vec]
σⁱ₂₂_vec = [σᵢⱼ[2,2] for σᵢⱼ in σⁱᵢⱼ_vec]
σᵒ₃₃_vec = [σᵢⱼ[3,3] for σᵢⱼ in σᵒᵢⱼ_vec]
σⁱ₃₃_vec = [σᵢⱼ[3,3] for σᵢⱼ in σⁱᵢⱼ_vec]

ϵⁱ₁₁_vec = [ϵᵢⱼ[1,1] for ϵᵢⱼ in ϵⁱᵢⱼ_vec]
ϵⁱ₁₂_vec = [ϵᵢⱼ[1,2] for ϵᵢⱼ in ϵⁱᵢⱼ_vec]
ϵⁱ₂₁_vec = [ϵᵢⱼ[2,1] for ϵᵢⱼ in ϵⁱᵢⱼ_vec]
ϵⁱ₂₂_vec = [ϵᵢⱼ[2,2] for ϵᵢⱼ in ϵⁱᵢⱼ_vec]
ϵⁱ₃₃_vec = [ϵᵢⱼ[3,3] for ϵᵢⱼ in ϵⁱᵢⱼ_vec]

τⁱ_vec =  DSB.get_τ.(σⁱᵢⱼ_vec; input=:σ)
τᵒ_vec =  DSB.get_τ.(σᵒᵢⱼ_vec; input=:σ)
pⁱ_vec = -1/3 * tr.(σⁱᵢⱼ_vec)
pᵒ_vec = -1/3 * tr.(σⁱᵢⱼ_vec)
KIⁱ_vec = [DSB.compute_KI(r,σⁱᵢⱼ_vec[i], Dⁱ_vec[i]) for i in eachindex(σⁱᵢⱼ_vec)]
KIᵒ_vec = [DSB.compute_KI(r,σᵒᵢⱼ_vec[i], Dᵒ_vec[i]) for i in eachindex(σᵒᵢⱼ_vec)]
γⁱ_vec = [ sqrt(2*ϵᵢⱼ⊡ϵᵢⱼ) for ϵᵢⱼ in ϵⁱᵢⱼ_vec]
ϵⁱkk_vec = [tr(ϵᵢⱼ) for ϵᵢⱼ in ϵⁱᵢⱼ_vec]
deviation_out_vec = DSB.get_stress_deviation_from_far_field.(σᵒ_p_vec ; offdiagtol=1e-5)
deviation_in_vec = DSB.get_stress_deviation_from_far_field.(σⁱ_p_vec ; offdiagtol=1e-5)

σᵒᵢⱼ_i_p, σⁱᵢⱼ_i_p, ϵᵒᵢⱼ_i_p, ϵⁱᵢⱼ_i_p = DSB.initialize_state_var_D(r,p,S_i,σ₃,Dⁱ,Dᵒ,θ ; coords=:principal)
σᵒᵢⱼ_i, σⁱᵢⱼ_i, ϵᵒᵢⱼ_i, ϵⁱᵢⱼ_i = DSB.initialize_state_var_D(r,p,S_i,σ₃,Dⁱ,Dᵒ,θ ; coords=:band)
deviation_out_check = DSB.get_stress_deviation_from_far_field(σᵒᵢⱼ_i_p ; offdiagtol=1e-12)
deviation_in_check = DSB.get_stress_deviation_from_far_field(σⁱᵢⱼ_i_p ; offdiagtol=1e-12)
deviation_out_eps = DSB.get_stress_deviation_from_far_field(ϵᵒᵢⱼ_i_p ; offdiagtol=1e-12)
deviation_in_eps = DSB.get_stress_deviation_from_far_field(ϵⁱᵢⱼ_i_p ; offdiagtol=1e-12)

deviation_out_check = DSB.get_stress_deviation_from_far_field(DSB.principal_coords(σᵒᵢⱼ_i,θ))
deviation_in_check = DSB.get_stress_deviation_from_far_field(DSB.principal_coords(σⁱᵢⱼ_i,θ))
deviation_out_eps = DSB.get_stress_deviation_from_far_field(DSB.principal_coords(ϵᵒᵢⱼ_i,θ))
deviation_in_eps = DSB.get_stress_deviation_from_far_field(DSB.principal_coords(ϵⁱᵢⱼ_i,θ))
## Plots ##
# figure size and backend
plotlyjs(size=(1000,300))

# S vs time
plot(t_vec,γⁱ_vec)
plot(t_vec,ϵⁱ₁₂_vec)
plot(t_vec,S_vec)
plot(ϵⁱ₁₂_vec,S_vec)
# principal stress orientation deviation from far field vs time
plot(t_vec,deviation_in_vec)
plot!(t_vec,deviation_out_vec)
#plot!(t_vec,-(Dⁱ_vec .- Dⁱ).*60)

plot(deviation_in_vec)
# Damage vs time
plot(t_vec,Dⁱ_vec)
plot!(t_vec,Dᵒ_vec)
# KI/KIC vs time
plot(t_vec,KIⁱ_vec./r.K₁c)
plot!(t_vec,KIᵒ_vec./r.K₁c)
# shear stress on band vs time
plot(t_vec,σⁱ₁₂_vec)
plot!(t_vec,σᵒ₁₂_vec)
# vol/shear strain invariants vs time
plot(t_vec, ϵⁱkk_vec./γⁱ_vec)
plot(t_vec[1:end-1], diff(ϵⁱkk_vec)./diff(γⁱ_vec))

# plot τ in and out
plot(t_vec, τⁱ_vec)
plot!(t_vec, τᵒ_vec)
## composite plot :
#lotlyjs(size=(1000,600))
pyplot(size=(1000,600))
layout = @layout [a ; b ; c]

p1 = plot(ϵⁱ₁₂_vec.*100,S_vec,label="S")
title!(p1,"D₀ = $(r.D₀) & D_in/out = ("*@sprintf("%.2f",Dⁱ)*", "*@sprintf("%.2f",Dᵒ)*") & gammadot = $(2*ϵ̇ⁱξη) & p=$(-σ₃)")
ylabel!("σᵒᵘᵗ₁ multiplier (S)")

p2 = plot(ϵⁱ₁₂_vec.*100,Dⁱ_vec,label="inside")
plot!(p2,γⁱ_vec.*100,Dᵒ_vec,label="outside")
ylabel!("Damage")

p3 = plot(ϵⁱ₁₂_vec.*100,deviation_in_vec,label="inside")
plot!(p3,ϵⁱ₁₂_vec.*100,deviation_out_vec,label="outside")
xlabel!("ϵⁱ₁₂ (%)")
ylabel!("stress orientation in band")

pl = plot(p1,p2,p3,layout=layout)

#savefig(pl,"./examples/figures/bifurcation_D0=$(r.D₀)_p=$(-σ₃)_ϵ̇ⁱξη=$(ϵ̇ⁱξη).pdf")