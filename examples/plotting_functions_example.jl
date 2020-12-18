@time using DamagedShearBand
const DSB = DamagedShearBand
using LaTeXStrings
@time using Plots


function plot_simulations(r_vec,t_vec_vec, σᵢⱼ_vec_vec, ϵᵢⱼ_vec_vec, D_vec_vec, title ; todisk=false, path=nothing, figsize=(400,1000),backend=pyplot)
  # extract σ₁₁ and ϵ₁₁
  backend(size=figsize)
  layout = @layout [a ; b ; c]
  plotids = 0
  for i in eachindex(t_vec_vec)
    # get raw time series
    r = r_vec[i]
    t_vec = t_vec_vec[i]
    D_vec = D_vec_vec[i]
    σᵢⱼ_vec = σᵢⱼ_vec_vec[i]
    ϵᵢⱼ_vec = ϵᵢⱼ_vec_vec[i]
    
    # extract relevant time series
    σ₁₁_vec = [σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec]
    ϵ₁₁_vec = [ϵᵢⱼ[1,1] for ϵᵢⱼ in ϵᵢⱼ_vec]
    ϵₖₖ_vec = [tr(ϵᵢⱼ) for ϵᵢⱼ in ϵᵢⱼ_vec]
    #KInorm_vec = [DSB.compute_KI(r,σᵢⱼ_vec[i], D_vec[i])/r.K₁c for i in eachindex(σᵢⱼ_vec)]

    if i == 1
      plotids = plot_first_simulation(σ₁₁_vec, ϵ₁₁_vec, ϵₖₖ_vec, D_vec)
    else
      plotids = add_data_to_plot!(plotids,σ₁₁_vec, ϵ₁₁_vec, ϵₖₖ_vec, D_vec)
    end
  end
  plt = assemble_plot(plotids,layout,title)

  if todisk
    if isnothing(path)
      savefig(plt,joinpath(@__DIR__,title,".pdf"))
    else
      savefig(plt,path)
    end
  end

  return plt
end

function plot_first_simulation(σ₁₁_vec, ϵ₁₁_vec, ϵₖₖ_vec, D_vec)
  plt1 = plot(-ϵ₁₁_vec.*100,-σ₁₁_vec.*1e-6,label=nothing)
  plt2 = plot(-ϵ₁₁_vec.*100,D_vec,label=nothing)
  #plot!(-ϵ₁₁_vec.*100,KInorm_vec)
  plt3 = plot(-ϵ₁₁_vec.*100,ϵₖₖ_vec,label=nothing)
  return plt1, plt2, plt3
end

function add_data_to_plot!(plotids,σ₁₁_vec, ϵ₁₁_vec, ϵₖₖ_vec, D_vec)
  plt1, plt2, plt3 = plotids
  plot!(plt1,-ϵ₁₁_vec.*100,-σ₁₁_vec.*1e-6,label=nothing)
  plot!(plt2,-ϵ₁₁_vec.*100,D_vec,label=nothing)
  #plot!(plt2,-ϵ₁₁_vec,KInorm_vec)
  plot!(plt3,-ϵ₁₁_vec.*100,ϵₖₖ_vec,label=nothing)
  return plt1, plt2, plt3
end

function assemble_plot(plotids,layout,title)
  plt1, plt2, plt3 = plotids
  ylabel!(plt1,"-σ₁₁ (MPa)")
  title!(plt1,title)
  ylabel!(plt2,"Damage")
  ylabel!(plt3,"Dilatancy (ϵₖₖ)")
  xlabel!(plt3,"-ϵ₁₁(%)")
  plt = plot(plt1, plt2, plt3,layout=layout)
  return plt
end



################
## Simulation ##
################
ϵ₁₁_max = -0.05
ϵ̇₁₁ = -1e-6
σ₂₂ = -1e8
n_vec = Float64[5,10,15,20,25,30,35] 
D₀ = 0.2
r = [DSB.Rheology(;n=n) for n in n_vec] 

tspan = (0.0,ϵ₁₁_max/ϵ̇₁₁) # simulate up to ϵ₁₁_max strain
n_iter_saved = 5000
n_iter_printed = 10
op = DSB.OutputParams(save_frequency = n_iter_saved/(tspan[2]-tspan[1]), # save approximately n_iter_saved points per simulation
                      save_period = nothing,
                      print_frequency = n_iter_printed/(tspan[2]-tspan[1]),
                      print_period = nothing)

sp = DSB.SolverParams(newton_abstol = 1e-12,
                      newton_maxiter = 100,
                      time_maxiter = nothing,
                      e₀ = (D=1e-1, σ=100.0, ϵ=1e-3)) 

p = DSB.Params(op,sp)

## run_simulations function accepts one of {r,D₀,σ₂₂} as a Vector in order to run one simulation per vector element.
@time t_vec_vec, σᵢⱼ_vec_vec, ϵᵢⱼ_vec_vec, D_vec_vec, r_vec = DSB.run_simulations(r,p,D₀,σ₂₂,ϵ̇₁₁,tspan ; multithreaded=true)

## Title and path for our figure
title = "ϵ̇₁₁=$(ϵ̇₁₁)_σ₂₂=$(σ₂₂)_n=$(n_vec)_D0=$(D₀)"
path = joinpath(@__DIR__,"figures",title*".pdf")

# This function plots vectors of time series generated with run_simulations function, and saves plot if needed.
fig = plot_simulations(r_vec, t_vec_vec, σᵢⱼ_vec_vec, ϵᵢⱼ_vec_vec, D_vec_vec, title ; todisk=false, path=path, figsize=(1000,1000))
