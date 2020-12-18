import Base.Threads: @spawn, @threads

function run_simulations(r::Rheology,p::Params,D₀::Vector,σ₂₂::Real,ϵ̇₁₁::Real,tspan ; multithreaded=true)
  # Vector variable
  vec_var = D₀
  
  # initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
  S = 1.0

  # Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
  r_vec = [Rheology(r,(D₀=D0,)) for D0 in D₀]

  # time integration parameters :
  Δt_i = 0.0001 # initial timestep

  # initialize containers
  t_vec_vec = Vector{Vector{Float64}}(undef,length(vec_var))
  σᵢⱼ_vec_vec = Vector{Vector{SymmetricTensor{2,3}}}(undef,length(vec_var))
  ϵᵢⱼ_vec_vec = Vector{Vector{SymmetricTensor{2,3}}}(undef,length(vec_var))
  D_vec_vec =  Vector{Vector{Float64}}(undef,length(vec_var))

  # Build initial stress and strain tensors 
  σᵢⱼ_i_vec = [build_principal_stress_tensor(r_vec[i],S,σ₂₂,D₀[i] ; abstol=1e-15) for i in eachindex(vec_var)]
  ϵᵢⱼ_i_vec = [compute_ϵij(r_vec[i],D₀[i],σᵢⱼ_i_vec[i]) for i in eachindex(vec_var)]

  if multithreaded
    @threads for i in eachindex(vec_var)
      # solve
      t_vec_vec[i], σᵢⱼ_vec_vec[i], ϵᵢⱼ_vec_vec[i], D_vec_vec[i] = adaptative_time_integration(r_vec[i],p,σᵢⱼ_i_vec[i],ϵᵢⱼ_i_vec[i],D₀[i],ϵ̇₁₁,Δt_i,tspan)
      
    end
  else
    for i in eachindex(vec_var)
      # solve
      t_vec_vec[i], σᵢⱼ_vec_vec[i], ϵᵢⱼ_vec_vec[i], D_vec_vec[i] = adaptative_time_integration(r_vec[i],p,σᵢⱼ_i_vec[i],ϵᵢⱼ_i_vec[i],D₀[i],ϵ̇₁₁,Δt_i,tspan)
      
    end
  end
  return t_vec_vec, σᵢⱼ_vec_vec, ϵᵢⱼ_vec_vec, D_vec_vec, r_vec
end

function run_simulations(r::Rheology,p::Params,D₀::Real,σ₂₂::Vector,ϵ̇₁₁::Real,tspan ; multithreaded=true)
  # Vector variable
  vec_var = σ₂₂
  
  # initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
  S = 1.0

  # Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
  r_vec = [Rheology(r,(D₀=D₀,)) for i in eachindex(vec_var)]

  # time integration parameters :
  Δt_i = 0.0001 # initial timestep

  # initialize containers
  t_vec_vec = Vector{Vector{Float64}}(undef,length(vec_var))
  σᵢⱼ_vec_vec = Vector{Vector{SymmetricTensor{2,3}}}(undef,length(vec_var))
  ϵᵢⱼ_vec_vec = Vector{Vector{SymmetricTensor{2,3}}}(undef,length(vec_var))
  D_vec_vec =  Vector{Vector{Float64}}(undef,length(vec_var))

  # Build initial stress and strain tensors 
  σᵢⱼ_i_vec = [build_principal_stress_tensor(r_vec[i],S,σ₂₂[i],D₀ ; abstol=1e-15) for i in eachindex(vec_var)]
  ϵᵢⱼ_i_vec = [compute_ϵij(r_vec[i],D₀,σᵢⱼ_i_vec[i]) for i in eachindex(vec_var)]

  if multithreaded
    @threads for i in eachindex(vec_var)
      # solve
      t_vec_vec[i], σᵢⱼ_vec_vec[i], ϵᵢⱼ_vec_vec[i], D_vec_vec[i] = adaptative_time_integration(r_vec[i],p,σᵢⱼ_i_vec[i],ϵᵢⱼ_i_vec[i],D₀,ϵ̇₁₁,Δt_i,tspan)
      
    end
  else
    for i in eachindex(vec_var)
      # solve
      t_vec_vec[i], σᵢⱼ_vec_vec[i], ϵᵢⱼ_vec_vec[i], D_vec_vec[i] = adaptative_time_integration(r_vec[i],p,σᵢⱼ_i_vec[i],ϵᵢⱼ_i_vec[i],D₀,ϵ̇₁₁,Δt_i,tspan)
      
    end
  end

  return t_vec_vec, σᵢⱼ_vec_vec, ϵᵢⱼ_vec_vec, D_vec_vec, r_vec
end

function run_simulations(r::Vector,p::Params,D₀::Real,σ₂₂::Real,ϵ̇₁₁::Real,tspan ; multithreaded=true)
  # Vector variable
  vec_var = r
  
  # initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
  S = 1.0

  # Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
  r_vec = [Rheology(r[i],(D₀=D₀,)) for i in eachindex(vec_var)]

  # time integration parameters :
  Δt_i = 0.0001 # initial timestep

  # initialize containers
  t_vec_vec = Vector{Vector{Float64}}(undef,length(vec_var))
  σᵢⱼ_vec_vec = Vector{Vector{SymmetricTensor{2,3}}}(undef,length(vec_var))
  ϵᵢⱼ_vec_vec = Vector{Vector{SymmetricTensor{2,3}}}(undef,length(vec_var))
  D_vec_vec =  Vector{Vector{Float64}}(undef,length(vec_var))

  # Build initial stress and strain tensors 
  σᵢⱼ_i_vec = [build_principal_stress_tensor(r_vec[i],S,σ₂₂,D₀ ; abstol=1e-15) for i in eachindex(vec_var)]
  ϵᵢⱼ_i_vec = [compute_ϵij(r_vec[i],D₀,σᵢⱼ_i_vec[i]) for i in eachindex(vec_var)]

  if multithreaded
    @threads for i in eachindex(vec_var)
      # solve
      t_vec_vec[i], σᵢⱼ_vec_vec[i], ϵᵢⱼ_vec_vec[i], D_vec_vec[i] = adaptative_time_integration(r_vec[i],p,σᵢⱼ_i_vec[i],ϵᵢⱼ_i_vec[i],D₀,ϵ̇₁₁,Δt_i,tspan)
      
    end
  else
    for i in eachindex(vec_var)
      # solve
      t_vec_vec[i], σᵢⱼ_vec_vec[i], ϵᵢⱼ_vec_vec[i], D_vec_vec[i] = adaptative_time_integration(r_vec[i],p,σᵢⱼ_i_vec[i],ϵᵢⱼ_i_vec[i],D₀,ϵ̇₁₁,Δt_i,tspan)
      
    end
  end
  return t_vec_vec, σᵢⱼ_vec_vec, ϵᵢⱼ_vec_vec, D_vec_vec, r_vec
end
