
function set_plane_strain_oop_stress(σᵢⱼ,r,D ; abstol=1e-12, maxiter=100, σoop_guess=nothing)
  # If the initial out of plane is different from σ₃₃ insert a guess
  !isnothing(σoop_guess) && (σᵢⱼ = insert_σoop(σᵢⱼ,σoop_guess))
  σoop = σᵢⱼ[3,3]
  # get damage constants:
  if r.D₀ == 0
    A1,B1 = 0.0, 0.0
  else
    A1,B1 = compute_A1B1(r,D)
  end
  for i in 1:maxiter
    # Joint computation of value and grad of ϵ_oop
    # result = GradientResult(σᵢⱼ)
    # gradient!(result, σᵢⱼ -> compute_ϵ_oop(r,A1,B1,σᵢⱼ), σᵢⱼ)
    # dϵ_oop = DiffResults.gradient(result)[3,3] # keep only derivative wrt σoop
    # ϵ_oop = value(result)
    dϵ_oop, ϵ_oop = Tensors.gradient(σᵢⱼ -> compute_ϵ_oop(r,A1,B1,σᵢⱼ), σᵢⱼ, :all)
    dϵ_oop = dϵ_oop[3,3]
    # Newton update
    σoop = σoop - ϵ_oop/dϵ_oop
    σᵢⱼ = insert_σoop(σᵢⱼ,σoop)
    # exit condition
    (abs(ϵ_oop) <= abstol) && break
  end
  return σᵢⱼ
end

function insert_σoop(σᵢⱼ::Tensor{2,3},σoop_guess)
  σᵢⱼ = Tensor{2,3}([σᵢⱼ[1,1] σᵢⱼ[1,2] σᵢⱼ[1,3] ; σᵢⱼ[2,1] σᵢⱼ[2,2] σᵢⱼ[2,3] ; σᵢⱼ[3,1] σᵢⱼ[3,2] σoop_guess])
end
function insert_σoop(σᵢⱼ::SymmetricTensor{2,3},σoop_guess)
  σᵢⱼ = SymmetricTensor{2,3}([σᵢⱼ[1,1] σᵢⱼ[1,2] σᵢⱼ[1,3] ; σᵢⱼ[2,1] σᵢⱼ[2,2] σᵢⱼ[2,3] ; σᵢⱼ[3,1] σᵢⱼ[3,2] σoop_guess])
end

function get_damage_onset(r::Rheology,σ₃,D)
  func_to_minimize = S -> abs(zero_at_damage_onset(r,first(S),σ₃,D)) #optimize function needs an array of variables
  Sc = optimize(func_to_minimize, [3.0],method=LBFGS(),autodiff = :forward).minimizer[1]
  return Sc
end
get_damage_onset(r::Rheology,σ₃) = get_damage_onset(r,σ₃,r.D₀)

function zero_at_damage_onset(r,S,σ₃,D)
  σᵢⱼ = build_principal_stress_tensor(r,S,σ₃,D)
  σ = 1/3*tr(σᵢⱼ)
  sᵢⱼ = dev(σᵢⱼ)
  τ = get_τ(sᵢⱼ)
  A, B = compute_AB(r,D)
  return abs(τ/σ) - A/B
end

function get_KI_mininizer_D(r,S,σ₃)
  func_to_minimize = D -> KI_from_external_load(r,S,σ₃,D) #optimize function needs an array of variables
  Dc = optimize(func_to_minimize, r.D₀, 0.99,rel_tol=1e-10).minimizer[1]
  return Dc
end

function KI_from_external_load(r,S,σ₃,D)
  σij = build_principal_stress_tensor(r,S,σ₃,D)
  KI = compute_KI(r,σij,D)
  return KI
end

function build_principal_stress_tensor(r,S,σ₃ ; abstol=1e-15)
  σᵢⱼ_prev = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_prev,r,r.D₀ ; abstol)
end

function build_principal_stress_tensor(r,S,σ₃,D ; abstol=1e-15)
  σᵢⱼ_prev = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_prev,r,D ; abstol)
end

# function build_stress_tensor(r,S,σ₃,D)
#   A1, B1 = compute_A1B1(r,D)
#   Γ = compute_Γ(r,A1,B1)
#   σᵢⱼ_prev = compute_σij(r,A1,B1,Γ,ϵij)
#   return set_plane_strain_oop_stress(σᵢⱼ_prev,r,r.D₀)
# end

function change_coords(σᵢⱼ,θ)
  Q = Tensor{2,2}([sind(θ) cosd(θ) ; -cosd(θ) sind(θ)])
  σ2D = Tensor{2,2}(σᵢⱼ[1:2,1:2])
  σ2D_rotated = Q ⋅ σ2D ⋅ Q'
  σᵢⱼ_rotated = Tensor{2,3}([σ2D_rotated zeros(eltype(σᵢⱼ),2) ; [zero(eltype(σᵢⱼ)) zero(eltype(σᵢⱼ)) σᵢⱼ[3,3]]])
  return σᵢⱼ_rotated
end

## Solve related functions :


# function solve(r,σᵢⱼ_in,ϵᵢⱼ_in,knowns,unknowns,D,Δt ; abstol=1e-12, maxiter=100)
#   Ḋ = zero(D)
#   ΔD = zero(Ḋ)
#   ∇res = similar(knowns)
#   res = similar(knowns)
#   u = unknowns
#   ϵᵢⱼ_in_next = ϵᵢⱼ_in
#   σᵢⱼ_in_next = σᵢⱼ_in
#   for i in 1:maxiter
#     Ḋ = compute_subcrit_damage_rate(r, compute_KI(r,σᵢⱼ_in_next,D), D)
#     ΔD = Ḋ*Δt
#     ∇res , res = Tensors.gradient(unknowns -> residual(r, ϵᵢⱼ_in_next, D+ΔD, knowns, unknowns), u, :all)
#     δu = - ∇res\res
#     u = u + δu
#     ϵᵢⱼ_in_next = increment_ϵᵢⱼ_in(ϵᵢⱼ_in,knowns,u)
#     σᵢⱼ_in_next = increment_σᵢⱼ_in(r,σᵢⱼ_in,D+ΔD,knowns,u) 
#     (abs(res) <= abstol) && break
#     @debug "abs res = $(abs(res))"
#     (i == maxiter) && @warn("maxiter reached, residual still higher than abstol")
#   end
#   D = D + ΔD
#   return σᵢⱼ_in_next, ϵᵢⱼ_in_next, knowns, u, D
# end

# function residual(r, ϵᵢⱼ, D, knowns, unknowns)
#   Δσηη , Δσξη, Δϵξξ = knowns
#   Δσξξ, Δϵηη, Δϵξη = unknowns
#   Δϵ_in = SymmetricTensor{2,3}([Δϵξξ Δϵξη  0 ; 
#                                 Δϵξη Δϵηη  0 ;
#                                  0    0    0 ])
#   # Δσ_in out of plane guess is an elastic undamaged one
#   Δσ_in_guess = SymmetricTensor{2,3}([ Δσξξ  Δσξη 0 ; 
#                                        Δσξη  Δσηη 0 ;
#                                         0     0   r.ν*(Δσξξ + Δσηη) ])
#   Δσ_in = set_plane_strain_oop_stress(Δσ_in_guess, r, D)
#   # staggered stiffness tensor
#   C = compute_damaged_stiffness_tensor(r,(2*ϵᵢⱼ+Δϵ_in)/2,D)
#   # stress from stiffness tensor and deformation
#   Δσ_in_from_ϵ = C ⊡ Δϵ_in
#   ΔΔσ = Δσ_in - Δσ_in_from_ϵ
#   return Vec(ΔΔσ[1,1], ΔΔσ[2,2], ΔΔσ[3,3])
#   #return norm(Δσ_in[1,2] - Δσ_in_from_ϵ[1,2], Δσ_in[2,2] - Δσ_in_from_ϵ[2,2])
# end

# function increment_ϵᵢⱼ_in(ϵᵢⱼ,knowns,unknowns) 
#   _ , _, Δϵξξ = knowns
#   _, Δϵηη, Δϵξη = unknowns
#   Δϵᵢⱼ = SymmetricTensor{2,3}([Δϵξξ Δϵξη 0 ; Δϵξη Δϵηη 0 ; 0 0 0])
#   ϵᵢⱼ_next = ϵᵢⱼ + Δϵᵢⱼ
#   return ϵᵢⱼ_next
# end

# function increment_σᵢⱼ_in(r,σᵢⱼ,D,knowns,unknowns) 
#   Δσηη , Δσξη, _ = knowns
#   Δσξξ, _, _ = unknowns
#   Δσᵢⱼ = SymmetricTensor{2,3}([Δσξξ Δσξη 0 ; Δσξη Δσξξ 0 ; 0 0 r.ν*(Δσξξ+Δσξξ) ])
#   σᵢⱼ_next = set_plane_strain_oop_stress(σᵢⱼ + Δσᵢⱼ, r, D)
#   return σᵢⱼ_next
# end

#####################
## Second strategy ##
#####################
function adaptative_time_integration(r::Rheology,p::Params,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan)
  #unpack
  time_maxiter = p.solver.time_maxiter

  # initializations
  t_vec = Float64[tspan[1]]
  tsim = tspan[1]
  σᵢⱼ_vec = SymmetricTensor{2,3}[σᵢⱼ_i]
  ϵᵢⱼ_vec = SymmetricTensor{2,3}[ϵᵢⱼ_i]
  D_vec = Float64[D_i]
  σᵢⱼnext = σᵢⱼ_i
  ϵᵢⱼnext = ϵᵢⱼ_i
  Dnext = D_i
  Δt_next = Δt
  last_tsim_printed = 0.0
  last_tsim_saved = 0.0 
  i = 1 # iter counter
  while tsim < tspan[2]
    print_flag, last_tsim_printed = get_print_flag(p,i,tsim,last_tsim_printed)
    print_flag && print_time_iteration(i,tsim)

    σᵢⱼnext, ϵᵢⱼnext, Dnext, Δt_used, Δt_next = adaptative_Δt_solve(r,p,σᵢⱼnext,ϵᵢⱼnext,Dnext,ϵ̇11,Δt_next)

    tsim += Δt_used

    save_flag, last_tsim_saved = get_save_flag(p,i,tsim,last_tsim_saved)
    if save_flag
      push!(σᵢⱼ_vec,σᵢⱼnext)
      push!(ϵᵢⱼ_vec,ϵᵢⱼnext)
      push!(D_vec,Dnext)
      push!(t_vec,tsim)
    end

    i += 1
    if !isnothing(time_maxiter)
      (length(t_vec)==time_maxiter+1) && break
    end
  end
  return t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec
end



function adaptative_Δt_solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt)
  e₀ = p.solver.e₀

  σᵢⱼnext1, ϵᵢⱼnext1, Dnext1, u1 = solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt)
  σᵢⱼmid, ϵᵢⱼmid, Dmid, umid = solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt/2)
  σᵢⱼnext2, ϵᵢⱼnext2, Dnext2, u2 = solve(r,p,σᵢⱼmid,ϵᵢⱼmid,Dmid,ϵ̇11,Δt/2)

  # compute errors for each unknowns physical quantity and ponderate 
  eD = Dnext2-Dnext1
  eσ = norm((u2-u1)[1:2])
  eϵ = norm((u2-u1)[3])
  if e₀ isa Real
    e₀ref = e₀
    e_normalized = (eD, eσ/r.G, eϵ)
    ok_flag = all(e_normalized.<e₀)m
    e = maximum(e_normalized)
  elseif e₀ isa NamedTuple
    e₀ref = e₀.D
    ok_flag = (eD<e₀.D) && (eσ<(e₀.σ)) && (eϵ<e₀.ϵ)
    e_normalized = (eD, eσ*(e₀ref/e₀.σ), eϵ*(e₀ref/e₀.ϵ))
    e,ind = findmax(e_normalized)
    ok_flag || @debug("maximum error comes from indice $(ind) of (D,σ,ϵ)")
  end

  if ok_flag
    # increse timestep
    Δt_next = min(Δt*abs(e₀ref/e),Δt*2)
    # keep best solution
    return σᵢⱼnext2, ϵᵢⱼnext2, Dnext2, Δt, Δt_next
  else
    # recursively run with decreased timestep
    adaptative_Δt_solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11, Δt*abs(e₀ref/e)) 
    #initialy Δt*abs(e₀ref/e)^2 but without the square seems to generaly require less iterations.
  end

end

function time_integration(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan)
  t_vec = tspan[1]:Δt:tspan[2]
  σᵢⱼ_vec = Vector{SymmetricTensor{2,3}}(undef,length(t_vec))
  ϵᵢⱼ_vec = similar(σᵢⱼ_vec)
  D_vec = Vector{Float64}(undef,length(t_vec))
  σᵢⱼ_vec[1] = σᵢⱼ_i
  ϵᵢⱼ_vec[1] = ϵᵢⱼ_i
  D_vec[1] = D_i
  last_tsim_printed = 0.0
  for i in 2:length(t_vec)
    print_flag, last_tsim_printed = get_print_flag(p,i,t_vec[i-1],last_tsim_printed)
    print_flag && print_time_iteration(i,t_vec[i-1])
  
    σᵢⱼ_vec[i], ϵᵢⱼ_vec[i], D_vec[i], u = solve(r,p,σᵢⱼ_vec[i-1],ϵᵢⱼ_vec[i-1],D_vec[i-1],ϵ̇11,Δt)
  end
  return t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec
end

function solve(r::Rheology,p::Params,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt)
  ps = p.solver
  D = D_i
  # get first guess of the unknowns with an elastic solve
  ϵᵢⱼnext = insert_into(ϵᵢⱼ_i, (ϵᵢⱼ_i[1,1] + ϵ̇11*Δt), (1,1))
  σᵢⱼnext = σᵢⱼ_i#σᵢⱼnext = compute_σij(r,D,ϵᵢⱼnext)
  #σᵢⱼnext = insert_into(σᵢⱼnext, -1e6, (2,2))
  u = Vec(σᵢⱼnext[1,1], σᵢⱼnext[3,3], ϵᵢⱼnext[2,2])
  #@debug "u_i = $u"
  for i in 1:ps.newton_maxiter
    # get residual and its gradient with respect to u
    ∇res , res = Tensors.gradient(u -> residual(r,D,ϵᵢⱼ_i,ϵᵢⱼnext,σᵢⱼ_i,σᵢⱼnext,Δt,u), u, :all)
    #@debug "norm res = $(norm(res))"
    #@debug "∇res = $∇res"

    # update u with Newton algo
    δu = - ∇res\res
    u = u + δu
    #@debug "δu = $δu"
    #@debug "typeof(u) = $(typeof(u))"

    (norm(res) <= ps.newton_abstol) && break
    
    (i == ps.newton_maxiter) && @debug("ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
  end
  # update ϵᵢⱼnext and σᵢⱼnext with converged u
  ϵᵢⱼnext = insert_into(ϵᵢⱼnext, u[3], (2,2)) 
  σᵢⱼnext = insert_into(σᵢⱼnext, (u[1], u[2]), ((1,1),(3,3)))
  _ , Ḋ = compute_ϵ̇ij(r,D,σᵢⱼ_i,σᵢⱼnext,Δt)
  D = D + Ḋ*Δt
  return σᵢⱼnext, ϵᵢⱼnext, D, u
end

function residual(r,D,ϵij,ϵijnext,σij,σijnext,Δt,u)
  σ11next, σ33next, ϵ22next = u
  σijnext = insert_into(σijnext, (σ11next, σ33next), ((1,1),(3,3)))
  ϵijnext = insert_into(ϵijnext, ϵ22next, (2,2))
  ϵ̇ij_analytical, _ = compute_ϵ̇ij(r,D,σij,σijnext,Δt)
  ϵ̇ij = (ϵijnext - ϵij)/Δt
  Δϵ̇ij = ϵ̇ij_analytical - ϵ̇ij
  #@debug " KI = $(compute_KI(r,σijnext, D))"
  return Vec(Δϵ̇ij[1,1],Δϵ̇ij[2,2],Δϵ̇ij[3,3]) 
end

function insert_into(tensor::SymmetricTensor{2,S},values,indices) where{S}
  return SymmetricTensor{2,S}((i,j) -> map_id_to_value(tensor,values,indices,i,j))
end

function map_id_to_value(tensor::SymmetricTensor,values,indices,i,j)
  ((i,j) == indices) | ((j,i) == indices) && return values
  if (i,j) in indices
    ind = findfirst(indices .== Ref((i,j)))
    return values[only(ind)]
  elseif (j,i) in indices
    ind = findfirst(indices .== Ref((j,i)))
    return values[only(ind)]
  else
    return tensor[i,j]
  end
end

function get_print_flag(p,iter,tsim,last_tsim_printed)
  if (iter==1) | print_flag_condition(p,iter,tsim,last_tsim_printed)
    print_flag = true
    last_tsim_printed = tsim
  else 
    print_flag = false
  end
  return print_flag, last_tsim_printed
end

function get_save_flag(p,iter,tsim,last_tsim_saved)
  if (iter==1) | save_flag_condition(p,iter,tsim,last_tsim_saved)
    save_flag = true
    last_tsim_saved = tsim
  else 
    save_flag = false
  end
  return save_flag, last_tsim_saved
end

print_flag_condition(p::Params{TF,Nothing},iter,tsim,last_tsim) where {TF<:Real} = ( (tsim-last_tsim) >= (1/p.output.print_frequency) )
print_flag_condition(p::Params{Nothing,TP},iter,tsim,last_tsim) where {TP<:Real} = (iter%p.output.print_period == 0)
function print_flag_condition(p::Params{TF,TP},iter,tsim,last_tsim) where {TF<:Real,TP<:Real}
  return ( (tsim-last_tsim) >= (1/p.output.print_frequency) ) || (iter%p.output.print_period == 0)
end

save_flag_condition(p::Params{T1,T2,TF,Nothing},iter,tsim,last_tsim) where {T1,T2,TF<:Real} = ( (tsim-last_tsim) >= (1/p.output.save_frequency) )
save_flag_condition(p::Params{T1,T2,Nothing,TP},iter,tsim,last_tsim) where {T1,T2,TP<:Real} = (iter%p.output.save_period == 0)
function save_flag_condition(p::Params{T1,T2,TF,TP},iter,tsim,last_tsim) where {T1,T2,TF<:Real,TP<:Real}
  return ( (tsim-last_tsim) >= (1/p.output.save_frequency) ) || (iter%p.output.save_period == 0)
end

print_time_iteration(iter,tsim) = print("------","\n","time iteration $iter : $tsim","\n","------")