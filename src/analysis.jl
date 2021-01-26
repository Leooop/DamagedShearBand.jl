
function set_plane_strain_oop_stress(σᵢⱼ,r,D ; abstol=1e-12, maxiter=100, σoop_guess=nothing)
  # If the initial out of plane is different from σ₃₃ insert a guess
  !isnothing(σoop_guess) && (σᵢⱼ = insert_σoop(σᵢⱼ,σoop_guess))
  σoop = σᵢⱼ[3,3]
  # get damage constants:
  A1,B1 = compute_A1B1(r,D)

  for i in 1:maxiter
    
    dϵ_oop, ϵ_oop = Tensors.gradient(σᵢⱼ -> compute_ϵ_oop(r,A1,B1,σᵢⱼ), σᵢⱼ, :all)
    #println(dϵ_oop)
    dϵ_oop = dϵ_oop[3,3]
    # Newton update
    σoop = σoop - ϵ_oop/dϵ_oop
    σᵢⱼ = insert_σoop(σᵢⱼ,σoop)
    # exit condition
    (abs(ϵ_oop) <= abstol) && break
  end
  return σᵢⱼ
end

function set_plane_strain_oop_stress_rate(σᵢⱼ,σ̇ᵢⱼ,r,D,Δt ; abstol=1e-16, maxiter=100, damaged_allowed=true)
  # If the initial out of plane is different from σ₃₃ insert a guess
  #!isnothing(σoop_guess) && (σᵢⱼ = insert_σoop(σᵢⱼ,σoop_guess)) # Removed because of type inference issues
  σoop = σᵢⱼ[3,3]
  σoop_rate = σ̇ᵢⱼ[3,3]
  σᵢⱼnext = σᵢⱼ + σ̇ᵢⱼ*Δt
  # get damage constants:
  A1,B1 = compute_A1B1(r,D)
  # preallocate differentiation results
  result = DiffResults.GradientResult(σᵢⱼnext) # careful, result stores values in Matrix not Tensor
  for i in 1:maxiter
    # Joint computation of value and grad of ϵ_oop
    ForwardDiff.gradient!(result, σᵢⱼnext -> compute_ϵ̇ij(r,D,σᵢⱼ,σᵢⱼnext,Δt ; damaged_allowed)[1][3,3], σᵢⱼnext)
    dϵ̇_oop = DiffResults.gradient(result)
    ϵ̇_oop  = DiffResults.value(result)
    #println(dϵ̇_oop)
    dϵ̇_oop_dσ̇_oop = dϵ̇_oop[3,3]
    # Newton update
    σoop_rate = σoop_rate - ϵ̇_oop/dϵ̇_oop_dσ̇_oop
    σᵢⱼnext = insert_σoop(σᵢⱼnext,σoop_rate)
    # exit condition
    (abs(ϵ̇_oop) <= abstol) && break
  end
  σ̇ᵢⱼ = (σᵢⱼnext - σᵢⱼ)/Δt
  return σ̇ᵢⱼ
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

function get_KI_minimizer_S(r,D,σ₃)
  func_to_minimize = S -> abs(zero_at_damage_onset(r,S,σ₃,D)) #optimize function needs an array of variables
  Sc = optimize(func_to_minimize, 1, 1e4,rel_tol=1e-15).minimizer[1]
  return Sc
end

#tested
function get_KI_minimizer_D_S(r,σ₃)
  func_to_minimize = D -> -get_KI_minimizer_S(r,D,σ₃)
  Dc = optimize(func_to_minimize, r.D₀, 0.999, rel_tol=1e-15).minimizer[1]
  Sc = get_KI_minimizer_S(r,Dc,σ₃)
  return Dc, Sc 
end

function get_KI_minimizer_D(r,S,σ₃)
  func_to_minimize = D -> KI_from_external_load(r,S,σ₃,D) #optimize function needs an array of variables
  Dc = optimize(func_to_minimize, r.D₀, 0.999, rel_tol=1e-10).minimizer[1]
  return Dc
end

function get_KI_minimizer_D_on_S_range(r,Smin,Smax,σ₃ ; len=1000)
  S_vec = range(Smin,Smax ; length=len)
  D_vec = range(r.D₀,0.999 ; length=len)
  σᵢⱼ_mat = [build_principal_stress_tensor(r,S,σ₃,D) for S in S_vec, D in D_vec]
  D_mat = repeat(D_vec',outer=(length(S_vec),1))
  KI_mat = compute_KI.(Ref(r),σᵢⱼ_mat,D_mat)
  Dc = 0.0
  S = 0.0
  for iS in reverse(axes(S_vec,1))
    valmin, idmin = findmin(KI_mat[iS,:])
    if valmin > 0
      continue
    else
      S = S_vec[iS]
      Dc = get_KI_minimizer_D(r,S,σ₃)
      break
    end
  end
  return Dc, S
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

function band_coords(σᵢⱼ,θ)
  Q = Tensor{2,2}([sind(θ) cosd(θ) ; -cosd(θ) sind(θ)])
  σ2D = Tensor{2,2,eltype(σᵢⱼ)}(σᵢⱼ[1:2,1:2])
  σ2D_rotated = Q ⋅ σ2D ⋅ Q'
  σᵢⱼ_rotated = SymmetricTensor{2,3,eltype(σᵢⱼ)}((σ2D_rotated[1,1],σ2D_rotated[2,1],σᵢⱼ[3,1],σ2D_rotated[2,2],σᵢⱼ[3,2],σᵢⱼ[3,3]))
  #σᵢⱼ_rotated = Tensor{2,3}([σ2D_rotated zeros(eltype(σᵢⱼ),2) ; [zero(eltype(σᵢⱼ)) zero(eltype(σᵢⱼ)) σᵢⱼ[3,3]]])
  return σᵢⱼ_rotated
end

function principal_coords(σᵢⱼ,θ)
  Q = Tensor{2,2}([sind(θ) -cosd(θ) ; cosd(θ) sind(θ)])
  σ2D = Tensor{2,2,eltype(σᵢⱼ)}(σᵢⱼ[1:2,1:2])
  σ2D_rotated = Q ⋅ σ2D ⋅ Q'
  σᵢⱼ_rotated = SymmetricTensor{2,3,eltype(σᵢⱼ)}((σ2D_rotated[1,1],σ2D_rotated[2,1],σᵢⱼ[3,1],σ2D_rotated[2,2],σᵢⱼ[3,2],σᵢⱼ[3,3]))
  #σᵢⱼ_rotated = Tensor{2,3}([σ2D_rotated zeros(eltype(σᵢⱼ),2) ; [zero(eltype(σᵢⱼ)) zero(eltype(σᵢⱼ)) σᵢⱼ[3,3]]])
  return σᵢⱼ_rotated
end

function get_stress_deviation_from_far_field(σᵢⱼ ; offdiagtol=1e-5, compression_axis=:x)
  σᵢⱼ = filter_offdiagonal(σᵢⱼ ; tol=offdiagtol)
  F = eigen(σᵢⱼ[1:2,1:2])
  #any(isnan.([F.values F.vectors])) && (F = eigen(Array(σᵢⱼ)))
  ind = findall(==(minimum(F.values)),F.values)
  σ₁_xy_direction = length(ind) == 1 ? F.vectors[:,ind] : Vec(1.0,0.0)
  σ₁_far_field_direction = (compression_axis==:x) ? Vec(1.0,0.0) : Vec(0.0,1.0)
  if all(σ₁_xy_direction .≈ σ₁_far_field_direction)
      angle = 0.0
      else
      angle = atand(σ₁_xy_direction[2],σ₁_xy_direction[1])
  end
  #(angle == 90) && (angle = zero(angle))
  (angle > 90) && (angle -= 180)
  (angle < -90) && (angle += 180)
  return angle
end

#############################
## Solve related functions ##
#############################

# function adaptative_time_integration(r::Rheology,p::Params,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan)
#   #unpack
#   time_maxiter = p.solver.time_maxiter

#   # initializations
#   t_vec = Float64[tspan[1]]
#   tsim = tspan[1]
#   σᵢⱼ_vec = SymmetricTensor{2,3}[σᵢⱼ_i]
#   ϵᵢⱼ_vec = SymmetricTensor{2,3}[ϵᵢⱼ_i]
#   D_vec = Float64[D_i]
#   σᵢⱼnext = σᵢⱼ_i
#   ϵᵢⱼnext = ϵᵢⱼ_i
#   Dnext = D_i
#   Δt_next = Δt
#   last_tsim_printed = 0.0
#   last_tsim_saved = 0.0
#   i = 1 # iter counter
#   while tsim < tspan[2]
#     print_flag, last_tsim_printed = get_print_flag(p,i,tsim,last_tsim_printed)
#     print_flag && print_time_iteration(i,tsim)

#     σᵢⱼnext, ϵᵢⱼnext, Dnext, Δt_used, Δt_next = adaptative_Δt_solve(r,p,σᵢⱼnext,ϵᵢⱼnext,Dnext,ϵ̇11,Δt_next)

#     tsim += Δt_used

#     save_flag, last_tsim_saved = get_save_flag(p,i,tsim,last_tsim_saved)
#     if save_flag
#       push!(σᵢⱼ_vec,σᵢⱼnext)
#       push!(ϵᵢⱼ_vec,ϵᵢⱼnext)
#       push!(D_vec,Dnext)
#       push!(t_vec,tsim)
#     end

#     i += 1
#     if !isnothing(time_maxiter)
#       (length(t_vec)==time_maxiter+1) && break
#     end
#   end
#   return t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec
# end



# function adaptative_Δt_solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt)
#   e₀ = p.solver.e₀

#   σᵢⱼnext1, ϵᵢⱼnext1, Dnext1, u1 = solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt)
#   σᵢⱼmid, ϵᵢⱼmid, Dmid, umid = solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt/2)
#   σᵢⱼnext2, ϵᵢⱼnext2, Dnext2, u2 = solve(r,p,σᵢⱼmid,ϵᵢⱼmid,Dmid,ϵ̇11,Δt/2)

#   # compute errors for each unknowns physical quantity and ponderate 
#   eD = Dnext2-Dnext1
#   eσ = norm((u2-u1)[1:2])
#   eϵ = norm((u2-u1)[3])
#   if e₀ isa Real
#     e₀ref = e₀
#     e_normalized = (eD, eσ/r.G, eϵ)
#     ok_flag = all(e_normalized.<e₀)
#     e = maximum(e_normalized)
#   elseif e₀ isa NamedTuple
#     e₀ref = e₀.D
#     ok_flag = (eD<e₀.D) && (eσ<(e₀.σ)) && (eϵ<e₀.ϵ)
#     e_normalized = (eD, eσ*(e₀ref/e₀.σ), eϵ*(e₀ref/e₀.ϵ))
#     e,ind = findmax(e_normalized)
#     ok_flag || @debug("maximum error comes from indice $(ind) of (D,σ,ϵ)")
#   end

#   if ok_flag
#     # increse timestep
#     Δt_next = min(Δt*abs(e₀ref/e),Δt*2)
#     # keep best solution
#     return σᵢⱼnext2, ϵᵢⱼnext2, Dnext2, Δt, Δt_next
#   else
#     # recursively run with decreased timestep
#     adaptative_Δt_solve(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11, Δt*abs(e₀ref/e)) 
#     #initialy Δt*abs(e₀ref/e)^2 but without the square seems to generaly require less iterations.
#   end

# end
function vermeer_simple_shear_update_du_2(du,u,p,t)
    # unpacking
    r, mp, σ₁₁, σ₃, ϵ̇₁₂ = p
    σ̇₁₂, σ̇ₒₒₚ, ϵ̇₁₁, ϵ̇₂₂, Ḋ = du
    σ₁₂, σₒₒₚ, ϵ̇₁₁, ϵ₂₂, D = u

    # build σ and D from u
    σᵢⱼ = Tensor{2,3}([σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])
    KI = compute_KI(r,σᵢⱼ,D)
    Ḋ = compute_subcrit_damage_rate(r,KI,D)

    nl_u = SA[σ̇₁₂, σ̇ₒₒₚ, ϵ̇₁₁, ϵ̇₂₂]
    if all(nl_u .== 0)
        nl_u = SA[ϵ̇₁₂*2r.G, ϵ̇₁₂*2r.G, ϵ̇₁₂, ϵ̇₁₂]
    end
    #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
    for i in 1:mp.solver.newton_maxiter
        # get residual and its gradient with respect to u
        ∇res = ForwardDiff.jacobian(nl_u -> residual_simple_shear_2(nl_u,u,σᵢⱼ,Ḋ,p), nl_u)
        res  = residual_simple_shear_2(nl_u,u,σᵢⱼ,Ḋ,p)

        #println(∇res)
        # update u with Newton algo
        δnl_u = - ∇res\res
        dnl_u = nl_u + δnl_u
        #@debug "δu = $δu"
        #@debug "typeof(u) = $(typeof(u))"
        #println(norm(res))
        (norm(res) <= mp.solver.newton_abstol) && break

        (i == mp.solver.newton_maxiter) && @debug("ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
    end
    du[:] .= [nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
    println("du after update : ", du)
    return du
end

function residual_simple_shear_2(du,u,σᵢⱼ,Ḋ,p)
    # unpacking
    r, mp, σ₁₁, σ₃, ϵ̇₁₂ = p
    σ̇₁₂, σ̇ₒₒₚ, ϵ̇₁₁, ϵ̇₂₂ = du
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    

    # build σ̇ and ϵ̇ from du
    #zeroTdu = zero(eltype(du))
    σ̇ᵢⱼ = Tensor{2,3}([  0    σ̇₁₂   0  ;
                        σ̇₁₂    0    0  ;
                         0     0   σ̇ₒₒₚ ])
    ϵ̇ᵢⱼ = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true)

    # build residual
    res = SA[ϵ̇₁₁ - ϵ̇ᵢⱼ[1,1],
             ϵ̇₂₂ - ϵ̇ᵢⱼ[2,2],
             ϵ̇ᵢⱼ[3,3],
             ϵ̇₁₂ - ϵ̇ᵢⱼ[1,2]] * 2*r.G
    return res
end

function vermeer_simple_shear_update_du!(du,u,p,t)
    # unpacking
    r, mp, σ₃, ϵ̇₁₂ = p
    σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂, Ḋ = du
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u

    # build σ and D from u
    σᵢⱼ = Tensor{2,3}([σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])
    KI = compute_KI(r,σᵢⱼ,D)
    Ḋ = compute_subcrit_damage_rate(r,KI,D)

    nl_u = SA[σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂]
    if all(nl_u .== 0)
        nl_u = SA[1e-6 .* rand(4)...]
    end
    #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
    res0 = 0.0
    for i in 1:mp.solver.newton_maxiter
        # get residual and its gradient with respect to u
        ∇res = ForwardDiff.jacobian(nl_u -> residual_simple_shear(nl_u,u,σᵢⱼ,Ḋ,p), nl_u)
        res  = residual_simple_shear(nl_u,u,σᵢⱼ,Ḋ,p)
        (i == 1) && (norm_res0 = norm(res))
        #println(∇res)
        # update u with Newton algo
        δnl_u = - ∇res\res
        dnl_u = nl_u + δnl_u
        #@debug "δu = $δu"
        #@debug "typeof(u) = $(typeof(u))"
        #println(norm(res))
        (norm(res) <= mp.solver.newton_abstol) && break

        (i == mp.solver.newton_maxiter) && @debug("starting/ending norm res = $((norm(res0),norm(res)))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
    end
    du[:] .= [nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
    #println("du after update : ", du)
    return du
end

function vermeer_simple_shear_update_du(u,p,t)
    # unpacking
    r, mp, σ₃, ϵ̇₁₂, du_prev = p
    σ̇₁₁_p, σ̇₁₂_p, σ̇ₒₒₚ_p, ϵ̇₂₂_p, Ḋ_p = du_prev
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u

    # build σ and D from u
    σᵢⱼ = Tensor{2,3}([σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])
    KI = compute_KI(r,σᵢⱼ,D)
    Ḋ = compute_subcrit_damage_rate(r,KI,D)

    nl_u = SA[σ̇₁₁_p, σ̇₁₂_p, σ̇ₒₒₚ_p, ϵ̇₂₂_p]
    if all(nl_u .== 0)
        nl_u = SA[1e-6 .* rand(4)...]
    end
    #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
    res0 = 0.0
    for i in 1:mp.solver.newton_maxiter
        # get residual and its gradient with respect to u
        ∇res = ForwardDiff.jacobian(nl_u -> residual_simple_shear(nl_u,u,σᵢⱼ,Ḋ,p), nl_u)
        res  = residual_simple_shear(nl_u,u,σᵢⱼ,Ḋ,p)
        (i == 1) && (norm_res0 = norm(res))
        #println(∇res)
        # update u with Newton algo
        δnl_u = - ∇res\res
        dnl_u = nl_u + δnl_u
        #@debug "δu = $δu"
        #@debug "typeof(u) = $(typeof(u))"
        #println(norm(res))
        (norm(res) <= mp.solver.newton_abstol) && break

        (i == mp.solver.newton_maxiter) && @debug("starting/ending norm res = $((norm(res0),norm(res)))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
    end
    du = [nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
    #println("du after update : ", du)
    p[5]=du
    return du
end

function residual_simple_shear(du,u,σᵢⱼ,Ḋ,p)
    # unpacking
    r, mp, σ₃, ϵ̇₁₂ = p
    σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂ = du
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    

    # build σ̇ and ϵ̇ from du
    #zeroTdu = zero(eltype(du))
    σ̇ᵢⱼ = Tensor{2,3}([σ̇₁₁      σ̇₁₂     0  ;
                        σ̇₁₂    0   0  ;
                        0  0     σ̇ₒₒₚ  ])
    ϵ̇ᵢⱼ = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true)

    # build residual
    res = SA[ϵ̇ᵢⱼ[1,1],
             ϵ̇₂₂ - ϵ̇ᵢⱼ[2,2],
             ϵ̇ᵢⱼ[3,3],
             ϵ̇₁₂ - ϵ̇ᵢⱼ[1,2]] * 2r.G
    return res
end

function time_integration(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan)
  flags = p.flags
  t_vec = tspan[1]:Δt:tspan[2]
  σᵢⱼ_vec = Vector{SymmetricTensor{2,3}}(undef,length(t_vec))
  ϵᵢⱼ_vec = similar(σᵢⱼ_vec)
  D_vec = Vector{Float64}(undef,length(t_vec))
  σᵢⱼ_vec[1] = σᵢⱼ_i
  ϵᵢⱼ_vec[1] = ϵᵢⱼ_i
  D_vec[1] = D_i
  last_tsim_printed = 0.0
  for i in 2:length(t_vec)
    set_print_flag!(p,i,t_vec[i-1])
    flags.print && print_time_iteration(i,t_vec[i-1])
  
    σᵢⱼ_vec[i], ϵᵢⱼ_vec[i], D_vec[i], u = solve(r,p,σᵢⱼ_vec[i-1],ϵᵢⱼ_vec[i-1],D_vec[i-1],ϵ̇11,Δt)
  end
  return t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec
end

function adaptative_time_integration_2_points(r::Rheology,p::Params,S_i,σ₃,Dⁱ,Dᵒ,ϵ̇11,ϵ̇ⁱξη,Δt,θ,tspan ; damage_growth_out=true)
  #unpack
  time_maxiter = p.solver.time_maxiter
  flags = p.flags

  # fill first values
  σᵒᵢⱼ_i, σⁱᵢⱼ_i, ϵᵒᵢⱼ_i, ϵⁱᵢⱼ_i = initialize_state_var_D(r,p,S_i,σ₃,Dⁱ,Dᵒ,θ ; coords=:band)
  Ttensors = eltype(σᵒᵢⱼ_i)

  # initialize containers
  t_vec = Float64[tspan[1]]
  S_vec = Float64[S_i]
  σⁱᵢⱼ_vec = SymmetricTensor{2,3,Ttensors}[σⁱᵢⱼ_i]
  σᵒᵢⱼ_vec = SymmetricTensor{2,3,Ttensors}[σᵒᵢⱼ_i]
  ϵⁱᵢⱼ_vec = SymmetricTensor{2,3,Ttensors}[ϵⁱᵢⱼ_i]
  Dⁱ_vec = Float64[Dⁱ]
  Dᵒ_vec = Float64[Dᵒ]

  # initialize values
  tsim::Float64 = tspan[1]
  Snext::Float64 = S_i
  σⁱᵢⱼnext::typeof(σⁱᵢⱼ_i) = σⁱᵢⱼ_i
  σᵒᵢⱼnext::typeof(σᵒᵢⱼ_i) = σᵒᵢⱼ_i
  ϵⁱᵢⱼnext::typeof(ϵⁱᵢⱼ_i) = ϵⁱᵢⱼ_i
  Dⁱnext::Float64 = Dⁱ
  Dᵒnext::Float64 = Dᵒ
  Δt_next::Float64 = Δt
  Δt_used::Float64 = Δt

  flags.bifurcation = (Dⁱ==Dᵒ) ? false : true
  i = 1 # iter counter
  while tsim < tspan[2]
    set_print_flag!(p,i,tsim)
    flags.print && print_time_iteration(i,tsim)

    # solving procedure depends on the sign of S derivative
    if flags.bifurcation
      Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, Δt_used, Δt_next = adaptative_solve_2_points(r,p,Snext,σ₃,σⁱᵢⱼnext,σᵒᵢⱼnext,ϵⁱᵢⱼnext,Dᵒnext,Dⁱnext,ϵ̇ⁱξη,θ,-1,Δt_next ; damage_growth_out)
    else
      σⁱᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Δt_used, Δt_next = adaptative_solve_1_point(r,p,σⁱᵢⱼnext,ϵⁱᵢⱼnext,Dⁱnext,ϵ̇11,θ,Δt)
      σᵒᵢⱼnext = σⁱᵢⱼnext
      Dᵒnext = Dⁱnext
      σᵒᵢⱼnext_principal = principal_coords(σᵒᵢⱼnext,θ)
      Snext = σᵒᵢⱼnext_principal[1,1]/σᵒᵢⱼnext_principal[2,2]
    end

    if !flags.bifurcation # activate bifurcation procedure if S starts to decrease or if the derivative is zero.
      if Snext-S_vec[end] <= 0
        flags.bifurcation = true
        Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, Δt_used, Δt_next = adaptative_solve_2_points(r,p,Snext,σ₃,σⁱᵢⱼnext,σᵒᵢⱼnext,ϵⁱᵢⱼnext,Dᵒnext,Dⁱnext,ϵ̇ⁱξη,θ,-1,Δt_next ; damage_growth_out)
      end
    end

    # test :
    p.flags.nan && println("Nan flag true in main function, should return")
    # update tsim if no nans
    !flags.nan ? (tsim += Δt_used) : nothing

    set_save_flag!(p,i,tsim)
    if flags.save
      push!(S_vec,Snext)
      push!(σⁱᵢⱼ_vec,σⁱᵢⱼnext)
      push!(σᵒᵢⱼ_vec,σᵒᵢⱼnext)
      push!(ϵⁱᵢⱼ_vec,ϵⁱᵢⱼnext)
      push!(Dⁱ_vec,Dⁱnext)
      push!(Dᵒ_vec,Dᵒnext)
      push!(t_vec,tsim)
    end

    i += 1

    # break loop under conditions 
    flags.nan && break
    if !isnothing(time_maxiter)
      (length(t_vec)==time_maxiter+1) && break
    end
    D_max = 0.999
    (Dⁱnext > D_max) && (@info("Dⁱ reached $D_max, ending simulation") ; break)
    (Dᵒnext > D_max) && (@info("Dᵒ reached $D_max, ending simulation") ; break)
  end
  return t_vec, S_vec, σⁱᵢⱼ_vec, σᵒᵢⱼ_vec, ϵⁱᵢⱼ_vec, Dⁱ_vec, Dᵒ_vec
end   

function initialize_state_var_D(r,p,S_i,σ₃,Dⁱ,Dᵒ,θ ; coords=:band)
  ps = p.solver
  # get outside state to be used as reference
  σᵒᵢⱼ_i_principal = build_principal_stress_tensor(r,S_i,σ₃,Dᵒ ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
  ϵᵒᵢⱼ_i_principal = compute_ϵij(r,Dᵒ,σᵒᵢⱼ_i_principal)
  σᵒᵢⱼ_i = band_coords(σᵒᵢⱼ_i_principal,θ)
  ϵᵒᵢⱼ_i = band_coords(ϵᵒᵢⱼ_i_principal,θ)

  σⁱᵢⱼ_i_principal = build_principal_stress_tensor(r,S_i,σ₃,Dⁱ ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
  ϵⁱᵢⱼ_i_principal = compute_ϵij(r,Dⁱ,σⁱᵢⱼ_i_principal)
  σⁱᵢⱼ_i = band_coords(σⁱᵢⱼ_i_principal,θ)
  ϵⁱᵢⱼ_i = band_coords(ϵⁱᵢⱼ_i_principal,θ)
  
  # insert stress and strain componant constrained by stress continuity and strain compatibility
  σⁱᵢⱼ_guess = insert_into(σⁱᵢⱼ_i, (σᵒᵢⱼ_i[2,2],σᵒᵢⱼ_i[1,2]), ((2,2),(1,2)))
  ϵⁱᵢⱼ_guess = insert_into(ϵⁱᵢⱼ_i, ϵᵒᵢⱼ_i[1,1], (1,1))

  # initialize first guess for u = [ σξξ, σoop, ϵηη, ϵξη ]
  u = Vec(σⁱᵢⱼ_i[1,1], σⁱᵢⱼ_i[3,3], ϵⁱᵢⱼ_i[2,2], ϵⁱᵢⱼ_i[1,2])
  result = DiffResults.JacobianResult(u)

  for i in 1:ps.newton_maxiter
    # get residual and its gradient with respect to u
    ForwardDiff.jacobian!(result, u -> residual_initialize(r,Dⁱ,ϵⁱᵢⱼ_guess,σⁱᵢⱼ_guess,u), u)
    ∇res = DiffResults.jacobian(result)
    res  = DiffResults.value(result)

    # update u with Newton algo
    δu = - ∇res\res
    u = u + δu
    #@debug "δu = $δu"
    #@debug "typeof(u) = $(typeof(u))"

    (norm(res) <= ps.newton_abstol) && break
    
    (i == ps.newton_maxiter) && @debug("ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
  end
  σξξ, σoop, ϵηη, ϵξη = u
  σⁱᵢⱼ = insert_into(σⁱᵢⱼ_guess, (σξξ, σoop), ((1,1),(3,3))) 
  ϵⁱᵢⱼ = insert_into(ϵⁱᵢⱼ_guess, (ϵηη,ϵξη), ((2,2),(1,2))) 
  if coords == :principal 
    σᵒᵢⱼ = σᵒᵢⱼ_i_principal
    ϵᵒᵢⱼ = ϵᵒᵢⱼ_i_principal
    σⁱᵢⱼ = principal_coords(σⁱᵢⱼ,θ)
    ϵⁱᵢⱼ = principal_coords(ϵⁱᵢⱼ,θ)
  elseif coords == :band
    σᵒᵢⱼ = σᵒᵢⱼ_i
    ϵᵒᵢⱼ = ϵᵒᵢⱼ_i
  else
    @error "`coords` keyword argument $coords is not recognized. Use `:principal` or `:band` instead, `:band` being the default"
  end
  return σᵒᵢⱼ, σⁱᵢⱼ, ϵᵒᵢⱼ, ϵⁱᵢⱼ
end

function residual_initialize(r,Dⁱ,ϵⁱᵢⱼ_guess,σⁱᵢⱼ_guess,u)
  σξξ, σoop, ϵηη, ϵξη = u
  σⁱᵢⱼ = insert_into(σⁱᵢⱼ_guess, (σξξ, σoop), ((1,1),(3,3))) 
  ϵⁱᵢⱼ = insert_into(ϵⁱᵢⱼ_guess, (ϵηη,ϵξη), ((2,2),(1,2))) 
  ϵⁱᵢⱼ_analytical = compute_ϵij(r,Dⁱ,σⁱᵢⱼ)
  Δϵⁱᵢⱼ = ϵⁱᵢⱼ_analytical - ϵⁱᵢⱼ
  res = Vec(Δϵⁱᵢⱼ[1,1], Δϵⁱᵢⱼ[2,2], Δϵⁱᵢⱼ[3,3], Δϵⁱᵢⱼ[1,2])
  return res
end

# function adaptative_time_integration_2_points(r::Rheology,p::Params,S_i,σ₃,D_i,ϵ̇11,ϵ̇ⁱξη,Δt,θ,tspan ; damage_growth_out=true, bifurcate_on=:KI)
#   #unpack
#   time_maxiter = p.solver.time_maxiter

#   # fill first values
#   σᵢⱼ_i_principal = build_principal_stress_tensor(r,S_i,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
#   ϵᵢⱼ_i_principal = compute_ϵij(r,D_i,σᵢⱼ_i_principal)
#   σᵢⱼ_i = band_coords(σᵢⱼ_i_principal,θ)
#   ϵᵢⱼ_i = band_coords(ϵᵢⱼ_i_principal,θ)

#   # initialize containers
#   t_vec = Float64[tspan[1]]
#   S_vec = Float64[S_i]
#   σⁱᵢⱼ_vec = SymmetricTensor{2,3}[σᵢⱼ_i]
#   σᵒᵢⱼ_vec = SymmetricTensor{2,3}[σᵢⱼ_i]
#   ϵⁱᵢⱼ_vec = SymmetricTensor{2,3}[ϵᵢⱼ_i]
#   Dⁱ_vec = Float64[D_i]
#   Dᵒ_vec = Float64[D_i]

#   # initialize values
#   tsim::Float64 = tspan[1]
#   Snext::Float64 = S_i
#   σⁱᵢⱼnext::typeof(σᵢⱼ_i) = σᵢⱼ_i
#   σᵒᵢⱼnext::typeof(σᵢⱼ_i) = σᵢⱼ_i
#   ϵⁱᵢⱼnext::typeof(ϵᵢⱼ_i) = ϵᵢⱼ_i
#   Dⁱnext::Float64 = D_i
#   Dᵒnext::Float64 = D_i
#   Δt_next::Float64 = Δt
#   Δt_used::Float64 = Δt

#   last_tsim_printed = 0.0
#   last_tsim_saved = 0.0 
#   bifurcation_flag = false
#   i = 1 # iter counter
#   while tsim < tspan[2]
#     print_flag, last_tsim_printed = set_print_flag!(p,i,tsim,last_tsim_printed)
#     print_flag && print_time_iteration(i,tsim)

#     # solving procedure depends on the sign of S derivative
#     if bifurcation_flag
#       Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, Δt_used, Δt_next = adaptative_solve_2_points(r,p,Snext,σ₃,σⁱᵢⱼnext,σᵒᵢⱼnext,ϵⁱᵢⱼnext,Dᵒnext,Dⁱnext,ϵ̇ⁱξη,θ,Δt_next ; damage_growth_out)
#     else
#       σⁱᵢⱼnext_test, ϵⁱᵢⱼnext_test, Dⁱnext_test, Δt_used_test, Δt_next_test = adaptative_solve_1_point(r,p,σⁱᵢⱼnext,ϵⁱᵢⱼnext,Dⁱnext,ϵ̇11,θ,Δt)
#       σᵒᵢⱼnext_test = σⁱᵢⱼnext_test
#       Dᵒnext_test = Dⁱnext_test
#       σᵒᵢⱼnext_principal_test = principal_coords(σᵒᵢⱼnext_test,θ)
#       Snext_test = σᵒᵢⱼnext_principal_test[1,1]/σᵒᵢⱼnext_principal_test[2,2]
      
#       # compute bifurcation criterion
#       (bifurcate_on==:KI) && ( bifurcation_criterion = (compute_KI(r,σⁱᵢⱼnext_test,Dⁱnext_test)>0) )
#       (bifurcate_on==:S)  && ( bifurcation_criterion = ((Snext_test - Snext)<=0) )

#       if bifurcation_criterion
#         bifurcation_flag = true
#         Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, Δt_used, Δt_next = adaptative_solve_2_points(r,p,Snext,σ₃,σⁱᵢⱼnext,σᵒᵢⱼnext,ϵⁱᵢⱼnext,Dᵒnext,Dⁱnext,ϵ̇ⁱξη,θ,Δt_next ; damage_growth_out)
#       else
#         Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, Δt_used, Δt_next = Snext_test, σⁱᵢⱼnext_test, σᵒᵢⱼnext_test, ϵⁱᵢⱼnext_test, Dⁱnext_test, Dᵒnext_test, Δt_used_test, Δt_next_test
#       end

#     end

#     tsim += Δt_used

#     save_flag, last_tsim_saved = get_save_flag(p,i,tsim,last_tsim_saved)
#     if save_flag
#       push!(S_vec,Snext)
#       push!(σⁱᵢⱼ_vec,σⁱᵢⱼnext)
#       push!(σᵒᵢⱼ_vec,σᵒᵢⱼnext)
#       push!(ϵⁱᵢⱼ_vec,ϵⁱᵢⱼnext)
#       push!(Dⁱ_vec,Dⁱnext)
#       push!(Dᵒ_vec,Dᵒnext)
#       push!(t_vec,tsim)
#     end

#     i += 1

#     # break loop under conditions 
#     if !isnothing(time_maxiter)
#       (length(t_vec)==time_maxiter+1) && break
#     end
#     (Dⁱnext > 0.999) && break
#     (Dᵒnext > 0.999) && break
#   end
#   return t_vec, S_vec, σⁱᵢⱼ_vec, σᵒᵢⱼ_vec, ϵⁱᵢⱼ_vec, Dⁱ_vec, Dᵒ_vec
# end

# find S_i and D_i automaticaly so that S_i is maximized with constraint KI(S_i,D_i)<=0
# and solve for in and out from the beginning of the simulation with fixed D_out = D_i
# function adaptative_time_integration_2_points(r::Rheology,p::Params,σ₃,ϵ̇ⁱξη,Δt,θ,tspan ; damage_growth_out=false)
#   #unpack
#   time_maxiter = p.solver.time_maxiter

#   D_i, S_i =  get_KI_mininizer_D_on_S_range(r,1,50,σ₃ ; len=1000)
#   # fill first values
#   σᵢⱼ_i_principal = build_principal_stress_tensor(r,S_i,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
#   ϵᵢⱼ_i_principal = compute_ϵij(r,D_i,σᵢⱼ_i_principal)
#   σᵢⱼ_i = band_coords(σᵢⱼ_i_principal,θ)
#   ϵᵢⱼ_i = band_coords(ϵᵢⱼ_i_principal,θ)

#   # initialize containers
#   t_vec = Float64[tspan[1]]
#   S_vec = Float64[S_i]
#   σⁱᵢⱼ_vec = SymmetricTensor{2,3}[σᵢⱼ_i]
#   σᵒᵢⱼ_vec = SymmetricTensor{2,3}[σᵢⱼ_i]
#   ϵⁱᵢⱼ_vec = SymmetricTensor{2,3}[ϵᵢⱼ_i]
#   Dⁱ_vec = Float64[D_i]
#   Dᵒ_vec = Float64[D_i]

#   # initialize values
#   tsim::Float64 = tspan[1]
#   Snext::Float64 = S_i
#   σⁱᵢⱼnext::typeof(σᵢⱼ_i) = σᵢⱼ_i
#   σᵒᵢⱼnext::typeof(σᵢⱼ_i) = σᵢⱼ_i
#   ϵⁱᵢⱼnext::typeof(ϵᵢⱼ_i) = ϵᵢⱼ_i
#   Dⁱnext::Float64 = D_i
#   Dᵒnext::Float64 = D_i
#   Δt_next::Float64 = Δt
#   Δt_used::Float64 = Δt

#   last_tsim_printed = 0.0
#   last_tsim_saved = 0.0 
#   i = 1 # iter counter
#   while tsim < tspan[2]
#     set_print_flag!(p,i,tsim,last_tsim_printed)
#     p.flags.print_flag && print_time_iteration(i,tsim)

#     # solving procedure depends on the sign of S derivative
#     Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, Δt_used, Δt_next = adaptative_solve_2_points(r,p,Snext,σ₃,σⁱᵢⱼnext,σᵒᵢⱼnext,ϵⁱᵢⱼnext,Dᵒnext,Dⁱnext,ϵ̇ⁱξη,θ,Δt_next ; damage_growth_out)
    
#     tsim += Δt_used # update simulation time

#     # save
#     set_save_flag!(p,i,tsim)
#     if p.flags.save_flag
#       push!(S_vec,Snext)
#       push!(σⁱᵢⱼ_vec,σⁱᵢⱼnext)
#       push!(σᵒᵢⱼ_vec,σᵒᵢⱼnext)
#       push!(ϵⁱᵢⱼ_vec,ϵⁱᵢⱼnext)
#       push!(Dⁱ_vec,Dⁱnext)
#       push!(Dᵒ_vec,Dᵒnext)
#       push!(t_vec,tsim)
#     end

#     # increment iter counter
#     i += 1

#     # break loop under conditions 
#     if !isnothing(time_maxiter)
#       (length(t_vec)==time_maxiter+1) && break
#     end
#     (Dⁱnext > 0.999) && break
#   end
#   return t_vec, S_vec, σⁱᵢⱼ_vec, σᵒᵢⱼ_vec, ϵⁱᵢⱼ_vec, Dⁱ_vec, Dᵒ_vec
# end

function time_integration_2_points(r,p,S_i,σ₃,D_i,ϵ̇11,ϵ̇ⁱξη,Δt,θ,tspan ; damage_growth_out=true)
  flags = p.flags
  #initialize vectors
  t_vec = tspan[1]:Δt:tspan[2]
  σⁱᵢⱼ_vec = Vector{SymmetricTensor{2,3}}(undef,length(t_vec))
  σᵒᵢⱼ_vec = Vector{SymmetricTensor{2,3}}(undef,length(t_vec))
  ϵⁱᵢⱼ_vec = similar(σⁱᵢⱼ_vec)
  Dⁱ_vec = Vector{Float64}(undef,length(t_vec))
  Dᵒ_vec = Vector{Float64}(undef,length(t_vec))
  S_vec = Vector{Float64}(undef,length(t_vec))
  
  # fill first values
  σᵢⱼ_i = build_principal_stress_tensor(r,S_i,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
  ϵᵢⱼ_i = compute_ϵij(r,D_i,σᵢⱼ_i)

  σⁱᵢⱼ_vec[begin] = band_coords(σᵢⱼ_i,θ)
  σᵒᵢⱼ_vec[begin] = band_coords(σᵢⱼ_i,θ)
  ϵⁱᵢⱼ_vec[begin] = band_coords(ϵᵢⱼ_i,θ) # test that stress-strain relation is still valid after rotation !! 
  Dⁱ_vec[begin] = D_i
  Dᵒ_vec[begin] = D_i
  S_vec[begin] = S_i
  
  last_iter = 0
  for i in 2:length(t_vec)
    set_print_flag!(p,i,t_vec[i-1])
    flags.print && print_time_iteration(i,t_vec[i-1])

    # solving procedure depends on the sign of S derivative
    if flags.bifurcation
      S_vec[i], σⁱᵢⱼ_vec[i], σᵒᵢⱼ_vec[i], ϵⁱᵢⱼ_vec[i], Dⁱ_vec[i], Dᵒ_vec[i], _ = solve_2_points(r,p,S_vec[i-1],σ₃,σⁱᵢⱼ_vec[i-1],σᵒᵢⱼ_vec[i-1],ϵⁱᵢⱼ_vec[i-1],Dᵒ_vec[i-1],Dⁱ_vec[i-1],ϵ̇ⁱξη,θ,Δt ; damage_growth_out)
    else
      σⁱᵢⱼ_vec[i], ϵⁱᵢⱼ_vec[i], Dⁱ_vec[i], _ = solve_1_point(r,p,σⁱᵢⱼ_vec[i-1],ϵⁱᵢⱼ_vec[i-1],Dⁱ_vec[i-1],ϵ̇11,θ,Δt)
      σᵒᵢⱼ_vec[i] = σⁱᵢⱼ_vec[i]
      Dᵒ_vec[i] = Dⁱ_vec[i]
      σᵒᵢⱼ_principal = principal_coords(σᵒᵢⱼ_vec[i],θ)
      S_vec[i] = σᵒᵢⱼ_principal[1,1]/σᵒᵢⱼ_principal[2,2]
    end

    if !flags.bifurcation # activate bifurcation procedure if S starts to decrease or if the derivative is zero.
      (S_vec[i]-S_vec[i-1] <= 0) && (flags.bifurcation = true)
      if flags.bifurcation
        S_vec[i], σⁱᵢⱼ_vec[i], σᵒᵢⱼ_vec[i], ϵⁱᵢⱼ_vec[i], Dⁱ_vec[i], Dᵒ_vec[i], _ = solve_2_points(r,p,S_vec[i-1],σ₃,σⁱᵢⱼ_vec[i-1],σᵒᵢⱼ_vec[i-1],ϵⁱᵢⱼ_vec[i-1],Dᵒ_vec[i-1],Dⁱ_vec[i-1],ϵ̇ⁱξη,θ,Δt ; damage_growth_out)
      end
    end
    if Dⁱ_vec[i] > 0.999
      last_iter = i
      break
    end
  end
  li = last_iter
  return t_vec[1:li], S_vec[1:li], σⁱᵢⱼ_vec[1:li], σᵒᵢⱼ_vec[1:li], ϵⁱᵢⱼ_vec[1:li], Dⁱ_vec[1:li], Dᵒ_vec[1:li]
end