function vermeer_simple_shear_update_du(u,p,t)
    # unpacking
    r, mp, σ₃, ϵ̇₁₂, du_prev = @view(p[1:5])
    σ̇₁₁_p, σ̇₁₂_p, σ̇ₒₒₚ_p, ϵ̇₂₂_p, _ = du_prev
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    # get stress tensor
    σᵢⱼ = SymmetricTensor{2,3}(SA[σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])

    # correct flaws orientation according to stress rotation
    if length(p) == 7
        ψ_cor = compute_flaws_angle_wrt_σ1(σᵢⱼ,p)
        r = Rheology(r,(ψ=ψ_cor,)) 
    end

    # compute damage growth rate
    KI = compute_KI(r,σᵢⱼ,D)
    #@show r.ψ KI
    Ḋ = compute_subcrit_damage_rate(r,KI,D)

    # solve for non linear derivatives
    nl_u = nl_u = SA[σ̇₁₁_p, σ̇₁₂_p, σ̇ₒₒₚ_p, ϵ̇₂₂_p] # first guess is previous iter
    nl_u = nl_solve(residual_simple_shear,nl_u,u,p,σᵢⱼ,Ḋ)

    # build back unknown vector
    du = SA[nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
    #println("du after update : ", du)
    p[5]=du
    
    return du
end

function compute_flaws_angle_wrt_σ1(σᵢⱼ,p)
    _, _, _, ϵ̇₁₂, _, ψᵢ, σ1_initial_orientation = p
    shear_strain_sign = (sign(ϵ̇₁₂) == 1) ? :positive : :negative
    σ1_orientation = get_stress_deviation_from_y(σᵢⱼ ; shear_mode=:simple, shear_strain_sign)
    Δσ1_orientation = σ1_orientation - σ1_initial_orientation
    ψ_cor = ψᵢ - Δσ1_orientation
    # ψ_cor has to be inside [-pi/2,pi/2] therefore: 
    (ψ_cor > 90) && (ψ_cor -= 180)
    (ψ_cor < -90) && (ψ_cor += 180)
    return ψ_cor
end

function nl_solve(res_func,nl_u,u,p,σᵢⱼ,Ḋ)
    # unpack
    r, mp, σ₃, ϵ̇₁₂, du_prev = @view(p[1:5])
    # scale strain term 
    nl_u_scaled = setindex(nl_u,nl_u[4]*r.G, 4)
    if all(nl_u .== 0) # in case du_prev is initialized to zero (should be able to remove that)
        nl_u_scaled = SA[r.G*ϵ̇₁₂, r.G*ϵ̇₁₂, r.G*ϵ̇₁₂, r.G*ϵ̇₁₂]
    end
    #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
    for i in 1:mp.solver.newton_maxiter
        # get residual and its gradient with respect to u
        ∇res = ForwardDiff.jacobian(nl_u -> res_func(nl_u,u,σᵢⱼ,Ḋ,p), nl_u_scaled)
        res  = res_func(nl_u_scaled,u,σᵢⱼ,Ḋ,p)
        #(i == 1) && (norm_res0 = norm(res))
        #println(∇res)
        # update u with Newton algo
        δnl_u = - ∇res\res
        nl_u_scaled = nl_u_scaled + δnl_u
        #@debug "δu = $δu"
        #@debug "typeof(u) = $(typeof(u))"
        #println(norm(res))
        (norm(res) <= mp.solver.newton_abstol) && (break) # @debug("Newton iter $i ending norm res = $(norm(res))") ;

        (i == mp.solver.newton_maxiter) && @debug("Newton maxiter ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
    end
    # rescale strain term :
    nl_u = setindex(nl_u_scaled,nl_u_scaled[4]/r.G, 4)
    return nl_u
end
function residual_simple_shear(du,u,σᵢⱼ,Ḋ,p)
    # unpacking
    r, mp, σ₃, ϵ̇₁₂ = @view(p[1:4])
    σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂ = du
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    ϵ̇₂₂ = ϵ̇₂₂/r.G # rescale ϵ̇₂₂

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
             ϵ̇₁₂ - ϵ̇ᵢⱼ[1,2]] / ϵ̇₁₂
    return res
end
# function vermeer_simple_shear_update_du_2(du,u,p,t)
#     # unpacking
#     r, mp, σ₁₁, σ₃, ϵ̇₁₂ = p
#     σ̇₁₂, σ̇ₒₒₚ, ϵ̇₁₁, ϵ̇₂₂, Ḋ = du
#     σ₁₂, σₒₒₚ, ϵ̇₁₁, ϵ₂₂, D = u

#     # build σ and D from u
#     σᵢⱼ = Tensor{2,3}([σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])
#     KI = compute_KI(r,σᵢⱼ,D)
#     Ḋ = compute_subcrit_damage_rate(r,KI,D)

#     nl_u = SA[σ̇₁₂, σ̇ₒₒₚ, ϵ̇₁₁, ϵ̇₂₂]
#     if all(nl_u .== 0)
#         nl_u = SA[ϵ̇₁₂*2r.G, ϵ̇₁₂*2r.G, ϵ̇₁₂, ϵ̇₁₂]
#     end
#     #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
#     for i in 1:mp.solver.newton_maxiter
#         # get residual and its gradient with respect to u
#         ∇res = ForwardDiff.jacobian(nl_u -> residual_simple_shear_2(nl_u,u,σᵢⱼ,Ḋ,p), nl_u)
#         res  = residual_simple_shear_2(nl_u,u,σᵢⱼ,Ḋ,p)

#         #println(∇res)
#         # update u with Newton algo
#         δnl_u = - ∇res\res
#         dnl_u = nl_u + δnl_u
#         #@debug "δu = $δu"
#         #@debug "typeof(u) = $(typeof(u))"
#         #println(norm(res))
#         (norm(res) <= mp.solver.newton_abstol) && break

#         (i == mp.solver.newton_maxiter) && @debug("ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
#     end
#     du[:] .= [nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
#     println("du after update : ", du)
#     return du
# end

# function residual_simple_shear_2(du,u,σᵢⱼ,Ḋ,p)
#     # unpacking
#     r, mp, σ₁₁, σ₃, ϵ̇₁₂ = p
#     σ̇₁₂, σ̇ₒₒₚ, ϵ̇₁₁, ϵ̇₂₂ = du
#     σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    

#     # build σ̇ and ϵ̇ from du
#     #zeroTdu = zero(eltype(du))
#     σ̇ᵢⱼ = Tensor{2,3}([  0    σ̇₁₂   0  ;
#                         σ̇₁₂    0    0  ;
#                          0     0   σ̇ₒₒₚ ])
#     ϵ̇ᵢⱼ = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true)

#     # build residual
#     res = SA[ϵ̇₁₁ - ϵ̇ᵢⱼ[1,1],
#              ϵ̇₂₂ - ϵ̇ᵢⱼ[2,2],
#              ϵ̇ᵢⱼ[3,3],
#              ϵ̇₁₂ - ϵ̇ᵢⱼ[1,2]] * 2*r.G
#     return res
# end

# function vermeer_simple_shear_update_du!(du,u,p,t)
#     # unpacking
#     r, mp, σ₃, ϵ̇₁₂ = p
#     σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂, Ḋ = du
#     σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u

#     # build σ and D from u
#     σᵢⱼ = Tensor{2,3}([σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])
#     KI = compute_KI(r,σᵢⱼ,D)
#     Ḋ = compute_subcrit_damage_rate(r,KI,D)

#     nl_u = SA[σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂]
#     if all(nl_u .== 0)
#         nl_u = SA[1e-6 .* rand(4)...]
#     end
#     #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
#     res0 = 0.0
#     for i in 1:mp.solver.newton_maxiter
#         # get residual and its gradient with respect to u
#         ∇res = ForwardDiff.jacobian(nl_u -> residual_simple_shear(nl_u,u,σᵢⱼ,Ḋ,p), nl_u)
#         res  = residual_simple_shear(nl_u,u,σᵢⱼ,Ḋ,p)
#         (i == 1) && (norm_res0 = norm(res))
#         #println(∇res)
#         # update u with Newton algo
#         δnl_u = - ∇res\res
#         nl_u = nl_u + δnl_u
#         #@debug "δu = $δu"
#         #@debug "typeof(u) = $(typeof(u))"
#         #println(norm(res))
#         (norm(res) <= mp.solver.newton_abstol) && (@debug("Newton iter $i ending norm res = $(norm(res))") ; break)

#         (i == mp.solver.newton_maxiter) && @debug("Newton maxiter ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
#     end
#     du[:] .= [nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
#     #println("du after update : ", du)
#     return du
# end

### old pure shear related time integration ###
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