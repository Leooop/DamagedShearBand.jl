function vermeer_simple_shear_update_du(u,p,t)
    # unpacking
    r, du_prev, params = p
    σ̇₁₁_prev, σ̇₁₂_prev, σ̇ₒₒₚ_prev, ϵ̇₂₂_prev, _ = du_prev
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    # get stress tensor
    σᵢⱼ = SymmetricTensor{2,3}(SA[σ₁₁ σ₁₂ 0 ; σ₁₂ params.σ₃ 0 ; 0 0 σₒₒₚ])
    # compute damage growth rate
    KI = compute_KI(r,σᵢⱼ,D)
    Ḋ = compute_subcrit_damage_rate(r,KI,D ; vmax = params.dldt_max)

    # solve for non linear derivatives
    nl_u = SA[σ̇₁₁_prev, σ̇₁₂_prev, σ̇ₒₒₚ_prev, ϵ̇₂₂_prev] # first guess is previous iter
    nl_u = nl_solve(residual_simple_shear_old,nl_u,u,p,σᵢⱼ,Ḋ)
    # (du,u,σᵢⱼ,Ḋ,p)->residual_simple_shear(du,u,σᵢⱼ,Ḋ,p ; R1_allowed=params.regime1)
    # build back unknown vector
    du = SA[nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
    #println("du after update : ", du)
    p[2]=du
    
    return du
end

function nl_solve(res_func::F,nl_u,u,p,σᵢⱼ,Ḋ) where F<:Function
    # unpack
    r, du_prev, params = p
    # scale strain term 
    nl_u_scaled = setindex(nl_u,nl_u[4]*r.G, 4)
    
    #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
    for i in 1:params.newton_maxiter
        # get residual and its gradient with respect to u
        ∇res = ForwardDiff.jacobian(nl_u -> res_func(nl_u,u,σᵢⱼ,Ḋ,p), nl_u_scaled)
        res  = res_func(nl_u_scaled,u,σᵢⱼ,Ḋ,p)
        
        # update u with Newton algo
        nl_u_scaled = nl_u_scaled - ∇res\res
        
        if (norm(res) <= params.newton_abstol) && (abs(res[3]) > 1e-10) #test plain strain condition
            @show(norm(res),res[3])
        end
        (norm(res) <= params.newton_abstol) && (break) # @debug("Newton iter $i ending norm res = $(norm(res))") ;

        (i == params.newton_maxiter) && @debug("Newton maxiter ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
    end
    # rescale strain term :
    nl_u = setindex(nl_u_scaled,nl_u_scaled[4]/r.G, 4)
    return nl_u
end

function residual_simple_shear(du,u,σᵢⱼ,Ḋ,p ; R1_allowed=false)
    # unpacking
    r, du_prev, params = p
    σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, Gϵ̇₂₂ = du
    _, _, _, _, D = u
    ϵ̇₂₂ = Gϵ̇₂₂/r.G # rescale ϵ̇₂₂

    # build σ̇ and ϵ̇ from du
    #zeroTdu = zero(eltype(du))
    σ̇ᵢⱼ = SymmetricTensor{2,3}([ σ̇₁₁    σ̇₁₂   0.0  ;
                                 σ̇₁₂    0.0   0.0  ;
                                 0.0    0.0   σ̇ₒₒₚ ])
    σ = tr(σᵢⱼ)/3
    sᵢⱼ = dev(σᵢⱼ)
    τ = get_τ(sᵢⱼ)

    σ̇ = tr(σ̇ᵢⱼ)/3
    𝕀 = SymmetricTensor{2,3}(δ)
    ṡᵢⱼ = σ̇ᵢⱼ - σ̇*𝕀
    τ̇ = sᵢⱼ ⊡ ṡᵢⱼ / (2*τ)
    # if R1_allowed && (Ḋ == 0)
    #     KI = DSB.compute_KI(r,σ,τ,D)
    #     if KI <= 0
    #         res = SA[σ̇ᵢⱼ[1,1] - (3r.ν/(1+r.ν))*σ̇,
    #                  -2*r.G*ϵ̇₂₂ + σ̇ᵢⱼ[2,2] - (3r.ν/(1+r.ν))*σ̇,
    #                  σ̇ᵢⱼ[3,3] - (3r.ν/(1+r.ν))*σ̇,
    #                  -2*r.G*params.ϵ̇₁₂ + σ̇ᵢⱼ[1,2]]
    #         return res
    #     end
    # end

    c1, c2, c3 = compute_c1c2c3(r,D)
    A, B = compute_AB(r,c1,c2,c3)
    A1, B1 = compute_A1B1(r,A,B)

    dc1dD = compute_dc1dD(r,D)
    dc2dD = compute_dc2dD(r,D)
    dc3dD = compute_dc3dD(r,D)
    dA1dD = compute_dA1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)
    dB1dD = compute_dB1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)

    la1 = λ₁(A1,B1,σ,τ)
    la2 = λ₂(r,A1,B1,σ,τ)
    la3 = λ₃(A1,B1)
    dla1dD  = dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ)
    dla2dD  = dλ₂dD(A1,B1,dA1dD,dB1dD,σ,τ)
    dla3dD  = dλ₃dD(A1,B1,dA1dD,dB1dD)
    dla1dσ  = dλ₁dσ(A1,B1,τ)
    dla1dτ  = dλ₁dτ(A1,B1,σ,τ)
    # build residual
    # res = SA[ la1*σ̇₁₁ - la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*σᵢⱼ[1,1] - dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[1,1],
    #           -2*r.G*ϵ̇₂₂ - la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*params.σ₃ - dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[2,2],
    #           la1*σ̇ₒₒₚ - la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*σᵢⱼ[3,3] - dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[3,3],
    #           -2*r.G*params.ϵ̇₁₂ + la1*σ̇₁₂ + Ḋ*dla1dD*σᵢⱼ[1,2] + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[1,2] ] 
    res = SA[ la1*σ̇₁₁ + la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*σᵢⱼ[1,1] + dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[1,1],
              -2*r.G*ϵ̇₂₂ + la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*params.σ₃ + dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[2,2],
              la1*σ̇ₒₒₚ + la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*σᵢⱼ[3,3] + dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[3,3],
              -2*r.G*params.ϵ̇₁₂ + la1*σ̇₁₂ + Ḋ*dla1dD*σᵢⱼ[1,2] + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[1,2] ] 
    return res
end

function residual_simple_shear_old(du,u,σᵢⱼ,Ḋ,p ; keep_2G=false)
    # unpacking
    r, du_prev, params = p
    σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, Gϵ̇₂₂ = du
    _, _, _, _, D = u
    ϵ̇₂₂ = Gϵ̇₂₂/r.G # rescale ϵ̇₂₂

    # build σ̇ and ϵ̇ from du
    #zeroTdu = zero(eltype(du))
    σ̇ᵢⱼ = SymmetricTensor{2,3}([ σ̇₁₁    σ̇₁₂   0.0  ;
                                 σ̇₁₂    0.0   0.0  ;
                                 0.0    0.0   σ̇ₒₒₚ ])
    ϵ̇ᵢⱼ = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true, keep_2G)

    # build residual
    if keep_2G
        res = SA[ϵ̇ᵢⱼ[1,1],
                -2*r.G*ϵ̇₂₂ + ϵ̇ᵢⱼ[2,2],
                ϵ̇ᵢⱼ[3,3],
                -2*r.G*params.ϵ̇₁₂ + ϵ̇ᵢⱼ[1,2] ] 
    else
        res = SA[ϵ̇ᵢⱼ[1,1],
                -ϵ̇₂₂ + ϵ̇ᵢⱼ[2,2],
                ϵ̇ᵢⱼ[3,3],
                -params.ϵ̇₁₂ + ϵ̇ᵢⱼ[1,2] ]*2*r.G
    end
    return res
end
# function vermeer_simple_shear_update_du(u,p,t)
#     # unpacking
#     r, mp, σ₃, ϵ̇₁₂, du_prev, vmax = @view(p[1:6])
#     σ̇₁₁_prev, σ̇₁₂_prev, σ̇ₒₒₚ_prev, ϵ̇₂₂_prev, _ = du_prev
#     σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
#     # get stress tensor
#     σᵢⱼ = SymmetricTensor{2,3}(SA[σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])

#     # correct flaws orientation according to stress rotation
#     if length(p) == 8
#         ψ_cor = compute_flaws_angle_wrt_σ1(σᵢⱼ,p)
#         r = Rheology(r,(ψ=ψ_cor,)) 
#     end

#     # compute damage growth rate
#     KI = compute_KI(r,σᵢⱼ,D)
#     #@show r.ψ KI
#     Ḋ = compute_subcrit_damage_rate(r,KI,D ; vmax)

#     # solve for non linear derivatives
#     nl_u = SA[σ̇₁₁_prev, σ̇₁₂_prev, σ̇ₒₒₚ_prev, ϵ̇₂₂_prev] # first guess is previous iter
#     nl_u = nl_solve(residual_simple_shear,nl_u,u,p,σᵢⱼ,Ḋ)

#     # build back unknown vector
#     du = SA[nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
#     #println("du after update : ", du)
#     p[5]=du
    
#     return du
# end

# function nl_solve(res_func,nl_u,u,p,σᵢⱼ,Ḋ)
#     # unpack
#     r, mp, σ₃, ϵ̇₁₂, du_prev = @view(p[1:5])
#     # scale strain term 
#     nl_u_scaled = setindex(nl_u,nl_u[4]*r.G, 4)
#     if all(nl_u .== 0) # in case du_prev is initialized to zero (should be able to remove that)
#         nl_u_scaled = SA[r.G*ϵ̇₁₂, r.G*ϵ̇₁₂, r.G*ϵ̇₁₂, r.G*ϵ̇₁₂]
#     end
#     #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
#     for i in 1:mp.solver.newton_maxiter
#         # get residual and its gradient with respect to u
#         ∇res = ForwardDiff.jacobian(nl_u -> res_func(nl_u,u,σᵢⱼ,Ḋ,p), nl_u_scaled)
#         res  = res_func(nl_u_scaled,u,σᵢⱼ,Ḋ,p)
#         #(i == 1) && (norm_res0 = norm(res))
#         #println(∇res)
#         # update u with Newton algo
#         δnl_u = - ∇res\res
#         nl_u_scaled = nl_u_scaled + δnl_u
#         #@debug "δu = $δu"
#         #@debug "typeof(u) = $(typeof(u))"
#         if (norm(res) <= mp.solver.newton_abstol) && (abs(res[3]) > 1e-10) #test plain strain condition
#             @show(norm(res),res[3])
#         end
#         (norm(res) <= mp.solver.newton_abstol) && (break) # @debug("Newton iter $i ending norm res = $(norm(res))") ;

#         (i == mp.solver.newton_maxiter) && @debug("Newton maxiter ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
#     end
#     # rescale strain term :
#     nl_u = setindex(nl_u_scaled,nl_u_scaled[4]/r.G, 4)
#     return nl_u
# end

# function residual_simple_shear(du,u,σᵢⱼ,Ḋ,p)
#     # unpacking
#     r, mp, σ₃, ϵ̇₁₂ = @view(p[1:4])
#     σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂ = du
#     _, _, _, _, D = u
#     ϵ̇₂₂ = ϵ̇₂₂/r.G # rescale ϵ̇₂₂

#     # build σ̇ and ϵ̇ from du
#     #zeroTdu = zero(eltype(du))
#     σ̇ᵢⱼ = SymmetricTensor{2,3}([ σ̇₁₁    σ̇₁₂   0.0  ;
#                                  σ̇₁₂    0.0   0.0  ;
#                                  0.0    0.0   σ̇ₒₒₚ ])
#     ϵ̇ᵢⱼ = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true)

#     # build residual
#     res = SA[ϵ̇ᵢⱼ[1,1],
#              ϵ̇₂₂ - ϵ̇ᵢⱼ[2,2],
#              ϵ̇ᵢⱼ[3,3],
#              ϵ̇₁₂ - ϵ̇ᵢⱼ[1,2]] / ϵ̇₁₂
#     return res
# end

# # SHEAR STRESS CONTROL

# function vermeer_simple_shear_sigma_update_du(u,p,t ; log_eps=false)
#     # unpacking
#     r, mp, σ₃, σ̇₁₂, du_prev = @view(p[1:5])
#     σ̇₁₁_p, σ̇ₒₒₚ_p, ϵ̇₂₂_p, ϵ̇₁₂_p, _ = du_prev
#     σ₁₁, σₒₒₚ, ϵ₂₂, ϵ₁₂, D = u
#     # get stress tensor
#     σᵢⱼ = SymmetricTensor{2,3}(SA[σ₁₁ σ̇₁₂*t 0 ; σ̇₁₂*t σ₃ 0 ; 0 0 σₒₒₚ])

#     # correct flaws orientation according to stress rotation
#     if length(p) == 7
#         ψ_cor = compute_flaws_angle_wrt_σ1(σᵢⱼ,p)
#         r = Rheology(r,(ψ=ψ_cor,)) 
#     end

#     # compute damage growth rate
#     KI = compute_KI(r,σᵢⱼ,D)
#     #@show r.ψ KI
#     Ḋ = compute_subcrit_damage_rate(r,KI,D)

#     # solve for non linear derivatives
#     nl_u = SA[σ̇₁₁_p, σ̇ₒₒₚ_p, ϵ̇₂₂_p, ϵ̇₁₂_p] # first guess is previous iter
#     nl_u = nl_solve_sigma(residual_simple_shear_sigma,nl_u,u,p,σᵢⱼ,Ḋ ; log_eps)

#     # build back unknown vector
#     du = SA[nl_u[1], nl_u[2], nl_u[3], nl_u[4], Ḋ]
#     #println("du after update : ", du)
#     p[5]=du
    
#     return du
# end

# function nl_solve_sigma(res_func,nl_u,u,p,σᵢⱼ,Ḋ ; log_eps=true)
#     # unpack
#     r, mp, σ₃, σ̇₁₂, du_prev = @view(p[1:5])
#     # scale strain term 
#     nl_u_scaled = nl_u
#     if log_eps
#         #@show nl_u[4]
#         log_ϵ̇₁₂ = log10(sign(σ̇₁₂)*nl_u[4]) # take the log to force sign(ϵ̇₁₂) = sign(σ̇₁₂)
#         #@show logged_ϵ̇₁₂
#         # scale everything on the order of 1
#         nl_u_scaled = SA[nl_u[1]/σ̇₁₂, nl_u[2]/σ̇₁₂, nl_u[3]*r.G/σ̇₁₂, log_ϵ̇₁₂]
#     else
#         nl_u_scaled = SA[nl_u[1], nl_u[2], nl_u[3]*r.G, nl_u[4]*r.G]
#     end

#     if all(nl_u .== 0) # in case du_prev is initialized to zero (should be able to remove that)
#         nl_u_scaled = SA[σ̇₁₂, σ̇₁₂, σ̇₁₂, σ̇₁₂]
#     end
#     #result = DiffResults.JacobianResult(nl_u) no need for DiffResults with static vectors
#     for i in 1:mp.solver.newton_maxiter
#         # get residual and its gradient with respect to u
#         ∇res = ForwardDiff.jacobian(nl_u -> res_func(nl_u,u,σᵢⱼ,Ḋ,p ; log_eps), nl_u_scaled)
#         res  = res_func(nl_u_scaled,u,σᵢⱼ,Ḋ,p; log_eps)
#         #(i == 1) && (norm_res0 = norm(res))
#         #println(∇res)
#         # update u with Newton algo
#         δnl_u = - ∇res\res
#         nl_u_scaled = nl_u_scaled + δnl_u
#         #@debug "δu = $δu"
#         #@debug "typeof(u) = $(typeof(u))"
#         #println(norm(res))
#         (norm(res) <= mp.solver.newton_abstol) && (break) # @debug("Newton iter $i ending norm res = $(norm(res))") ;

#         (i == mp.solver.newton_maxiter) && @debug("Newton maxiter ending norm res = $(norm(res))")#("max newton iteration reached ($i), residual still higher than abstol with $(norm(res))")
#     end
#     # rescale everything :
#     if log_eps
#         nl_u = SA[nl_u_scaled[1]*σ̇₁₂, nl_u_scaled[2]*σ̇₁₂, nl_u_scaled[3]*σ̇₁₂/r.G, sign(σ̇₁₂)*exp10(nl_u_scaled[4])]
#     else
#         nl_u = SA[nl_u_scaled[1], nl_u_scaled[2], nl_u_scaled[3]/r.G, nl_u_scaled[4]/r.G]
#     end

#     return nl_u
# end

# function residual_simple_shear_sigma(du,u,σᵢⱼ,Ḋ,p ; log_eps=false)
#     # unpacking
#     r, mp, σ₃, σ̇₁₂ = @view(p[1:4])
#     scaled_σ̇₁₁, scaled_σ̇ₒₒₚ, scaled_ϵ̇₂₂, scaled_ϵ̇₁₂ = du
#     σ₁₁, σₒₒₚ, ϵ₂₂, ϵ₁₂, D = u

#     # rescale 
#     if log_eps
#         du = SA[scaled_σ̇₁₁*σ̇₁₂, scaled_σ̇ₒₒₚ*σ̇₁₂, scaled_ϵ̇₂₂*σ̇₁₂/r.G, sign(σ̇₁₂)*exp10(scaled_ϵ̇₁₂)]
#     else
#         du = SA[scaled_σ̇₁₁, scaled_σ̇ₒₒₚ, scaled_ϵ̇₂₂/r.G, scaled_ϵ̇₁₂/r.G]
#     end
#     σ̇₁₁, σ̇ₒₒₚ, ϵ̇₂₂, ϵ̇₁₂ = du

#     # build σ̇ and ϵ̇ from du
#     #zeroTdu = zero(eltype(du))
#     σ̇ᵢⱼ = Tensor{2,3}([σ̇₁₁      σ̇₁₂     0  ;
#                         σ̇₁₂    0   0  ;
#                         0  0     σ̇ₒₒₚ  ])
#     ϵ̇ᵢⱼ = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true)

#     # build residual
#     res = SA[ϵ̇ᵢⱼ[1,1],
#              ϵ̇₂₂ - ϵ̇ᵢⱼ[2,2],
#              ϵ̇ᵢⱼ[3,3],
#              ϵ̇₁₂ - ϵ̇ᵢⱼ[1,2]] / (σ̇₁₂/r.G)
#     return res
# end


# function compute_flaws_angle_wrt_σ1(σᵢⱼ,p)
#     _, _, _, control_rate, _, ψᵢ, σ1_initial_orientation = p
#     shear_strain_sign = (control_rate > 0) ? :positive : :negative
#     σ1_orientation = get_stress_deviation_from_y(σᵢⱼ ; shear_mode=:simple, shear_strain_sign)
#     Δσ1_orientation = σ1_orientation - σ1_initial_orientation
#     ψ_cor = ψᵢ - Δσ1_orientation
#     # ψ_cor has to be inside [-pi/2,pi/2] therefore: 
#     (ψ_cor > 90) && (ψ_cor -= 180)
#     (ψ_cor < -90) && (ψ_cor += 180)
#     return ψ_cor
# end
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