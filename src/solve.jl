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

function adaptative_solve_1_point(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,θ,Δt)
    e₀ = p.solver.e₀

    σᵢⱼnext1, ϵᵢⱼnext1, Dnext1, u1 = solve_1_point(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,θ,Δt)
    σᵢⱼmid, ϵᵢⱼmid, Dmid, umid = solve_1_point(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,θ,Δt/2)
    σᵢⱼnext2, ϵᵢⱼnext2, Dnext2, u2 = solve_1_point(r,p,σᵢⱼmid,ϵᵢⱼmid,Dmid,ϵ̇11,θ,Δt/2)

    # compute errors for each unknowns physical quantity and ponderate 
    eD = Dnext2-Dnext1
    eσ = norm((u2-u1)[1:2])
    eϵ = norm((u2-u1)[3])
    if e₀ isa Real
        e₀ref = e₀
        e_normalized = (eD, eσ/r.G, eϵ)
        ok_flag = all(e_normalized.<e₀)
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
        adaptative_solve_1_point(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,θ,Δt*abs(e₀ref/e)) 
        #initialy Δt*abs(e₀ref/e)^2 but without the square seems to generaly require less iterations.
    end
end

function solve_1_point(r::Rheology,p::Params,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,θ,Δt) 
    # this function is similar to solve but takes and returns tensor in the shear band coordinates instead of the principal stresses coordinates
    ps = p.solver
    D = D_i

    ϵᵢⱼ_i = principal_coords(ϵᵢⱼ_i,θ)
    σᵢⱼ_i = principal_coords(σᵢⱼ_i,θ)
    #println(eps()*norm(σᵢⱼ_i))
    @assert isapprox(σᵢⱼ_i[1,2],0.0,atol=1e-12*norm(σᵢⱼ_i)) # put this in tests at some point

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
    #_ , Ḋ = compute_ϵ̇ij(r,D,σᵢⱼ_i,σᵢⱼnext,Δt) # replaced by two next lines
    KI = compute_KI(r,(σᵢⱼnext+σᵢⱼ_i)/2,D)
    Ḋ = compute_subcrit_damage_rate(r,KI,D)
    Dnext = D + Ḋ*Δt
    return band_coords(σᵢⱼnext,θ), band_coords(ϵᵢⱼnext,θ), Dnext, u
end

function adaptative_solve_2_points(r,p,S,σ₃,σⁱᵢⱼ,σᵒᵢⱼ,ϵⁱᵢⱼ,Dᵒ,Dⁱ,ϵ̇ⁱξη,θ,Δt ; damage_growth_out=true)
    e₀ = p.solver.e₀

    Snext1, σⁱᵢⱼnext1, σᵒᵢⱼnext1, ϵⁱᵢⱼnext1, Dⁱnext1, Dᵒnext1, u1 = solve_2_points(r,p,S,σ₃,σⁱᵢⱼ,σᵒᵢⱼ,ϵⁱᵢⱼ,Dᵒ,Dⁱ,ϵ̇ⁱξη,θ,Δt ; damage_growth_out)
    Smid, σⁱᵢⱼmid, σᵒᵢⱼmid, ϵⁱᵢⱼmid, Dⁱmid, Dᵒmid, umid = solve_2_points(r,p,S,σ₃,σⁱᵢⱼ,σᵒᵢⱼ,ϵⁱᵢⱼ,Dᵒ,Dⁱ,ϵ̇ⁱξη,θ,Δt/2 ; damage_growth_out)
    Snext2, σⁱᵢⱼnext2, σᵒᵢⱼnext2, ϵⁱᵢⱼnext2, Dⁱnext2, Dᵒnext2, u2 = solve_2_points(r,p,Smid,σ₃,σⁱᵢⱼmid,σᵒᵢⱼmid,ϵⁱᵢⱼmid,Dᵒ,Dⁱmid,ϵ̇ⁱξη,θ,Δt/2 ; damage_growth_out)

    # compute errors for each unknowns physical quantity and ponderate 
    eD = max(Dⁱnext2-Dⁱnext1,Dᵒnext2-Dᵒnext1)
    eS = abs(Snext2-Snext1)
    eσ = norm((u2-u1)[2:3])
    eϵ = norm((u2-u1)[4])
    if e₀ isa Real
        e₀ref = e₀
        e_normalized = (eD, eS, eσ/r.G, eϵ) # todo better
        ok_flag = all(e_normalized.<e₀)
        e = maximum(e_normalized)
    elseif e₀ isa NamedTuple
        e₀ref = e₀.D
        ok_flag = (eD<e₀.D) && (eS<e₀.S) && (eσ<(e₀.σ)) && (eϵ<e₀.ϵ)
        e_normalized = (eD, eS*(e₀ref/e₀.S), eσ*(e₀ref/e₀.σ), eϵ*(e₀ref/e₀.ϵ))
        emax,ind = findmax(e_normalized)
        ok_flag || @debug("maximum error comes from indice $(ind) of (D,S,σ,ϵ)")
    end

    if ok_flag
        # increse timestep
        Δt_next::Float64 = min(Δt*abs(e₀ref/emax),Δt*2)
        # keep best solution
        return Snext2, σⁱᵢⱼnext2, σᵒᵢⱼnext2, ϵⁱᵢⱼnext2, Dⁱnext2, Dᵒnext2, Δt, Δt_next
    else
        # recursively run with decreased timestep
        factor::Float64 = abs(e₀ref/emax)
        adaptative_solve_2_points(r,p,S,σ₃,σⁱᵢⱼ,σᵒᵢⱼ,ϵⁱᵢⱼ,Dᵒ,Dⁱ,ϵ̇ⁱξη,θ, Δt*factor ; damage_growth_out) 
        #initialy Δt*abs(e₀ref/e)^2 but without the square seems to generaly require less iterations.
    end
end

function solve_2_points(r::Rheology,p::Params,S,σ₃,σⁱᵢⱼ,σᵒᵢⱼ,ϵⁱᵢⱼ,Dᵒ,Dⁱ,ϵ̇ⁱξη,θ,Δt ; damage_growth_out=true)
    ps = p.solver
    # get first guess of the unknowns with an elastic solve
    u = Vec(S, σⁱᵢⱼ[1,1], σⁱᵢⱼ[3,3], ϵⁱᵢⱼ[2,2]) # Snext, σⁱξξnext, σⁱoopnext, ϵⁱηηnext
    #@debug "u_i = $u"
    for i in 1:ps.newton_maxiter
        # get residual and its gradient with respect to u
        #∇res , res = Tensors.gradient(u -> residual_2_points(r,S,σ₃,Dⁱ,Dᵒ,ϵⁱᵢⱼ,σⁱᵢⱼ,σᵒᵢⱼ,ϵ̇ⁱξη,Δt,u), u, :all)
        result = DiffResults.JacobianResult(u)
        ForwardDiff.jacobian!(result, u -> residual_2_points(r,S,σ₃,Dⁱ,Dᵒ,ϵⁱᵢⱼ,σⁱᵢⱼ,σᵒᵢⱼ,ϵ̇ⁱξη,Δt,θ,u;damage_growth_out), u)
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
    # update strains and stresses with converged u
    Ṡ, σ̇ᵒᵢⱼ, σ̇ⁱᵢⱼ, ϵ̇ⁱᵢⱼ = compute_stress_strain_derivatives_from_u(r,S,σ₃,Dⁱ,Dᵒ,ϵⁱᵢⱼ,σⁱᵢⱼ,σᵒᵢⱼ,ϵ̇ⁱξη,Δt,θ,u)
    Snext = S + Ṡ*Δt
    σᵒᵢⱼnext = σᵒᵢⱼ + σ̇ᵒᵢⱼ*Δt
    σⁱᵢⱼnext = σⁱᵢⱼ + σ̇ⁱᵢⱼ*Δt
    ϵⁱᵢⱼnext = ϵⁱᵢⱼ + ϵ̇ⁱᵢⱼ*Δt

    KIᵒ = compute_KI(r,(σᵒᵢⱼnext+σᵒᵢⱼ)/2,Dᵒ)
    KIⁱ = compute_KI(r,(σⁱᵢⱼnext+σⁱᵢⱼ)/2,Dⁱ) # compute KI at intermediate stress : TO TEST.
    Ḋᵒ = compute_subcrit_damage_rate(r,KIᵒ,Dᵒ)
    Ḋⁱ = compute_subcrit_damage_rate(r,KIⁱ,Dⁱ)
    #_ , Ḋⁱ2 = compute_ϵ̇ij(r,Dⁱ,σⁱᵢⱼ,σⁱᵢⱼnext,Δt) # compute damage growth inside the shear band
    #@assert Ḋⁱ == Ḋⁱ2 # shouldn't error out, then remove preceeding line
    Dᵒnext = Dᵒ + Ḋᵒ*Δt
    Dⁱnext = Dⁱ + Ḋⁱ*Δt
    return Snext, σⁱᵢⱼnext, σᵒᵢⱼnext, ϵⁱᵢⱼnext, Dⁱnext, Dᵒnext, u
end

function residual_2_points(r,S,σ₃,Dⁱ,Dᵒ,ϵⁱᵢⱼ,σⁱᵢⱼ,σᵒᵢⱼ,ϵ̇ⁱξη,Δt,θ,u ; damage_growth_out=true)

    Ṡ, σ̇ᵒᵢⱼ, σ̇ⁱᵢⱼ, ϵ̇ⁱᵢⱼ = compute_stress_strain_derivatives_from_u(r,S,σ₃,Dⁱ,Dᵒ,ϵⁱᵢⱼ,σⁱᵢⱼ,σᵒᵢⱼ,ϵ̇ⁱξη,Δt,θ,u; damage_growth_out)
    σⁱᵢⱼnext = σⁱᵢⱼ + σ̇ⁱᵢⱼ*Δt

    ϵ̇ⁱᵢⱼ_analytical, _ = compute_ϵ̇ij(r,Dⁱ,σⁱᵢⱼ,σⁱᵢⱼnext,Δt ; damaged_allowed=true)
    Δϵ̇ⁱᵢⱼ = ϵ̇ⁱᵢⱼ_analytical - ϵ̇ⁱᵢⱼ
    res = Vec(Δϵ̇ⁱᵢⱼ[1,1], Δϵ̇ⁱᵢⱼ[2,2], Δϵ̇ⁱᵢⱼ[3,3], Δϵ̇ⁱᵢⱼ[1,2])
    return res
end

function compute_stress_strain_derivatives_from_u(r,S,σ₃,Dⁱ,Dᵒ,ϵⁱᵢⱼ,σⁱᵢⱼ,σᵒᵢⱼ,ϵ̇ⁱξη,Δt,θ,u ; damage_growth_out=true)
    Snext, σⁱξξnext, σⁱoopnext, ϵⁱηηnext = u
    σⁱξξ = σⁱᵢⱼ[1,1]
    σⁱoop = σⁱᵢⱼ[3,3]
    ϵⁱηη = ϵⁱᵢⱼ[2,2]
    ϵⁱξη = ϵⁱᵢⱼ[1,2]
    Ṡ = (Snext - S) / Δt
    ϵⁱξηnext = ϵⁱξη + ϵ̇ⁱξη*Δt

    σ̇ᵒᵢⱼ = compute_rotated_stress_rate_from_band_coords(r,Ṡ,σ₃,σᵒᵢⱼ,Dᵒ,Δt,θ ; damaged_allowed=damage_growth_out) # no damage outside the band
    σ̇ⁱᵢⱼ = get_σ̇ⁱij_from_primitive_vars(σ̇ᵒᵢⱼ,σⁱξξ,σⁱoop,σⁱξξnext,σⁱoopnext,Δt)
    #σ̇ⁱᵢⱼ = set_plane_strain_oop_stress_rate(σⁱᵢⱼ,σ̇ⁱᵢⱼ,r,Dⁱ,Δt ; damaged_allowed=true)
    ϵ̇ⁱᵢⱼ = get_ϵ̇ⁱᵢⱼ_from_primitive_vars(r,σᵒᵢⱼ,σ̇ᵒᵢⱼ,ϵⁱηη,ϵⁱξη,Dⁱ,ϵⁱηηnext,ϵⁱξηnext,Δt ; damage_growth_out)
    return Ṡ, σ̇ᵒᵢⱼ, σ̇ⁱᵢⱼ, ϵ̇ⁱᵢⱼ
end

function get_σ̇ⁱij_from_primitive_vars(σ̇ᵒᵢⱼ_band,σⁱξξ,σⁱoop,σⁱξξnext,σⁱoopnext,Δt)
    σ̇ⁱξξ = (σⁱξξnext - σⁱξξ)/Δt
    σ̇ⁱoop = (σⁱoopnext - σⁱoop)/Δt
    σ̇ⁱij_band = SymmetricTensor{2,3}([    σ̇ⁱξξ       σ̇ᵒᵢⱼ_band[1,2]     0    ;
                                        σ̇ᵒᵢⱼ_band[2,1] σ̇ᵒᵢⱼ_band[2,2]     0    ;
                                            0              0          σ̇ⁱoop   ])
    return σ̇ⁱij_band
end

function get_ϵ̇ⁱᵢⱼ_from_primitive_vars(r,σᵒᵢⱼ,σ̇ᵒᵢⱼ,ϵⁱηη,ϵⁱξη,D,ϵⁱηηnext,ϵⁱξηnext,Δt ; damage_growth_out=true)
    σᵒᵢⱼnext = σᵒᵢⱼ + σ̇ᵒᵢⱼ*Δt
    ϵ̇ⁱηη = (ϵⁱηηnext - ϵⁱηη) /Δt
    ϵ̇ⁱξη = (ϵⁱξηnext - ϵⁱξη) /Δt
    ϵ̇ᵒᵢⱼ, _ = compute_ϵ̇ij(r,D,σᵒᵢⱼ,σᵒᵢⱼnext,Δt ; damaged_allowed=damage_growth_out)
    ϵ̇ⁱᵢⱼ = SymmetricTensor{2,3}([ ϵ̇ᵒᵢⱼ[1,1]   ϵ̇ⁱξη   0  ;
                                    ϵ̇ⁱξη      ϵ̇ⁱηη   0  ;
                                        0         0     0  ]) # ϵ̇ᵒᵢⱼ[3,3] put zero on last term if needed to force plain strain
    return ϵ̇ⁱᵢⱼ
end

function compute_rotated_stress_rate_from_principal_coords(r,Ṡ,σ₃,σᵢⱼ_principal,D,Δt,θ ; damaged_allowed=true)
    σ̇ᵢⱼ_principal = SymmetricTensor{2,3}([Ṡ*σ₃ 0 0 ; 0 0 0 ; 0 0 r.ν*Ṡ*σ₃])
    σ̇ᵢⱼ_principal = set_plane_strain_oop_stress_rate(σᵢⱼ_principal,σ̇ᵢⱼ_principal,r,D,Δt ; abstol=1e-16, damaged_allowed)
    σ̇ᵢⱼ_band = band_coords(σ̇ᵢⱼ_principal,θ)
    return σ̇ᵢⱼ_band
end
function compute_rotated_stress_rate_from_band_coords(r,Ṡ,σ₃,σᵢⱼ_band,D,Δt,θ; damaged_allowed=true)
    σ̇ᵢⱼ_band = compute_rotated_stress_rate_guess(r,Ṡ,σ₃,θ)
    σ̇ᵢⱼ_band = set_plane_strain_oop_stress_rate(σᵢⱼ_band,σ̇ᵢⱼ_band,r,D,Δt ; abstol=1e-16, damaged_allowed)
    return σ̇ᵢⱼ_band
end

function compute_rotated_stress_rate_guess(r,Ṡ,σ₃,θ)
    σ̇ᵢⱼ_principal = SymmetricTensor{2,3}([Ṡ*σ₃ 0 0 ; 0 0 0 ; 0 0 r.ν*Ṡ*σ₃])
    σ̇ᵢⱼ_band = band_coords(σ̇ᵢⱼ_principal,θ)
    return σ̇ᵢⱼ_band
end