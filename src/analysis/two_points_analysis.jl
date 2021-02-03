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