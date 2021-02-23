
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

function set_plane_strain_oop_stress_rate_2(r,σᵢⱼ,σ̇ᵢⱼ,D,Ḋ ; abstol=1e-16, maxiter=100, damaged_allowed=true)
  # If the initial out of plane is different from σ₃₃ insert a guess
  #!isnothing(σoop_guess) && (σᵢⱼ = insert_σoop(σᵢⱼ,σoop_guess)) # Removed because of type inference issues
  σoop = σᵢⱼ[3,3]
  σoop_rate = σ̇ᵢⱼ[3,3]
  # get damage constants:
  A1,B1 = compute_A1B1(r,D)
  # use a static array version of σ̇ᵢⱼ for AD
  #σ̇ᵢⱼ_sa = SA[σ̇ᵢⱼ[1,1] σ̇ᵢⱼ[1,2] σ̇ᵢⱼ[1,3] ; σ̇ᵢⱼ[2,1] σ̇ᵢⱼ[2,2] σ̇ᵢⱼ[2,3] ; σ̇ᵢⱼ[3,1] σ̇ᵢⱼ[3,2] σ̇ᵢⱼ[3,3]]
  for i in 1:maxiter
    # Joint computation of value and grad of ϵ_oop 
    dϵ̇_oop = ForwardDiff.gradient(σ̇ᵢⱼ -> compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed)[3,3], σ̇ᵢⱼ)
    ϵ̇_oop  = compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed)[3,3]
    #println(dϵ̇_oop)
    dϵ̇_oop_dσ̇_oop = dϵ̇_oop[3,3]
    # Newton update
    σoop_rate = σoop_rate - ϵ̇_oop/dϵ̇_oop_dσ̇_oop
    σ̇ᵢⱼ = insert_σoop(σ̇ᵢⱼ,σoop_rate)
    #σ̇ᵢⱼ_sa = setindex(σ̇ᵢⱼ_sa,σoop_rate,9)
    # exit condition
    (abs(ϵ̇_oop) <= abstol) && break
  end
  return σ̇ᵢⱼ
end

# function get_strain_from_stress(r,D,σᵢⱼ)
#   ϵᵢⱼ = compute_ϵij(r,D,σᵢⱼ)

# end

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
  σᵢⱼ_guess = SymmetricTensor{2,3}(SA[S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_guess,r,r.D₀ ; abstol)
end

function build_principal_stress_tensor(r,S,σ₃,D ; abstol=1e-15)
  σᵢⱼ_guess = SymmetricTensor{2,3}(SA[S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_guess,r,D ; abstol)
end

function band_coords(σᵢⱼ,θ)
  Q = Tensor{2,2}([sind(θ) cosd(θ) ; -cosd(θ) sind(θ)])
  σ2D = SymmetricTensor{2,2,eltype(σᵢⱼ)}(SA[σᵢⱼ[1,1] σᵢⱼ[1,2] ; σᵢⱼ[1,2] σᵢⱼ[2,2]])
  σ2D_rotated = Q ⋅ σ2D ⋅ Q'
  σᵢⱼ_rotated = SymmetricTensor{2,3,eltype(σᵢⱼ)}((σ2D_rotated[1,1],σ2D_rotated[2,1],σᵢⱼ[3,1],σ2D_rotated[2,2],σᵢⱼ[3,2],σᵢⱼ[3,3]))
  #σᵢⱼ_rotated = Tensor{2,3}([σ2D_rotated zeros(eltype(σᵢⱼ),2) ; [zero(eltype(σᵢⱼ)) zero(eltype(σᵢⱼ)) σᵢⱼ[3,3]]])
  return σᵢⱼ_rotated
end

function principal_coords(σᵢⱼ,θ)
  Q = Tensor{2,2}([sind(θ) -cosd(θ) ; cosd(θ) sind(θ)])
  σ2D = SymmetricTensor{2,2,eltype(σᵢⱼ)}(SA[σᵢⱼ[1,1] σᵢⱼ[1,2] ; σᵢⱼ[1,2] σᵢⱼ[2,2]])
  σ2D_rotated = Q ⋅ σ2D ⋅ Q'
  σᵢⱼ_rotated = SymmetricTensor{2,3,eltype(σᵢⱼ)}((σ2D_rotated[1,1],σ2D_rotated[2,1],σᵢⱼ[3,1],σ2D_rotated[2,2],σᵢⱼ[3,2],σᵢⱼ[3,3]))
  #σᵢⱼ_rotated = Tensor{2,3}([σ2D_rotated zeros(eltype(σᵢⱼ),2) ; [zero(eltype(σᵢⱼ)) zero(eltype(σᵢⱼ)) σᵢⱼ[3,3]]])
  return σᵢⱼ_rotated
end

function get_stress_deviation_from_far_field(σᵢⱼ ; offdiagtol=1e-5, compression_axis=:x, shear_mode=:pure, shear_strain_sign=:positive)
  σᵢⱼ2D = SymmetricTensor{2,2}(SA[σᵢⱼ[1,1] σᵢⱼ[1,2] ; σᵢⱼ[2,1] σᵢⱼ[2,2]])
  if shear_mode==:simple
    if (σᵢⱼ2D[1,1]==σᵢⱼ2D[2,2]) && (σᵢⱼ2D[1,2]==0)
      (shear_strain_sign==:positive) && (return 45.0)
      (shear_strain_sign==:negative) && (return -45.0)
    end
  end

  σᵢⱼ = filter_offdiagonal(σᵢⱼ ; tol=offdiagtol)
  F = eigen(σᵢⱼ[1:2,1:2])
  #any(isnan.([F.values F.vectors])) && (F = eigen(Array(σᵢⱼ)))
  ind = findall(==(minimum(F.values)),F.values)
  σ₁_xy_direction = length(ind) == 1 ? Vec(F.vectors[:,ind]...) : Vec(1.0,0.0)
  σ₁_far_field_direction = (compression_axis==:x) ? Vec(1.0,0.0) : Vec(0.0,1.0)
  if all(σ₁_xy_direction .≈ σ₁_far_field_direction)
    angle = 0.0
  else
    angle = atand(σ₁_xy_direction[2],σ₁_xy_direction[1]) # wrt x direction
    (compression_axis==:y) && (angle = 90 + angle)
  end
  #(angle == 90) && (angle = zero(angle))
  (angle > 90) && (angle -= 180)
  return angle
end

get_stress_deviation_from_y(σᵢⱼ ; shear_mode=:pure, shear_strain_sign = :positive) = get_stress_deviation_from_far_field(σᵢⱼ ; offdiagtol=1e-5, compression_axis=:y, shear_mode, shear_strain_sign)

include("solve.jl")
include("single_point_analysis.jl")
include("two_points_analysis.jl")
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

