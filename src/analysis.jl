
function set_plane_strain_oop_stress(σᵢⱼ,r,D ; abstol=1e-12, maxiter=100, σoop_guess=nothing)
  # If the initial out of plane is different from σ₃₃ insert a guess
  !isnothing(σoop_guess) && (σᵢⱼ = insert_σoop(σᵢⱼ,σoop_guess))
  σoop = σᵢⱼ[3,3]
  # get damage constants:
  A1,B1 = compute_A1B1(r,D)
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

function get_KI_mininizer_D(r)
  func_to_minimize = D -> -A_over_B(r,D) #optimize function needs an array of variables
  Dc = optimize(func_to_minimize, r.D₀, 0.99).minimizer[1]
  return Dc
end

function A_over_B(r,D)
  A, B = compute_AB(r,D)
  return A/B
end

function build_principal_stress_tensor(r,S,σ₃)
  σᵢⱼ_prev = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_prev,r,r.D₀)
end

function build_principal_stress_tensor(r,S,σ₃,D)
  σᵢⱼ_prev = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_prev,r,D)
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

function solve(r,σᵢⱼ_in,ϵᵢⱼ_in,knowns,unknowns,D,Δt ; abstol=1e-12, maxiter=100)
  Ḋ = zero(D)
  ΔD = similar(Ḋ)
  ∇residual = similar(Δσ_out)
  residual = similar(Δσ_out)
  u = similar(unknowns)
  ϵᵢⱼ_in_next = ϵᵢⱼ_in
  for i in 1:maxiter
    Ḋ = compute_subcrit_damage_rate(r, σij_in, D)
    ΔD = Ḋ*Δt
    ∇residual , residual = Tensors.gradient(unknowns -> residual(r, ϵᵢⱼ_in_next, D+ΔD, knowns, unknowns), u, :all)
    u = u - residual/∇residual
    ϵᵢⱼ_in_next = increment_ϵᵢⱼ_in(ϵᵢⱼ_in,knowns,u)
    (abs(residual) <= abstol) && break
    (i == maxiter) && @warn("maxiter reached, residual still higher than abstol")
  end
  σᵢⱼ_in_next = increment_σᵢⱼ_in(r,σᵢⱼ_in,D+ΔD,knowns,u) 
  return σᵢⱼ_in_next, ϵᵢⱼ_in_next, knowns, u
end

function residual(r, ϵᵢⱼ, D, knowns, unknowns)
  Δσηη , Δσξη, Δϵξξ = knowns
  Δσξξ, Δϵηη, Δϵξη = unknowns
  Δϵ_in = SymmetricTensor{2,3}([Δϵξξ Δϵξη  0 ; 
                                Δϵξη Δϵηη  0 ;
                                 0    0    0 ])
  # Δσ_in out of plane guess is an elastic undamaged one
  Δσ_in_guess = SymmetricTensor{2,3}([ Δσξξ  Δσξη 0 ; 
                                       Δσξη  Δσηη 0 ;
                                        0     0   r.ν*(Δσξξ + Δσηη) ])
  Δσ_in = set_plane_strain_oop_stress(Δσ_in_guess, r, D)
  C = compute_damaged_stiffness_tensor(r,ϵij,D)
  return Δσ_in - C ⊡ Δϵ_in
end

function increment_ϵᵢⱼ_in(ϵᵢⱼ,knowns,unknowns) 
  _ , _, Δϵξξ = knowns
  _, Δϵηη, Δϵξη = unknowns
  Δϵᵢⱼ = SymmetricTensor{2,3}([Δϵξξ Δϵξη 0 ; Δϵξη Δϵηη 0 ; 0 0 0])
  ϵᵢⱼ_next = ϵᵢⱼ + Δϵᵢⱼ
  return ϵᵢⱼ_next
end

function increment_σᵢⱼ_in(r,σᵢⱼ,D,knowns,unknowns) 
  Δσηη , Δσξη, _ = knowns
  Δσξξ, _, _ = unknowns
  Δσᵢⱼ = SymmetricTensor{2,3}([Δσξξ Δσξη 0 ; Δσξη Δσξξ 0 ; 0 0 ν*(Δσξξ+Δσξξ) ])
  σᵢⱼ_next = set_plane_strain_oop_stress(σᵢⱼ + Δσᵢⱼ, r, D)
  return σᵢⱼ_next
end