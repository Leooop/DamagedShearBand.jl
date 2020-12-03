
function set_plane_strain_oop_stress(σᵢⱼ,r,D; abstol=1e-12, maxiter=100, σoop_guess=nothing)
  # If the initial out of plane is different from σ₃₃ insert a guess
  !isnothing(σoop_guess) && (σᵢⱼ = insert_σoop(σᵢⱼ,σoop_guess))
  σoop = σᵢⱼ[3,3]
  # use DiffResult object to compute ϵoop and dϵoop at the same time.
  result = GradientResult(σᵢⱼ)
  # get damage constants:
  A1,B1 = compute_A1B1(r,D)
  for i in 1:maxiter
    # Joint computation of value and grad of ϵ_oop
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

function get_damage_onset(r,σ₃)
  func_to_minimize = S -> abs(zero_at_damage_onset(r,first(S),σ₃)) #optimize function needs an array of variables
  Sc = optimize(func_to_minimize, [3.0],method=LBFGS(),autodiff = :forward).minimizer[1]
  return Sc
end

function zero_at_damage_onset(r,S,σ₃)
  σᵢⱼ = build_elastic_stress_tensor(r,S,σ₃)
  σ = 1/3*tr(σᵢⱼ)
  sᵢⱼ = dev(σᵢⱼ)
  τ = get_τ(sᵢⱼ)
  return abs(τ/σ) - r.μ
end

function build_elastic_stress_tensor(r,S,σ₃)
  σᵢⱼ_prev = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])
  return set_plane_strain_oop_stress(σᵢⱼ_prev,r,r.D₀)
end
