
function set_plane_strain_oop_stress(σᵢⱼ,r,D; abstol=1e-9, maxiter=100, σoop_guess=nothing)
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
