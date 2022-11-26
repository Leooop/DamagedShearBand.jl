E_from_Gν(G,ν) = 2G*(1 + ν)
λ_from_Gν(G,ν) = 2G*ν / (1 - 2ν)


function compute_c1(r,D)
  α = cosd(r.ψ)
  return 1/(π^2 * α^(3/2) * ((D/r.D₀)^(1/3) - 1 + r.β/α)^(3/2))
end


function compute_c2(r,D)
  α = cosd(r.ψ)
  return 2 * ((D/r.D₀)^(1/3) - 1)^(1/2) * (r.D₀^(2/3)/(1-D^(2/3))) / (π^2 * α^(3/2)) 
end

function compute_c3(r,D)
  α = cosd(r.ψ)
  return 2/π * sqrt(α) * ((D/r.D₀)^(1/3) - 1)^(1/2) 
end
compute_c1c2c3(r::Rheology,D) = (compute_c1(r,D), compute_c2(r,D), compute_c3(r,D))


compute_A1(r) = π * sqrt(r.β/3) * (sqrt(1+r.μ^2) - r.μ) #TODO: Check if + or - μ makes sense
compute_A3(r, A1) = A1 * (sqrt(1+r.μ^2) + r.μ) / (sqrt(1+r.μ^2) - r.μ)
compute_A3(r) = compute_A1(r) * (sqrt(1+r.μ^2) + r.μ) / (sqrt(1+r.μ^2) - r.μ)
compute_A1A3(r::Rheology) = (A1 = compute_A1(r) ; A3 = compute_A3(r, A1) ; return (A1, A3))

"principal stress formulation of KI from Bhat 2011"
compute_KI(r::Rheology, σ₁, σ₃, A1, A3, c1, c2, c3) = sqrt(π*r.a) * ((σ₃*A3 - σ₁*A1) * (c1 + c2) + σ₃*c3)
compute_KI(r::Rheology, σ₁, σ₃, D) = compute_KI(r, σ₁, σ₃, compute_A1A3(r)..., compute_c1c2c3(r,D)...)



