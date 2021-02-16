E_from_Gν(G,ν) = 2G*(1 + ν)
λ_from_Gν(G,ν) = 2G*ν / (1 - 2ν)

Dᵉ_func(i,j,k,l,G,λ) = λ*(δ(i,j)*δ(k,l)) + G*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
Cᵉ_func(i,j,k,l,G,λ) = (λ/(2G*(3λ + 2G)))*(δ(i,j)*δ(k,l)) + (1/2G)*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

get_elastic_stiffness_tensor(G,λ) = SymmetricTensor{4, 3}( (i,j,k,l) -> Dᵉ_func(i,j,k,l,G,λ))
get_elastic_stiffness_tensor(r::Rheology) = SymmetricTensor{4, 3}( (i,j,k,l) -> Dᵉ_func(i,j,k,l,r.G,λ_from_Gν(r.G,r.ν)))
get_elastic_compliance_tensor(r::Rheology) = SymmetricTensor{4, 3}( (i,j,k,l) -> Cᵉ_func(i,j,k,l,r.G,λ_from_Gν(r.G,r.ν)))

free_energy_convexity(r::Rheology,D) = 1/compute_Γ(r,compute_A1B1(r,D)...) > 0 ? true : false
# eq 16 Bhat2012 & 2016 & notes (because in Bhat2011 c2 isn't the same form as in Harsha's notes) :
#compute_c1(d::Damage,D) = sqrt(1-cos(d.ψ)^2)/(π*cos(d.ψ)^(3/2)*((D/d.D₀)^(1/3) - 1 + d.β/cos(d.ψ))^(3/2))
function compute_c1(r,D)
  α = cosd(r.ψ)
  #(α < 0) && (α = -α) #### careful with that
  @assert α >= 0

  if r.D₀>0
    isnan(D) && error("D is NaN")
    isnan(r.D₀) && error("r.D₀ is NaN")
    ((D/r.D₀) < 1) && println("D = $D & D₀ = $(r.D₀)")
    @assert (D/r.D₀) >= 1
    return sqrt(1-α^2)/(π*α^(3/2)*((D/r.D₀)^(1/3) - 1 + r.β/α)^(3/2))
  elseif (r.D₀==0) & (D==0)
    return sqrt(1-α^2)/(π*α^(3/2)*(r.β/α)^(3/2)) # lim D->0 D/D0 = 1
  end
end
# Perol&Bhat2016 : 1/α  or  Harsha's notes : 1/α^2 ???
function compute_c2(d::Rheology,D)
  α = cosd(d.ψ)
  (d.D₀==0) && (return 0.0)
  return (sqrt(1 - α^2)/α^2) * (d.D₀^(2/3)/(1 - D^(2/3))) ###
end

function compute_c3(d::Rheology,D)
  α = cosd(d.ψ)
  #(α < 0) && (α = -α) #### careful with that
  @assert α > 0
  (d.D₀ == D) && (return 0.0)
  @assert (D/d.D₀) >= 1
  return (2sqrt(α)/π)*((D/d.D₀)^(1/3) - 1)^(1/2)
end

function compute_c1c2c3(r,D)
  c1 = compute_c1(r,D)
  c2 = compute_c2(r,D)
  c3 = compute_c3(r,D)
  return c1, c2, c3
end
# eq 15 Bhat2012 (A1 : *c2*c3), Perol&Bhat2016 (A1 : ...*c2)*c3):
# Perol&Bhat2016 is the corrected version, and the one implemented
compute_A(r::Rheology,c1,c2,c3) = r.μ*c1 + (1.0 + r.μ*c2)*c3
compute_B(c1,c2,c3) = c1 + c2*c3

function compute_AB(r::Rheology,c1,c2,c3)
  A = compute_A(r,c1,c2,c3)
  B = compute_B(c1,c2,c3)
  return A, B
end
compute_AB(r::Rheology,D) = compute_AB(r,compute_c1c2c3(r,D)...)

# eq 11 in Harsha's notes :
compute_A1(r::Rheology,A) = A * sqrt((π*r.D₀*(1 - r.ν))/cosd(r.ψ)^3)
compute_B1(r::Rheology,B) = B * sqrt((π*r.D₀*(1 - r.ν))/cosd(r.ψ)^3)

function compute_A1B1(r::Rheology,A,B)
  A1 = compute_A1(r,A)
  B1 = compute_B1(r,B)
  return A1, B1
end
compute_A1B1(r::Rheology,D) = compute_A1B1(r,compute_AB(r,D)...)

function compute_a1(B1,Γ)
  return (1/Γ)*(1 + B1^2/2)
end

function compute_b1(A1,B1,Γ)
  return -(1/Γ)*((A1*B1)/2)
end

function compute_b2(r::Rheology,A1,Γ)
  return (1/Γ)*(A1^2/2 + (3*(1-2r.ν))/(2*(1+r.ν)))
end

function get_τ(tensor::AbstractArray ; input=:s)
  (input == :s) && (return sqrt(0.5 * tensor ⊡ tensor))
  if input == :σ
    s = dev(tensor)
    return sqrt(0.5 * s ⊡ s)
  end
end

function compute_KI(r::Rheology,σ,τ,A,B)
  return (A*σ + B*τ) * sqrt(π*r.a)
end

function compute_KI(r::Rheology,σ,τ,D)
  c1, c2, c3 = compute_c1c2c3(r,D)
  A, B = compute_AB(r,c1,c2,c3)
  # println("c1 : ",c1)
  # println("c2 : ",c2)
  # println("c3 : ",c3)
  # println("σ : ",σ)
  # println("τ : ",τ)
  return (A*σ + B*τ) * sqrt(π*r.a)
end

function compute_KI(r::Rheology,σij::SymmetricTensor,D)
  A, B = compute_AB(r,D)
  p = 1/3 * tr(σij) # trial pressure, negative in compression
  sij = dev(σij) # trial deviatoric stress
  τ = get_τ(sij)
  return (A*p + B*τ) * sqrt(π*r.a)
end

function compute_Γ(r::Rheology,A₁,B₁)
  return (3*(1-2r.ν))/(2*(1+r.ν)) + (3*(1-2r.ν)*B₁^2)/(4*(1+r.ν)) + A₁^2/2
end

function compute_Γ(r::Rheology,D)
  c1 = compute_c1(r,D)
  c2 = compute_c2(r,D)
  c3 = compute_c3(r,D)
  A = compute_A(r,c1,c2,c3)
  B = compute_B(c1,c2,c3)
  A₁ = compute_A1(r,A)
  B₁ = compute_B1(r,B)
  ν = r.ν
  return (3*(1-2ν))/(2*(1+ν)) + (3*(1-2ν)*B₁^2)/(4*(1+ν)) + A₁^2/2
end

function compute_dc1dD(r::Rheology,D)
  α = cosd(r.ψ)
  term1 = (-sqrt(1-α^2))/(2*π*α^(3/2)*D^(2/3)*r.D₀^(1/3))
  term2 = ((D/r.D₀)^(1/3) - 1 + (r.β/α))^(-5/2)
  return term1 * term2
end

function compute_dc2dD(r::Rheology,D)
  α = cosd(r.ψ)
  term1 = (2*sqrt(1-α^2)*r.D₀^(2/3))/(3*α^2*D^(1/3))
  term2 = (1 - D^(2/3))^(-2)
  return term1 * term2
end

function compute_dc3dD(r::Rheology,D)
  α = cosd(r.ψ)
  term1 = (sqrt(α))/(3*π*D^(2/3)*r.D₀^(1/3))
  term2 = ((D/r.D₀)^(1/3) - 1)^(-1/2)

  # ensure term2 isn't Infinity
  (term2 == Inf) && (term2 = 1e9)
  return term1 * term2
end

function compute_dA1dD(r::Rheology,dc1dD,dc2dD,dc3dD,c2,c3)
  α = cosd(r.ψ)
  term1 = sqrt((π*r.D₀*(1 - r.ν))/(α^3))
  term2 = r.μ*dc1dD + dc3dD + r.μ*c2*dc3dD + r.μ*c3*dc2dD
  return term1 * term2
end

function compute_dB1dD(r::Rheology,dc1dD,dc2dD,dc3dD,c2,c3)
  α = cosd(r.ψ)
  term1 = sqrt((π*r.D₀*(1 - r.ν))/(α^3))
  term2 = dc1dD + c2*dc3dD + c3*dc2dD
  return term1 * term2
end

function compute_dΓdD(r::Rheology,A1,B1,dA1dD,dB1dD)
  return ((3*(1-2r.ν))/(2*(1+r.ν)))*B1*dB1dD + A1*dA1dD
end

function compute_dA1dt(r::Rheology,dc1dt,dc2dt,dc3dt,c2,c3)
  α = cosd(r.ψ)
  term1 = sqrt((π*r.D₀*(1 - r.ν))/(α^3))
  term2 = r.μ*dc1dt + dc3dt + r.μ*c2*dc3dt + r.μ*c3*dc2dt
  return term1 * term2
end

function compute_dB1dt(r::Rheology,dc1dt,dc2dt,dc3dt,c2,c3)
  α = cosd(r.ψ)
  term1 = sqrt((π*r.D₀*(1 - r.ν))/(α^3))
  term2 = dc1dt + c2*dc3dt + c3*dc2dt
  return term1 * term2
end

function compute_dΓdt(r::Rheology,A1,B1,dA1dt,dB1dt)
  return ((3*(1-2r.ν))/(2*(1+r.ν)))*B1*dB1dt + A1*dA1dt
end

# function compute_dΓdt(p::RockParams,A1,B1,dA1dt,dB1dt)
#     return ((3*(1-2p.ν))/(2*(1+p.ν)))*B1*dB1dt + A1*dA1dt
# end

function compute_da1dt(B1,Γ,dB1dt,dΓdt)
  return -(dΓdt/Γ^2)*(1 + B1^2/2) + (B1*dB1dt)/Γ
end

function compute_db1dt(A1,B1,Γ,dA1dt,dB1dt,dΓdt)
  return (dΓdt/Γ^2)*((A1*B1)/2) - (1/2Γ)*(dA1dt*B1 + A1*dB1dt)
end

function compute_db2dt(r::Rheology,A1,Γ,dA1dt,dΓdt)
  return -(dΓdt/Γ^2)*(A1^2/2 + (3*(1-2r.ν))/(2*(1+r.ν))) + (2/Γ)*A1*dA1dt
end

function compute_dσdt(r::Rheology,a1,b1,da1dt,db1dt,ϵ,γ,dϵdt,dγdt)
  return r.G * (da1dt*ϵ + a1*dϵdt + db1dt*γ + b1*dγdt)
end

function compute_dτdt(r::Rheology,b1,b2,db1dt,db2dt,ϵ,γ,dϵdt,dγdt)
  return r.G * (db1dt*ϵ + b1*dϵdt + db2dt*γ + b2*dγdt)
end

function compute_dDdl(r::Rheology,D)
  return (3*D^(2/3)*r.D₀^(1/3))/(cosd(r.ψ)*r.a)
end

function compute_subcrit_damage_rate(r::Rheology, KI, D ; vmax = :l̇₀)
  ((KI <= 0) || (D >= 1)) && (return 0.0)
  ρ = 2700 ##### TODO better
  Vs = sqrt(r.G/ρ)
  Vr = Vs * (0.862 + 1.14r.ν)/(1 + r.ν)

  dDdl = compute_dDdl(r,D) # damage derivative wrt crack length
  dlmax = (vmax == :l̇₀) ? r.l̇₀ : Vr
  dldt = min(r.l̇₀*(KI/r.K₁c)^(r.n),dlmax)  #Vr cracks growth rate
  #@debug dDdl * dldt
  @assert dDdl * dldt >= 0
  return dDdl * dldt
end

function compute_free_energy(r,D,ϵij)
  G = r.G
  ν = r.ν
  ϵ = tr(ϵij)
  e = dev(ϵij)
  γ = sqrt(2.0 * e ⊡ e)

  A1, B1 = compute_A1B1(r,D)
  Γ = compute_Γ(r,A1,B1)
  a1 = compute_a1(B1,Γ)
  b1 = compute_b1(A1,B1,Γ)
  b2 = compute_b2(r,A1,Γ)

  return G * (0.5*a1*ϵ^2 + 0.5*b2*γ^2 + b1*ϵ*γ)
end

function compute_σij(r,A1,B1,Γ,ϵij)
  # TODO make a visco elastic version of this function

  if all(x->x==0,ϵij)
      return Tensor{2,3}(ϵij)
  end

  G = r.G
  ν = r.ν
  Id = SymmetricTensor{2,3}(δ)
  # strain invariants
  ϵ = tr(ϵij)
  e = dev(ϵij)
  γ = sqrt(2.0 * e ⊡ e)

  # stress tensor calculation
  term1 = ( (3*(1-2ν))/(1+ν) + A1^2 - A1*B1*ϵ/γ ) * ϵij
  term2 = (3ν/(1+ν) + B1^2/2 - A1^2/3 + A1*B1*ϵ/(3γ)) * ϵ
  term3 = -A1*B1*γ/2
  return (G/Γ) * (term1 + (term2 + term3)*Id)
end
compute_σij(r::Rheology,D,ϵij) = D==0 ? compute_σij(r,0.0,0.0,compute_Γ(r,0.0,0.0),ϵij) : compute_σij(r,compute_A1B1(r,D)...,compute_Γ(r,D),ϵij)

#TODO : 
function compute_ϵ_oop(r,A1,B1,σᵢⱼ)

  σoop = σᵢⱼ[3,3]
  σ = 1/3*tr(σᵢⱼ)
  sᵢⱼ = dev(σᵢⱼ)
  τ = get_τ(sᵢⱼ)
  G = r.G
  ν = r.ν
  
  # stress tensor calculation
  term1 = ( 1 + A1*B1*σ/(2*τ) + B1^2/2) * σoop
  term2 = (3ν/(1+ν) + A1*B1*σ/(2*τ) - A1^2/3 + B1^2/2) * σ
  term3 = A1*B1*τ/3
  return (1/(2G)) * (term1 - term2 + term3)
end
compute_ϵ_oop(r,D,σᵢⱼ)=compute_ϵ_oop(r,compute_A1B1(r,D)...,σᵢⱼ)

function compute_ϵij(r,A1,B1,σᵢⱼ)

  σ = 1/3*tr(σᵢⱼ)
  sᵢⱼ = dev(σᵢⱼ)
  τ = get_τ(sᵢⱼ)
  G = r.G
  ν = r.ν
  Id = SymmetricTensor{2,3}(δ)
  # strain invariants 
  term1 = ( 1 + A1*B1*σ/(2*τ) + B1^2/2) * σᵢⱼ
  term2 = (3ν/(1+ν) + A1*B1*σ/(2*τ) - A1^2/3 + B1^2/2) * σ
  term3 = A1*B1*τ/3
  return (1/(2G)) * (term1 + (- term2 + term3)*Id)
end
compute_ϵij(r,D,σᵢⱼ) = D==0 ? compute_ϵij(r,0.0,0.0,σᵢⱼ) : compute_ϵij(r,compute_A1B1(r,D)...,σᵢⱼ)


function compute_damaged_stiffness_tensor(r::Rheology,ϵij,D)

  # unpack
  G = r.G
  ν = r.ν

  # Damage constants
  A₁,B₁ = compute_A1B1(r,D)
  Γ = compute_Γ(r,A₁,B₁)

  # strain invariants
  ϵ = tr(ϵij)
  e = dev(ϵij)
  γ = sqrt(2.0 * e ⊡ e)

  @assert !isnan(G)
  @assert !isnan(A₁)
  @assert !isnan(B₁)
  @assert !isnan(Γ)
  @assert !isnan(ϵ)
  @assert !isnan(γ)
  @assert !isnan((3*(1-2ν))/(2*(1+ν)))
  @assert !isnan(A₁^2/2)
  # println("A₁*B₁*ϵ : ", A₁*B₁*ϵ)
  # println("(2*γ) : ", (2*γ))
  #@assert !isnan(A₁*B₁*ϵ/(2*γ)) # returns NaN, because 0/0

  (γ == 0) && (γ += 1e-15)#zero(typeof(ϵij))) # TODO remove redundancy
  ϵ̂ = ϵij/γ

  # get stiffness factors
  if G/Γ == 0
      Cμ = 0.0
      Cλ = 0.0
      Cσ = 0.0
      Cσσ = 0.0
  else
      Cμ = G/Γ * ( (3*(1-2ν))/(2*(1+ν)) + A₁^2/2 - A₁*B₁*ϵ/(2*γ) )
      Cλ = G/Γ * ( 3*ν/(1+ν) + B₁^2/2 - A₁^2/3 + A₁*B₁*ϵ/γ + 2A₁*B₁*ϵ^3/(9γ^3) )
      Cσ = - G/Γ * ( A₁*B₁ + 2*A₁*B₁*ϵ^2/(3*γ^2) )
      Cσσ = G/Γ * (2A₁*B₁*ϵ/γ)
  end



  if isnan(Cμ)
      println("Cμ = ", Cμ)
      @assert !isnan(G/Γ)
      println("G/Γ = ", G/Γ)
      @assert !isnan( (3*(1-2ν))/(2*(1+ν)) + A₁^2/2 - A₁*B₁*ϵ/(2*γ) )
      println("second term Cμ = ", (3*(1-2ν))/(2*(1+ν)) + A₁^2/2 - A₁*B₁*ϵ/(2*γ) )
      @assert !isnan(G/Γ * ( (3*(1-2ν))/(2*(1+ν)) + A₁^2/2 - A₁*B₁*ϵ/(2*γ) ))
  end
  if isnan(Cλ)
      println("Cλ = ", Cλ)
  end
  @assert !isnan(Cσ)
  @assert !isnan(Cσσ)
  # functional form of the stiffness tensor
  C_func(i,j,k,l) = Cμ * ( δ(k,i)*δ(l,j) + δ(l,i)*δ(k,j) ) +
                    Cλ * ( δ(i,j)*δ(k,l) ) +
                    Cσ * ( ϵ̂[i,j]*δ(k,l) + δ(i,j)*ϵ̂[k,l] ) +
                    Cσσ * (ϵ̂[i,j]*ϵ̂[k,l])

  # assemble the tensor
  return SymmetricTensor{4,3}(C_func)
end

## 
function compute_ϵ̇ij_2(r,D,Ḋ,σᵢⱼ,σ̇ᵢⱼ; damaged_allowed=true)

  # Convert in case automatic differentiation supplies a Matrix to the function
  if σ̇ᵢⱼ isa Matrix
    sym_test = σ̇ᵢⱼ - σ̇ᵢⱼ'
    if all(.≈(0),sym_test)
      σ̇ᵢⱼ = SymmetricTensor{2,3}(σ̇ᵢⱼ)
    else
      println("printed :       ",[elem.value.value for elem in sym_test])
    end
  end

  𝕀 = SymmetricTensor{2,3}(δ) # Second order identity tensor

  # stress at previous timestep
  σ = (1/3)*(tr(σᵢⱼ))
  sᵢⱼ = σᵢⱼ - σ*𝕀
  τ = get_τ(sᵢⱼ)

  # stress derivatives
  σ̇ = (1/3)*(tr(σ̇ᵢⱼ))
  τ̇ = sᵢⱼ ⊡ σ̇ᵢⱼ / (2*τ)

  #ṡᵢⱼ = σ̇ᵢⱼ - σ̇*𝕀
  #τ̇ = get_τ(ṡᵢⱼ)

  #@assert τ̇ == τ̇_2
  #damage constants and derivatives
  if r.D₀ == 0
    A1, B1 = 0.0, 0.0
    dA1dD, dB1dD = 0.0, 0.0
  else
    c1, c2, c3 = compute_c1c2c3(r,D)
    A, B = compute_AB(r,c1,c2,c3)
    A1, B1 = compute_A1B1(r,A,B)
    dc1dD = compute_dc1dD(r,D)
    dc2dD = compute_dc2dD(r,D)
    dc3dD = compute_dc2dD(r,D)
    dA1dD = compute_dA1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)
    dB1dD = compute_dB1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)
  end

  
  
  t1 = λ₁(A1,B1,σ,τ)*σ̇ᵢⱼ - ( λ₂(r,A1,B1,σ,τ)*σ̇ - (1/3)*A1*B1*τ̇ )*𝕀
  t2 = ( dλ₁dσ(A1,B1,τ)*σ̇ + dλ₁dτ(A1,B1,σ,τ)*τ̇ )*sᵢⱼ
  if damaged_allowed
    t3 = dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ)*Ḋ*σᵢⱼ - ( dλ₂dD(A1,B1,dA1dD,dB1dD,σ,τ)*Ḋ*σ - (1/3)*Ḋ*(dA1dD*B1 + A1*dB1dD)*τ )*𝕀
  else
    t3 = zero(Tensor{2, 3})
  end
  ϵ̇ᵢⱼ = 1/(2r.G) * (t1 + t2 + t3)

  return convert(SymmetricTensor{2,3,eltype(ϵ̇ᵢⱼ)},ϵ̇ᵢⱼ)
end
function compute_ϵ̇ij(r,D,σij,σijnext,Δt ; damaged_allowed=true)

  # Convert in case automatic differentiation supplies a Matrix to the function
  if σijnext isa Matrix
    sym_test = σijnext - σijnext'
    if all(.≈(0),sym_test)
      σijnext = SymmetricTensor{2,3}(σijnext)
    else
      println("printed :       ",[elem.value.value for elem in sym_test])
    end
  end

  𝕀 = SymmetricTensor{2,3}(δ) # Second order identity tensor

  # stress at previous timestep
  σ = (1/3)*(tr(σij))
  sij = σij - σ*𝕀
  τ = get_τ(sij)

  # stress derivatives
  σ̇ij = (σijnext-σij)/Δt
  σ̇ = (1/3)*(tr(σ̇ij))
  ṡᵢⱼ = σ̇ij - σ̇*𝕀
  #τ̇_2 = get_τ(ṡᵢⱼ)
  τ̇ = sij ⊡ σ̇ij / (2*τ)
  #@show(τ̇,τ̇_i)

  # initialize damage rate
  Ḋ = 0.0

  #damage constants and derivatives
  if r.D₀ == 0
    A1, B1 = 0.0, 0.0
    dA1dD, dB1dD = 0.0, 0.0
  else
    c1, c2, c3 = compute_c1c2c3(r,D)
    A, B = compute_AB(r,c1,c2,c3)
    A1, B1 = compute_A1B1(r,A,B)
    dc1dD = compute_dc1dD(r,D)
    dc2dD = compute_dc2dD(r,D)
    dc3dD = compute_dc2dD(r,D)
    dA1dD = compute_dA1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)
    dB1dD = compute_dB1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)
  end

  
  
  t1 = λ₁(A1,B1,σ,τ)*σ̇ij - ( λ₂(r,A1,B1,σ,τ)*σ̇ - (1/3)*A1*B1*τ̇ )*𝕀
  t2 = ( dλ₁dσ(A1,B1,τ)*σ̇ + dλ₁dτ(A1,B1,σ,τ)*τ̇ )*sij
  if damaged_allowed
    KI = compute_KI(r,(σijnext+σij)/2,D) # compute KI at intermediate stress : TO TEST.
    Ḋ = compute_subcrit_damage_rate(r,KI,D)
    t3 = dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ)*Ḋ*σij - ( dλ₂dD(A1,B1,dA1dD,dB1dD,σ,τ)*Ḋ*σ - (1/3)*Ḋ*(dA1dD*B1 + A1*dB1dD)*τ )*𝕀
  else
    t3 = zero(Tensor{2, 3})
  end
  ϵ̇ij = 1/(2r.G) * (t1 + t2 + t3)

  #ϵ̇ij = insert_into(ϵ̇ij, -1e-5, (1,1)) ######## !!!!!!!!
  return ϵ̇ij, Ḋ
end

λ₁(A1,B1,σ,τ) =  1 + A1*B1*σ/(2*τ) + B1^2/2
λ₂(r,A1,B1,σ,τ) = 3*r.ν/(1+r.ν) + A1*B1*σ/(2*τ) - A1^2/3 + B1^2/2
dλ₁dσ(A1,B1,τ) = A1*B1/(2*τ)
dλ₁dτ(A1,B1,σ,τ) = -A1*B1*σ/(2*τ^2)
dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ) = (dA1dD*B1 + A1*dB1dD)*σ/(2*τ) + B1*dB1dD
dλ₂dD(A1,B1,dA1dD,dB1dD,σ,τ) = dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ) - (2/3)*A1*dA1dD

