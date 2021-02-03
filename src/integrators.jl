function RK4!(du, u, p, update_du!, Δt::T) where T<:AbstractFloat

  # initialize derivative ponderated sum
  du_sum = zeros(eltype(u),size(u))

  # increment derivative ponderated sum
  update_du!(du,u,p,0.0)
  du_sum .+= du

  update_du!(du,u.+((Δt/2.0).*du),p,0.0)
  du_sum .+= 2.0 .* du

  update_du!(du,u.+(Δt/2.0).*du,p,0.0)
  du_sum .+= 2.0 .* du

  update_du!(du,u.+Δt.*du,p,0.0)
  du_sum .+= du

  # update u with the mean of the ponderated sum over Δt
  return (Δt/6).*du_sum .+ u
end

function RK4(u, p, update_du, Δt::T) where T<:AbstractFloat

  # increment derivative ponderated sum
  du1 = update_du(u,p,0.0)
  du2 = update_du(u.+((Δt/2.0).*du1),p,0.0)
  du3 = update_du(u.+((Δt/2.0).*du2),p,0.0)
  du4 = update_du(u.+Δt.*du3,p,0.0)
  du_sum = du1 + 2*du2 + 2*du3 + du4

  # update u with the mean of the ponderated sum over Δt
  return (Δt/6)*du_sum + u
end

function timestep_RK4_adaptative!(du, u, p, update_du!, Δt::T, e₀::T) where T<: AbstractFloat
  u1 = RK4!(du, u, p, update_du!, Δt)
  #println("RK4 u1",u1)
  uint = RK4!(du, u, p, update_du!, Δt/2)
  #println("RK4 uint",uint)
  u2 = RK4!(du, uint, p, update_du!, Δt/2)
  #println("RK4 u2",u2)

  e = abs(u2[1]-u1[1]) # error calculated on damage only
  if e < e₀
      return u1, Δt, min(Δt*abs(e₀/e),Δt*2)
  else
      timestep_RK4_adaptative(du, u, p, update_du!, float(Δt*abs(e₀/e)^2), e₀)
  end
end

function timestep_RK4_adaptative_2points(u, p, update_du, Δt::T, e₀::T) where T<: AbstractFloat
  u1 = RK4(u, p, update_du, Δt)
  #println("RK4 u1",u1)
  uint = RK4(u, p, update_du, Δt/2)
  #println("RK4 uint",uint)
  u2 = RK4(uint, p, update_du, Δt/2)
  #println("RK4 u2",u2)
  Ṡ1, σ̇ⁱξξ1, σ̇ⁱoop1, ϵ̇ⁱηη1, Ḋᵒ1, Ḋⁱ1 = u1 
  Ṡ2, σ̇ⁱξξ2, σ̇ⁱoop2, ϵ̇ⁱηη2, Ḋᵒ2, Ḋⁱ2 = u2

  eD = max(abs((Ḋⁱ2-Ḋⁱ1)/Ḋⁱ2),abs((Ḋᵒ2-Ḋᵒ1)/Ḋᵒ2))
  eS = abs((Ṡ2-Ṡ1)/Ṡ2)
  eσ = max(abs((σ̇ⁱξξ2-σ̇ⁱξξ1)/σ̇ⁱξξ2),abs((σ̇ⁱoop2-σ̇ⁱoop1)/σ̇ⁱoop2))
  eϵ = abs((ϵ̇ⁱηη2-ϵ̇ⁱηη1)/ϵ̇ⁱηη2)
  e_vec = Vec(eD,eS,eσ,eϵ)
  e_normalized = e_vec./e₀
  e_norm_max, id_e_norm_max = findmax(e_normalized)

  if e_norm_max < e₀
      return u2, Δt, min(Δt*abs(e₀/e_norm_max),Δt*2)
  else
      timestep_RK4_adaptative_2points(u, p, update_du, float(Δt*abs(e₀/e_norm_max)^2), e₀)
  end
end

function compute_RK4_adaptative(u0,p,update_du,tspan,Δt_ini,e₀)
  u_vec = Vector{SVector{6,Float64}}()
  t_vec = Float64[tspan[1]]
  #u_next = similar(u0)
  cohesion_loss_flag = false
  push!(u_vec,u0)
  next_Δt = Δt_ini
  while t_vec[end] < tspan[2]
      (t_vec[end]+next_Δt > tspan[2]) && (next_Δt = tspan[2] - t_vec[end]) # prevents time overshoot
      u_next, used_Δt, next_Δt = timestep_RK4_adaptative_2points(u_vec[end], p, update_du,next_Δt,e₀)
      push!(u_vec,u_next)
      push!(t_vec,t_vec[end]+used_Δt)

      # break if D >= 1
      if any(@view(u_next[5:6]) .>= 0.98)
          cohesion_loss_flag = true
          break
      end
  end
  return t_vec, u_vec, cohesion_loss_flag
end