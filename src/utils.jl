δ(i,j) = i == j ? 1.0 : 0.0
Isym_func(i,j,k,l) = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
Isymdev_func(i,j,k,l) = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) - 1.0/3.0*δ(i,j)*δ(k,l)

function insert_σoop(σᵢⱼ::Tensor{2,3,T},σoop_guess) where T
  σᵢⱼ = Tensor{2,3,T}([σᵢⱼ[1,1] σᵢⱼ[1,2] σᵢⱼ[1,3] ; σᵢⱼ[2,1] σᵢⱼ[2,2] σᵢⱼ[2,3] ; σᵢⱼ[3,1] σᵢⱼ[3,2] σoop_guess])
end
function insert_σoop(σᵢⱼ::SymmetricTensor{2,3,T},σoop_guess) where T
  σᵢⱼ = SymmetricTensor{2,3,T}([σᵢⱼ[1,1] σᵢⱼ[1,2] σᵢⱼ[1,3] ; σᵢⱼ[2,1] σᵢⱼ[2,2] σᵢⱼ[2,3] ; σᵢⱼ[3,1] σᵢⱼ[3,2] σoop_guess])
end

function insert_into(tensor::SymmetricTensor{2,S},values,indices) where{S}
  return SymmetricTensor{2,S}((i,j) -> map_id_to_value(tensor,values,indices,i,j))
end

function map_id_to_value(tensor::SymmetricTensor,values,indices,i,j)
  ((i,j) == indices) | ((j,i) == indices) && return values
  if (i,j) in indices
    ind = findfirst(indices .== Ref((i,j)))
    return values[only(ind)]
  elseif (j,i) in indices
    ind = findfirst(indices .== Ref((j,i)))
    return values[only(ind)]
  else
    return tensor[i,j]
  end
end

function filter_offdiagonal(σᵢⱼ ; tol = 1e-5)
  (abs(σᵢⱼ[1,2]) <= tol) && (σᵢⱼ = insert_into(σᵢⱼ, 0.0, (1,2)))
  return σᵢⱼ
end

function set_print_flag!(p,iter,tsim)
  flags = p.flags
  if (iter==1) | print_flag_condition(p,iter,tsim)
    flags.print = true
    p.output.last_tsim_printed[] = tsim
  elseif flags.nan & (p.output.last_tsim_printed[] != tsim)
    flags.print = true
    p.output.last_tsim_printed[] = tsim
  else 
    flags.print = false
  end
  return nothing
end

function set_save_flag!(p,iter,tsim)
  flags = p.flags
  if (iter==1) | save_flag_condition(p,iter,tsim)
    flags.save = true
    p.output.last_tsim_saved[] = tsim
  elseif flags.nan & (p.output.last_tsim_saved[] != tsim)
    flags.save = true
    p.output.last_tsim_saved[] = tsim
  else 
    flags.save = false
  end
  return nothing
end

print_flag_condition(p::Params{TF,Nothing},iter,tsim) where {TF<:Real} = ( (tsim-p.output.last_tsim_printed[]) >= (1/p.output.print_frequency) )
print_flag_condition(p::Params{Nothing,TP},iter,tsim) where {TP<:Real} = (iter%p.output.print_period == 0)
function print_flag_condition(p::Params{TF,TP},iter,tsim) where {TF<:Real,TP<:Real}
  return ( (tsim-p.output.last_tsim_printed[]) >= (1/p.output.print_frequency) ) || (iter%p.output.print_period == 0)
end

save_flag_condition(p::Params{T1,T2,TF,Nothing},iter,tsim) where {T1,T2,TF<:Real} = ( (tsim-p.output.last_tsim_saved[]) >= (1/p.output.save_frequency) )
save_flag_condition(p::Params{T1,T2,Nothing,TP},iter,tsim) where {T1,T2,TP<:Real} = (iter%p.output.save_period == 0)
function save_flag_condition(p::Params{T1,T2,TF,TP},iter,tsim) where {T1,T2,TF<:Real,TP<:Real}
  return ( (tsim-p.output.last_tsim_saved[]) >= (1/p.output.save_frequency) ) || (iter%p.output.save_period == 0)
end

print_time_iteration(iter,tsim) = print("------","\n","time iteration $iter : $tsim","\n","------")

function print_nans_error(p::Params,Δt)
    flags = p.flags
    if flags.nan1
        flags.nan = true
        @error "integration over Δt=$(Δt) produced NaNs."
    elseif flags.nan2
        flags.nan = true
        @error "integration over Δt/2=$(Δt/2) twice produced NaNs."
    end
end

