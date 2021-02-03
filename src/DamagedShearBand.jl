module DamagedShearBand

  using Tensors
  using StaticArrays
  using ForwardDiff
  using Optim
  
  include("types.jl")
  include("utils.jl")
  include("damaged_rheology.jl")
  include("analysis/analysis.jl")
  include("simulations_helpers.jl")
  include("integrators.jl")

  #export Rheology
  #export set_plane_strain_oop_stress, get_damage_onset

end
