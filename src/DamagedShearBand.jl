module DamagedShearBand

  using Reexport
  @reexport using Tensors
  import DiffResults
  import ForwardDiff
  using Optim

  include("types.jl")
  include("utils.jl")
  include("damaged_rheology.jl")
  include("integrators.jl")
  include("analysis.jl")
  include("simulations_helpers.jl")

  #export Rheology
  #export set_plane_strain_oop_stress, get_damage_onset

end
