module DamagedShearBand

  using Reexport
  @reexport using Tensors
  using DiffResults: GradientResult, value, gradient
  using ForwardDiff: gradient!
  using Optim

  include("types.jl")
  include("damaged_rheology.jl")
  include("integrators.jl")
  include("analysis.jl")
  include("simulations_helpers.jl")

  export Rheology
  export set_plane_strain_oop_stress, get_damage_onset

end
