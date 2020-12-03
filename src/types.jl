using Base: @kwdef

export Rheology

@kwdef struct Rheology
  G::Float64 = 30e9
  ν::Float64 = 0.3
  μ::Float64 = 0.6 # Friction coef
  β::Float64 = 0.1 # Correction factor
  K₁c::Float64 = 1.74e6 # Critical stress intensity factor (Pa.m^(1/2))
  a::Float64 = 1e-3 # Initial flaw size (m)
  ψ::Float64 = atand(0.6)# crack angle to the principal stress (radians)
  D₀::Float64 = 0.1# Initial flaw density
  n::Float64 = 34.0 # Stress corrosion index
  l̇₀::Float64 = 0.24 # Ref. crack growth rate (m/s)
  H::Float64 = 50e3 # Activation enthalpy (J/mol)
  A::Float64 = 5.71 # Preexponential factor (m/s)
end

function Base.show(io::IO, ::MIME"text/plain", r::Rheology)
  print(io, "Rheology instance with fields :\n",
  "\t├── G (shear modulus)                             : $(r.G)\n",
  "\t├── ν (poisson ratio)                             : $(r.ν)\n",
  "\t├── μ (flaws friction coefficient)                : $(r.μ)\n",
  "\t├── β (correction factor)                         : $(r.μ)\n",
  "\t├── K₁c (fracture toughness)                      : $(r.K₁c)\n",
  "\t├── a (flaws radius)                              : $(r.a)\n",
  "\t├── ψ (flaws angle wrt σ₁ in degree)              : $(r.ψ)\n",
  "\t├── D₀ (initial damage, constrains flaws density) : $(r.D₀)\n",
  "\t├── n (stress corrosion index)                    : $(r.n)\n",
  "\t├── l̇₀ (ref crack growth rate in m/s)             : $(r.l̇₀)\n",
  "\t├── H (Activation enthalpy (J/mol))               : $(r.H)\n",
  "\t└── A (preexponetial factor (m/s))                : $(r.A)\n")
end