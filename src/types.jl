using Base: @kwdef

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

function Rheology(r::Rheology, kw::NamedTuple)
  values_dict = Dict{Symbol,Float64}()
  for sym in propertynames(r)
    values_dict[sym] = isdefined(kw,sym) ? getproperty(kw,sym) : getproperty(r,sym)
  end
  return Rheology(; G = values_dict[:G],
                    ν = values_dict[:ν],
                    μ = values_dict[:μ],
                    β = values_dict[:β],
                    K₁c = values_dict[:K₁c],
                    a = values_dict[:a],
                    ψ = values_dict[:ψ],
                    D₀ = values_dict[:D₀],
                    n = values_dict[:n],
                    l̇₀ = values_dict[:l̇₀],
                    H = values_dict[:H],
                    A = values_dict[:A])
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

Maybe(T)=Union{T,Nothing}

@kwdef struct SolverParams{TI<:Maybe(Integer),TE<:Union{Real,NamedTuple}}
  newton_abstol::Float64 = 1e-12
  newton_maxiter::Int = 100
  time_maxiter::TI = nothing
  e₀::TE = 1e-12
end

@kwdef struct OutputParams{T1<:Maybe(Real),T2<:Maybe(Integer),T3<:Maybe(Real),T4<:Maybe(Integer)}
  print_frequency::T1=nothing
  print_period::T2=1
  save_frequency::T3=nothing
  save_period::T4=1
end

#OutputParams(::T1,::T2,::T3,::T4) where{T1<:Real,T2<:Real}

struct Params{T1,T2,T3,T4,TI,TE}
  solver::SolverParams{TI,TE}
  output::OutputParams{T1,T2,T3,T4}
end

Params(o::OutputParams,s::SolverParams) = Params(s::SolverParams,o::OutputParams)

# Model type ? 2points, 1point