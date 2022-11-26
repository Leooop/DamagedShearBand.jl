using Base: @kwdef
const R = Real
@kwdef struct Rheology{t1<:R,t2<:R,t3<:R,t4<:R,t5<:R,t6<:R,t7<:R,t8<:R,t9<:R,t10<:R,t11<:R,t12<:R}
  G::t1 = 30e9
  ν::t2 = 0.3
  μ::t3 = 0.6 # Friction coef
  β::t4 = 0.1 # Correction factor
  K₁c::t5 = 1.74e6 # Critical stress intensity factor (Pa.m^(1/2))
  a::t6 = 1e-3 # Initial flaw size (m)
  ψ::t7 = atand(0.6)# crack angle to the principal stress (degree)
  D₀::t8 = 0.1# Initial flaw density
  n::t9 = 34.0 # Stress corrosion index
  l̇₀::t10 = 0.24 # Ref. crack growth rate (m/s)
  H::t11 = 50e3 # Activation enthalpy (J/mol)
  A::t12 = 5.71 # Preexponential factor (m/s)
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

Base.Broadcast.broadcastable(r::Rheology) = Ref(r)

function Base.show(io::IO, ::MIME"text/plain", r::Rheology)
  print(io, "Rheology instance with fields :\n",
  "\t├── G (shear modulus)                             : $(r.G)\n",
  "\t├── ν (poisson ratio)                             : $(r.ν)\n",
  "\t├── μ (flaws friction coefficient)                : $(r.μ)\n",
  "\t├── β (correction factor)                         : $(r.β)\n",
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
  adaptative_maxrecursions::Int = 50
end

@kwdef struct OutputParams{T1<:Maybe(Real),T2<:Maybe(Integer),T3<:Maybe(Real),T4<:Maybe(Integer)}
  print_frequency::T1=nothing
  print_period::T2=1
  save_frequency::T3=nothing
  save_period::T4=1
  last_tsim_printed::Ref{Float64}=Ref(0.0)
  last_tsim_saved::Ref{Float64}=Ref(0.0)
end

mutable struct Flags
  print::Bool
  save::Bool
  bifurcation::Bool
  acceptable_error::Bool
  nan::Bool
  nan1::Bool
  nan2::Bool
  Flags()=new(false,false,false,false,false,false,false)
end

struct Params{T1,T2,T3,T4,TI,TE}
  solver::SolverParams{TI,TE}
  output::OutputParams{T1,T2,T3,T4}
  flags::Flags
end
Params(s::SolverParams,o::OutputParams) = Params(s,o,Flags())
Params(o::OutputParams,s::SolverParams) = Params(s,o,Flags())
Params() = Params(SolverParams(),OutputParams(),Flags())

@kwdef mutable struct DiffEqParams{R<:Rheology,P<:Params,DU<:AbstractVector{<:Real},A<:AbstractVector{<:Real}}
  r::R = Rheology()
  mp::P = Params()
  du::DU = nothing
  scalars::A = nothing
  allow_Ḋᵒ::Bool = true
  terminate_flag::Bool = false
end

# Model type ? 2points, 1point