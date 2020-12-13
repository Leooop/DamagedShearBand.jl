# DamagedShearBand

A [Julia](http://julialang.org) package for simgle or dual points description of shear banding in a damaged elastic 2D plane strain (x-y plane) material. Damage theory used here is derived from [*Ashby & Sammis model (1990)*](https://link.springer.com/article/10.1007/BF00878002) and uses a micromechanical thermodynamical framework to express constitutive laws. 

This package is a PhD WIP. The examples may not be up to date at all time.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Leooop.github.io/DamagedShearBand.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Leooop.github.io/DamagedShearBand.jl/dev)
[![Build Status](https://github.com/Leooop/DamagedShearBand.jl/workflows/CI/badge.svg)](https://github.com/Leooop/DamagedShearBand.jl/actions)

#### Author
- Léo Petit, École Normale Supérieure de Paris, France.

#### License

`DamagedShearBand` is licensed under the [MIT license](./LICENSE.md).

#### Installation

`DamagedShearBand` is not a registered package and can be installed from the package REPL with
```julia
pkg> add https://github.com/Leooop/DamagedShearBand.jl.git
```
or similarly with
```julia
julia> using Pkg ; Pkg.add("https://github.com/Leooop/DamagedShearBand.jl.git")
```
Requires Julia v1.3 or higher

#### Examples

See the examples folder for how to plot simulation output or how to export it to MATLAB .mat format.

##### Single point deformation description with imposed constant strain rate along x direction (ϵ̇₁₁) and constant stress along y direction (σ₂₂).

```julia
using DamagedShearBand ; const DSB = DamagedShearBand

# imposed values
ϵ̇₁₁ = -1e-4
σ₂₂ = -1e-6 # negative in compression

# initial higher compressive stress multiplier (S = σ₁₁/σ₂₂)
S = 1.0

# Create a Rheology type instance containing elastic moduli and damage parameters. Change default values by  supplying keywords arguments.
r = DSB.Rheology(D₀=0.3) 

# initial damage
D_i = r.D₀

# time integration parameters :
tspan = (0.0,20) # initial and final simulation time
Δt_i = 0.1 # initial timestep

# Build stress and strain tensors 
σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₂₂,D_i ; abstol=1e-15) # takes care of the plane strain constraint by solving non linear out of plane strain wrt σ₃₃ using Newton algorithm
ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)


# Integrate over time, by solving for the unknowns [σ₁₁next, σ₃₃next, ϵ₂₂next] at each timestep
# -- e₀ is the absolute tolerance used to adapt time stepping, may be a scalar or a NamedTuple with keys (D,σ,ϵ)
# -- abstol is the absolute tolerance on the unknowns in the Newton solver 
t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec = DSB.adaptative_time_integration(r,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇₁₁,Δt_i,tspan ; abstol=1e-12, time_maxiter=nothing, newton_maxiter=100, e₀=(D=1e-4, σ=1.0, ϵ=1e-6))

```