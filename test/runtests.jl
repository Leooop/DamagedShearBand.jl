using DamagedShearBand; const DSB = DamagedShearBand
using Tensors
using StaticArrays
using Test

# Data :
S, σ₃ = 5, -1e6
r = DSB.Rheology()
D = r.D₀
σᵢⱼ = SymmetricTensor{2,3}([S*σ₃ 0 0 ; 0 σ₃ 0 ; 0 0 r.ν*(S+1)*σ₃])

@testset "get_plane_strain_oop_stress" begin
    atol=1e-12
    σᵢⱼ_cor = DSB.set_plane_strain_oop_stress(σᵢⱼ, r, D ; abstol=atol, maxiter=100, σoop_guess=nothing)
    ϵoop = DSB.compute_ϵ_oop(r,D,σᵢⱼ_cor)
    @test abs(ϵoop)<=atol
end

@testset "get_damage_onset" begin
    Sc = DSB.get_damage_onset(r,σ₃)

    # KI at damage onset
    σᵢⱼ_onset = DSB.build_principal_stress_tensor(r,Sc,σ₃)
    KI_onset = DSB.compute_KI(r, σᵢⱼ_onset, r.D₀)
    @test KI_onset < 1e-10

    # KI after damage onset
    σᵢⱼ_up = DSB.build_principal_stress_tensor(r,Sc+0.1,σ₃)
    KI_up = DSB.compute_KI(r, σᵢⱼ_up, r.D₀)
    @test KI_up > 0

    # KI before damage onset
    σᵢⱼ_down = DSB.build_principal_stress_tensor(r,Sc-0.1,σ₃)
    KI_down = DSB.compute_KI(r, σᵢⱼ_down, r.D₀)
    @test KI_down < 0
end

@testset "band_coords" begin
    σ0 = DSB.band_coords(σᵢⱼ,0) # plane is vertical, 
    @test σ0[1,1] == σᵢⱼ[2,2]
    σm90 = DSB.band_coords(σᵢⱼ,-90) # plane is horizontal, 
    @test all(σm90 .== σᵢⱼ)
    σ90 = DSB.band_coords(σᵢⱼ,90) # plane is horizontal, 
    @test all(σ90 .== σᵢⱼ)
    σ90 = DSB.band_coords(σᵢⱼ,90+360) # plane is horizontal, 
    @test all(σ90 .== σᵢⱼ)
    σ_rotated = DSB.band_coords(σᵢⱼ,30)
    σ_back = DSB.principal_coords(σ_rotated,30)
    @test σ_back ≈ σᵢⱼ
end

# @testset "solving using given ϵ̇11. D=D₀=0 test vs elastic rheology" begin
#     tspan = (0.0,1.0)
#     Δt = 0.1
#     ϵ̇11 = -1e-5
#     r = DSB.Rheology(D₀=0.0) # object containing elastic moduli and damage parameters.
#     D_i = r.D₀
#     σ₃ = -1e6
#     S = 1
    
#     n_iter_saved = 5000
#     n_iter_printed = 10
#     op = DSB.OutputParams(save_frequency = n_iter_saved/(tspan[2]-tspan[1]),
#                         save_period = nothing,
#                         print_frequency = n_iter_printed/(tspan[2]-tspan[1]),
#                         print_period = nothing)

#     sp = DSB.SolverParams(newton_abstol = 1e-12,
#                         newton_maxiter = 100,
#                         time_maxiter = nothing,
#                         e₀ = (D=1e-1, σ=100.0, ϵ=1e-3)) 

#     p = DSB.Params(op,sp)

#     σᵢⱼ_i = DSB.build_principal_stress_tensor(r,S,σ₃,D_i ; abstol=1e-15) # takes care of the plane strain constraint
#     ϵᵢⱼ_i = DSB.compute_ϵij(r,D_i,σᵢⱼ_i)
#     t_vec, σᵢⱼ_vec, ϵᵢⱼ_vec, D_vec = DSB.time_integration(r,p,σᵢⱼ_i,ϵᵢⱼ_i,D_i,ϵ̇11,Δt,tspan)

#     Ce = DSB.get_elastic_stiffness_tensor(r)
#     σᵢⱼ_vec_elast = Ref(Ce) .⊡ ϵᵢⱼ_vec

#     @test all([σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,1] for σᵢⱼ in σᵢⱼ_vec_elast])
#     @test all([σᵢⱼ[2,2] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[2,2] for σᵢⱼ in σᵢⱼ_vec_elast])
#     @test all([σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[3,3] for σᵢⱼ in σᵢⱼ_vec_elast])
#     @test all([σᵢⱼ[1,2] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,2] for σᵢⱼ in σᵢⱼ_vec_elast])
#     @test all([σᵢⱼ[1,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[1,3] for σᵢⱼ in σᵢⱼ_vec_elast])
#     @test all([σᵢⱼ[2,3] for σᵢⱼ in σᵢⱼ_vec] .≈ [σᵢⱼ[2,3] for σᵢⱼ in σᵢⱼ_vec_elast])

    
#     # check tangent modulus along xx (σxx/ϵxx)
#     Emod = DSB.E_from_Gν(r.G,r.ν)
#     tan_mod_elast_model = (σᵢⱼ_vec_elast[end][1,1]-σᵢⱼ_vec_elast[1][1,1]) / (ϵᵢⱼ_vec[end][1,1]-ϵᵢⱼ_vec[1][1,1])
#     tan_mod_Dat0_model = (σᵢⱼ_vec[end][1,1]-σᵢⱼ_vec_elast[1][1,1]) / (ϵᵢⱼ_vec[end][1,1]-ϵᵢⱼ_vec[1][1,1])
#     tan_mod_theoretical = Emod/(1-r.ν^2)
#     @test abs(tan_mod_Dat0_model-tan_mod_theoretical) < 1e-2
#     @test abs(tan_mod_elast_model-tan_mod_theoretical) < 1e-2
# end

@testset "from principal stress rate to band orientation" begin
    θ = 60.0
    Ṡ = 1e-2
    Δt = 1
    σᵢⱼ_principal = DSB.build_principal_stress_tensor(r,S,σ₃,D ; abstol=1e-15)
    σᵢⱼ1 = DSB.compute_rotated_stress_rate_from_principal_coords(r,Ṡ,σ₃,σᵢⱼ_principal,D,Δt,θ ; damaged_allowed=true)
    σᵢⱼ2 = DSB.compute_rotated_stress_rate_from_band_coords(r,Ṡ,σ₃,DSB.band_coords(σᵢⱼ_principal,θ),D,Δt,θ; damaged_allowed=true)
    @test all(σᵢⱼ1 .≈ σᵢⱼ2)
end

@testset "get_stress_deviation_from_far_field" begin
    σᵢⱼ = DSB.build_principal_stress_tensor(r,S,σ₃,D)
    σᵢⱼ_rotated_90 = DSB.insert_into(σᵢⱼ, (σᵢⱼ[1,1],σᵢⱼ[2,2]), ((2,2),(1,1))) # swap two principals in plane stresses
    σᵢⱼ_rotated_30 = DSB.band_coords(σᵢⱼ,60)
    angle_90 = DSB.get_stress_deviation_from_far_field(σᵢⱼ_rotated_90 ; offdiagtol=1e-5, compression_axis=:x)
    angle_30 = DSB.get_stress_deviation_from_far_field(σᵢⱼ_rotated_30 ; offdiagtol=1e-5, compression_axis=:x)
    @test abs(angle_90) ≈ 90
    @test abs(angle_30) ≈ 30

    # # from simple shear test :
    Dᵢ, _ = DSB.get_KI_minimizer_D_S(r,σ₃)
    σᵢⱼ0 = DSB.build_principal_stress_tensor(r,2,σ₃,Dᵢ ; abstol=1e-15) # σ1 along x
    σᵢⱼ1 = DSB.build_principal_stress_tensor(r,0.5,σ₃,Dᵢ ; abstol=1e-15) # σ1 along y
    σᵢⱼ2p = SymmetricTensor{2,3,Float64}(SA[σᵢⱼ0[1,1], -0.1*σᵢⱼ0[1,1], 0, σᵢⱼ0[2,2], 0, σᵢⱼ0[3,3]]) # σᵢⱼ0 with low positive shear stress
    σᵢⱼ2n = SymmetricTensor{2,3,Float64}(SA[σᵢⱼ0[1,1], 0.1*σᵢⱼ0[1,1], 0, σᵢⱼ0[2,2], 0, σᵢⱼ0[3,3]]) # σᵢⱼ0 with low negative shear stress
    σᵢⱼ3 = DSB.build_principal_stress_tensor(r,1,σ₃,Dᵢ ; abstol=1e-15) # hydrostatic in plane stress
    σ_angle_0 = DSB.get_stress_deviation_from_y(σᵢⱼ0)
    σ_angle_1 = DSB.get_stress_deviation_from_y(σᵢⱼ1)
    σ_angle_2p = DSB.get_stress_deviation_from_y(σᵢⱼ2p)
    σ_angle_2n = DSB.get_stress_deviation_from_y(σᵢⱼ2n)
    σ_angle_3 = DSB.get_stress_deviation_from_y(σᵢⱼ3 ; shear_mode=:simple)
    @test σ_angle_0 == 90 # σ1 along x
    @test σ_angle_1 == 0 # σ1 along y
    @test 0 < σ_angle_2p < 90 # positive trigonometric angle from σyy
    @test -90 < σ_angle_2n < 0 # negative trigonometric angle from σyy => because the angle is expressed in [-π/2 ; π/2]
    @test σ_angle_3 == 45 # set to 45° for hydrostatic conditions in simple shear test because it goes to this value as soon as shear terms become non zero
end

@testset "get_KI_minimizer_D" begin
    r = DSB.Rheology(D₀=0.25)
    Dc = DSB.get_KI_minimizer_D(r,S,σ₃)
    σᵢⱼ_Dc = DSB.build_principal_stress_tensor(r,S,σ₃,Dc)
    KI_Dc = DSB.compute_KI(r, σᵢⱼ_Dc, Dc)

    # KI after damage onset
    KI_up = DSB.KI_from_external_load(r, S, σ₃, Dc+0.01)
    @test KI_up > KI_Dc

    # KI before damage onset
    KI_down = DSB.KI_from_external_load(r, S, σ₃, Dc-0.01)
    @test KI_down > KI_Dc
end

@testset "get_D_and_S_that_minimize_KI" begin
    σ₃ = -1e6
    r = DSB.Rheology(D₀=0.175)
    Dc, Sc = DSB.get_KI_minimizer_D_S(r,σ₃)
    σᵢⱼ_min = DSB.build_principal_stress_tensor(r,Sc,σ₃,Dc)
    KI_min = DSB.compute_KI(r,σᵢⱼ_min,Dc)
    @test abs(KI_min) < 1e-8

    eps = 1e-6
    S_more = Sc + eps
    S_less = Sc - eps
    D_more = Dc + eps
    D_less = Dc - eps
    KI_S_more = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,S_more,σ₃,Dc),Dc)
    KI_S_less = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,S_less,σ₃,Dc),Dc)
    KI_D_more = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,Sc,σ₃,D_more),D_more)
    KI_D_less = DSB.compute_KI(r,DSB.build_principal_stress_tensor(r,Sc,σ₃,D_less),D_less)

    @test abs(KI_S_more) > abs(KI_min)
    @test abs(KI_S_less) > abs(KI_min)
    @test abs(KI_D_more) > abs(KI_min)
    @test abs(KI_S_less) > abs(KI_min)
end

@testset "initialize stresses and strains from D_in and D_out" begin
    # parameters initialization
    σ₃ = -1e6
    S_i = 1.0
    Dⁱ = 0.5
    Dᵒ = 0.4
    θ = 60.0
    op = DSB.OutputParams()
    sp = DSB.SolverParams() 
    p = DSB.Params(op,sp)
    r = DSB.Rheology()

    # tested function
    σᵒᵢⱼ, σⁱᵢⱼ, ϵᵒᵢⱼ, ϵⁱᵢⱼ = DSB.initialize_state_var_D(r,p,S_i,σ₃,Dⁱ,Dᵒ,θ ; coords=:band)

    # tests on stress continuity
    @test σᵒᵢⱼ[1,2] ≈ σⁱᵢⱼ[1,2]
    @test σᵒᵢⱼ[2,2] ≈ σⁱᵢⱼ[2,2]

    # test on strain compatibility
    @test ϵᵒᵢⱼ[1,1] ≈ ϵⁱᵢⱼ[1,1]

    #test that epsilon oop is zero
    @test ϵᵒᵢⱼ[3,3] ≈ 0
    @test ϵⁱᵢⱼ[3,3] ≈ 0

    # test that functional form of epsilon applied to sigma gives plane strain
    ϵ_out = DSB.compute_ϵij(r,Dᵒ,σᵒᵢⱼ)
    ϵ_in = DSB.compute_ϵij(r,Dⁱ,σⁱᵢⱼ)
    abstol = 1e-15
    @test abs(ϵ_out[3,3]) < abstol*abs(ϵ_out[1,1])
    @test abs(ϵ_in[3,3]) < abstol*abs(ϵ_in[1,1])
    @test all(i -> i < abstol, abs.(ϵ_out .- ϵᵒᵢⱼ)) == true
    @test all(i -> i < abstol, abs.(ϵ_in .- ϵⁱᵢⱼ)) == true
end

@testset "comparison between two residual forms" begin
    # input :
    r = DSB.Rheology(
      G = 28e9,
      ν = 0.25,
      μ = 0.6, # Friction coef
      β = 0.1, # Correction factor
      K₁c = 1.74e6, # Critical stress intensity factor (Pa.m^(1/2))
      a = 0.1, # Initial flaw size (m)
      ψ = 0.5*atand(1/0.6),# crack angle to the principal stress (radians)
      D₀ = 0.55,# Initial flaw density
      n = 5.5, # Stress corrosion index
      l̇₀ = 7e-5, # Ref. crack growth rate (m/s)
      H = 50e3, # Activation enthalpy (J/mol)
      A = 5.71 # Preexponential factor (m/s)
    )

    σ₃ = -1e6
    du = [-17.060700132915812, -4.572215062877375, -12.380368686287465, 1.530946116594802e-8, 8.045129698897287e-7]
    u = [-1.8458631082604998e6, 878593.6772462169, -1.942972357187292e6, 0.00024964206656874537, 0.9899999999999999]
    σ₁₁, σ₁₂, σₒₒₚ, ϵ₂₂, D = u
    σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂, Ḋ = du
    σᵢⱼ = SymmetricTensor{2,3}(SA[σ₁₁ σ₁₂ 0 ; σ₁₂ σ₃ 0 ; 0 0 σₒₒₚ])
    Ḋ = du[5]
    du_nl = SA[σ̇₁₁, σ̇₁₂, σ̇ₒₒₚ, ϵ̇₂₂*r.G]

    σ̇ᵢⱼ = SymmetricTensor{2,3}([ σ̇₁₁    σ̇₁₂   0.0  ;
                                 σ̇₁₂    0.0   0.0  ;
                                 0.0    0.0   σ̇ₒₒₚ ])

    𝕀 = SymmetricTensor{2,3}(DSB.δ) # Second order identity tensor

    # stress at previous timestep
    σ = (1/3)*(tr(σᵢⱼ))
    sᵢⱼ = dev(σᵢⱼ)
    τ = DSB.get_τ(sᵢⱼ)

    # stress derivatives
    σ̇ = (1/3)*(tr(σ̇ᵢⱼ))
    τ̇ = sᵢⱼ ⊡ σ̇ᵢⱼ / (2*τ)

    # damage constants
    c1, c2, c3 = DSB.compute_c1c2c3(r,D)
    A, B = DSB.compute_AB(r,c1,c2,c3)
    A1, B1 = DSB.compute_A1B1(r,A,B)

    dc1dD = DSB.compute_dc1dD(r,D)
    dc2dD = DSB.compute_dc2dD(r,D)
    dc3dD = DSB.compute_dc3dD(r,D)
    dA1dD = DSB.compute_dA1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)
    dB1dD = DSB.compute_dB1dD(r,dc1dD,dc2dD,dc3dD,c2,c3)

    la1 = DSB.λ₁(A1,B1,σ,τ)
    la2 = DSB.λ₂(r,A1,B1,σ,τ)
    la3 = DSB.λ₃(A1,B1)
    dla1dD  = DSB.dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ)
    dla2dD  = DSB.dλ₂dD(A1,B1,dA1dD,dB1dD,σ,τ)
    dla3dD  = DSB.dλ₃dD(A1,B1,dA1dD,dB1dD)
    dla1dσ  = DSB.dλ₁dσ(A1,B1,τ)
    dla1dτ  = DSB.dλ₁dτ(A1,B1,σ,τ)

    # RES JAO
    res1 = SA[ la1*σ̇₁₁ - la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*σᵢⱼ[1,1] - dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[1,1],
              -2*r.G*ϵ̇₂₂ - la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*params.σ₃ - dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[2,2],
              la1*σ̇ₒₒₚ - la2*σ̇ + la3*τ̇ + Ḋ * (dla1dD*σᵢⱼ[3,3] - dla2dD*σ + dla3dD*τ) + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[3,3],
              -2*r.G*params.ϵ̇₁₂ + la1*σ̇₁₂ + Ḋ*dla1dD*σᵢⱼ[1,2] + (dla1dσ*σ̇ + dla1dτ*τ̇)*sᵢⱼ[1,2] ] 

    # RES initial
    t1 = DSB.λ₁(A1,B1,σ,τ)*σ̇ᵢⱼ - ( DSB.λ₂(r,A1,B1,σ,τ)*σ̇ - (1/3)*A1*B1*τ̇ )*𝕀
    t2 = ( DSB.dλ₁dσ(A1,B1,τ)*σ̇ + DSB.dλ₁dτ(A1,B1,σ,τ)*τ̇ )*sᵢⱼ
    t3 = DSB.dλ₁dD(A1,B1,dA1dD,dB1dD,σ,τ)*Ḋ*σᵢⱼ - ( DSB.dλ₂dD(A1,B1,dA1dD,dB1dD,σ,τ)*Ḋ*σ - (1/3)*Ḋ*(dA1dD*B1 + A1*dB1dD)*τ )*𝕀
    ϵ̇ᵢⱼ = t1 + t2 + t3

    res2 = SA[ϵ̇ᵢⱼ[1,1],
              -2*r.G*ϵ̇₂₂ + ϵ̇ᵢⱼ[2,2],
              ϵ̇ᵢⱼ[3,3],
              -2*r.G*params.ϵ̇₁₂ + ϵ̇ᵢⱼ[1,2]]

    @test maximum(abs.(res1 .- res2)) <=1e-12
end
# @testset "residual function DiffEq form" begin
#     # parameters initialization
#     σ₃ = -1e6
#     S_i = 1.0
#     Dⁱ = 0.5
#     Dᵒ = 0.4
#     θ = 60.0
#     ϵ̇ⁱξη = 0.0
#     op = DSB.OutputParams()
#     sp = DSB.SolverParams() 
#     mp = DSB.Params(op,sp)
#     r = DSB.Rheology()

#     # parameters
#     p = DSB.DiffEqParams(r = r,
#                     mp = mp,
#                     du = fill(ϵ̇ⁱξη,6),
#                     scalars = SA[σ₃, ϵ̇ⁱξη, θ],
#                     allow_Ḋᵒ = true)
#     σᵒᵢⱼ, σⁱᵢⱼ, ϵᵒᵢⱼ, ϵⁱᵢⱼ = DSB.initialize_state_var_D(r,mp,S_i,σ₃,Dⁱ,Dᵒ,θ ; coords=:band)
#     u0 = SA_F64[S_i, σⁱᵢⱼ[1,1], σⁱᵢⱼ[3,3], ϵⁱᵢⱼ[2,2], Dᵒ, Dⁱ]
#     du_nl = SA_F64[0.0, 0.0, 0.0, 0.0]

#     KIᵒ = DSB.compute_KI(r,σᵒᵢⱼ,Dᵒ)
#     KIⁱ = DSB.compute_KI(r,σⁱᵢⱼ,Dⁱ)
#     Ḋᵒ = DSB.compute_subcrit_damage_rate(r, KIᵒ, Dᵒ)
#     Ḋⁱ = DSB.compute_subcrit_damage_rate(r, KIⁱ, Dⁱ)

#     res  = DSB.residual_2_points_2(du_nl, σᵒᵢⱼ, σⁱᵢⱼ, Dⁱ, Ḋⁱ, Dᵒ, Ḋᵒ, p)
#     @test norm(res) ≈ 0
# end