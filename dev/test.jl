using DamagedShearBand ; const DSB = DamagedShearBand

σ₃ = -1e6
r = DSB.Rheology()
Sc = get_damage_onset(r,σ₃)
σᵢⱼ_onset = DSB.build_elastic_stress_tensor(r,Sc,σ₃)
KI_onset = DSB.compute_KI(r, σᵢⱼ_onset, r.D₀)

σᵢⱼ_up = DSB.build_elastic_stress_tensor(r,Sc+0.1,σ₃)
KI_up = DSB.compute_KI(r, σᵢⱼ_up, r.D₀)
σᵢⱼ_down = DSB.build_elastic_stress_tensor(r,Sc-0.1,σ₃)
KI_down = DSB.compute_KI(r, σᵢⱼ_down, r.D₀)