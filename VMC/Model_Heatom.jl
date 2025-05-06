module Model_Heatom

using StaticArrays
using LinearAlgebra: norm

export EL, drift, ln_psi2, Ψ, N, D, Eexact,  λ
export V
export trial, α, α12, β
export Elocal_multi, ln_psi2_multi, drift_multi, psi_multi
export Ψ_i_per_Ψ_multi

const λ = 0.5   # hbar^2/(2m) in a.u.

const N = 2              # number of electrons
const Z = 2
const D = 3              # dimension  
const Eexact = -2.903724377  # accurate ground state energy

# =================
# choose trial type
# =================
# pick one:
#trial = "1S"    # Hydrogen 1S only
#trial = "2par"  # α and α12
trial = "3par"   # α, α12 and β
#trial = "3par_opt" # α, α12 and β optimization
# 


# trial wafe function parameters
if trial=="1S"
    # energy optimized, analytical:
    const α = 27/16   # 1.6875
    # Note: exact E = α^2 - 27/8*α 
    # ignored:
    const α12 = 0.0
    const β = 0.0
end
if trial=="2par"
    # cusp condition values:
    const α = 2.0 
    const α12 = 0.5
    # ignored:
    const β = 0.0
end
if trial=="3par"
    # cusp condition values:
    #const α = 2.0
    #const α12 = 0.5
    # energy optimized, correlated sampling:
    #const β = 0.1437 #-2.878080 +/- 0.000100 
    #const β = 0.143    # => -2.87820 ± 0.0001
    #const β = 0.144   # => -2.87829 ± 0.0001
    #const β = 0.145   # => -2.87814 ± 0.0001

    # energy optimized, correlated sampling:
    # all three free parameters
    # <E> = -2.891115 +/- 0.000020 
    const α = 1.847529
    const α12 = 0.359070
    const β = 0.159321
    

    #const α = 1.848602
    #const α12 = 0.367626
    #const β = 0.168144
    
    # energy optimized, correlated sampling:
    #const α = 1.8487529
    #const α12 = 0.359070
    #const β = 0.159321
    
    # variance optimized, correlated sampling:
    #const α = 1.952246
    #const α12 = 0.458058
    #const β = 0.308074
    

    # fixed α12, energy optimized
    #const α = 1.842157
    #const α12 = 0.50000
    #const β = 0.348455
    
    # fixed α and α12, variance optimized
    #const α = 2.0000
    #const α12 = 0.50000
    #const β = 0.326771
    # fixed α and α12, energy optimized
    #const α = 2.0000
    #const α12 = 0.50000
    #const β = 0.143799
    
end

if trial=="3par_opt"
    # cusp condition values:
    #const α = 2.0
    #const α12 = 0.5
    #const β = 0.327
    # energy optimized, correlated sampling:
    const α = 1.847825
    const α12 = 0.351858
    const β = 0.150352

    # variance optimized, correlated sampling:
    #const α = 1.949544
    #const α12 = 0.526500
    #const β = 0.430728

end



# utility
@inline function get_unitvecs(R::MMatrix{D,N,Float64})
    vecr1 = R[:,1]
    r1 = norm(vecr1)  
    vecr2 = R[:,2]
    r2 = norm(vecr2)
    hatr2  = vecr2/r2
    vecr12 =  R[:,1]-R[:,2]
    r12 = norm(vecr12)
    vecr1/r1,vecr2/r2,vecr12/r12
end

# potential energy
function V(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    - Z/r1 - Z/r2 + 1/r12 
end



# Fixed wf parameter codes
# ========================
function EL(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    hatr1, hatr2, hatr12 = get_unitvecs(R)
    b = 1+β*r12
    ∇S = 0.5*drift(R)
    ∇2S = -α*2/r1 -α*2/r2  + 4*α12/b^3*1/r12 
    EL = -0.5*(sum(∇S.^2) + ∇2S) - Z/r1 - Z/r2 + 1/r12
end

# 2∇S
@inline function drift(R ::MMatrix)
    hatr1, hatr2, hatr12 =  get_unitvecs(R)
    r12 = norm(R[:,1]-R[:,2])
    b = 1+β*r12    
    ∇S = hcat(-α*hatr1 + α12/b^2*hatr12, -α*hatr2 - α12/b^2*hatr12)
    2∇S
end

# ln(φ_T^2) = 2*ln(φ_T)
function ln_psi2(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1 = norm(R[:,1])
    r2 = norm(R[:,2])
    b = 1+β*r12
    2*(-α*(r1+r2) + α12*r12/b)
end

function Ψ(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1 = norm(R[:,1])
    r2 = norm(R[:,2])
    b = 1+β*r12
    exp(-α*(r1+r2) + α12*r12/b)
end

#=
# Variable wf parameter codes
# ===========================
# parameters  α1, α2, α12, β
function Elocal(R ::MMatrix, wf_params ::Vector)
    α1, α2, α12, β = wf_params
    if α1 != α2
        @show(α1)
        @show(α2)
        println("Error: Elocal(R,wf_params) should be used with α1 = α2")
        exit()
    end
    r12 = norm(R[:,1]-R[:,2])
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    hatr1, hatr2, hatr12 = get_unitvecs(R)
    b = 1+β*r12
    ∇S = 0.5*drift(R, wf_params)
    ∇2S = -α1*2/r1 -α2*2/r2  + 4*α12/b^3*1/r12 
    Elocal = -0.5*(sum(∇S.^2) + ∇2S)  - Z/r1 - Z/r2 + 1/r12
end


@inline function drift(R ::MMatrix, wf_params ::Vector)
    α1, α2, α12, β = wf_params
    if α1 != α2
        @show(α1)
        @show(α2)
        println("Error: drift(R,wf_params) should be used with α1 = α2")
        exit()
    end
    hatr1, hatr2, hatr12 =  get_unitvecs(R)
    r12 = norm(R[:,1]-R[:,2])
    b = 1+β*r12    
    ∇S = hcat(-α1*hatr1 + α12/b^2*hatr12, -α2*hatr2 - α12/b^2*hatr12)
    2*∇S #  symmetric only if α1=α2
end
=#

function ln_psi2(R ::MMatrix, wf_params ::Vector)
    α1, α2, α12, β = wf_params
    if α1 != α2
        @show(α1)
        @show(α2)
        println("Error: ln_psi2(R,wf_params) should be used with α1 = α2")
        exit()
    end
    r12 = norm(R[:,1]-R[:,2])
    r1 = norm(R[:,1])
    r2 = norm(R[:,2])
    b = 1+β*r12
    Ψlog_12 = 2*(-α1*r1 - α2*r2 + α12*r12/b)  # symmetric only if α1=α2
end

# Multiparameter functions
# ------------------------
# parameters c, α1, α2, α12, β
#
# ψ :=  sum_k ck^2*exp(-α1*r1 - α2*r2 - α12*r12)  + (1<->2)
#   :=  sum_k ck^2*exp(Sk) + (1<->2) 
#   Sk :=  -α1*r1 - α2*r2 - α12*r12 
#    
# ln(|Ψ|^2) = 2ln|\Psi| = 2*ln( sum_k ck*exp(Sk) + (1<->2) )
# ∇Ψ_i =  sum_k ck*exp(Sk) ∇_iSk +  (1<->2)  
# drift_i := 2(∇_iΨ)/Ψ = 2/Ψ* [ sum_k ck*exp(Sk) ∇Sk  + (1<->2) ]
#       
# ∇_i^2Ψ =  sum_k ck^2*exp(Sk) (∇_i^2Sk + (∇_iSk)^2)  + (1<->2) 
# TL = -1/2*Σ_i (∇_i^2Ψ)/Ψ = -1/(2Ψ)* [ sum_k ck^2*exp(Sk) Σ_i(∇_i^2Sk + (∇_iSk)^2)  + (1<->2) ] 
# EL = TL + V(R)
#
@inline function Elocal_multi(R ::MMatrix, wf_params ::Vector)    
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    r12 = norm(R[:,1]-R[:,2])
    hatr1, hatr2, hatr12 =  get_unitvecs(R)
    
    TL = 0.0
    Ψ = 0.0    
    for k in 1:4:length(wf_params)
        ck, α1, α2, α12 = wf_params[k:k+3]
        if abs(ck)<1e-15 continue end
        for sym = 1:2
            α1, α2 =  α2, α1                   
            Sk = -α1*r1 - α2*r2 - α12*r12
            ∇Sk = hcat(-α1*hatr1 - α12*hatr12, -α2*hatr2 + α12*hatr12)
            ∇2Sk = -α1*2/r1 -α2*2/r2  - 4*α12*1/r12
            Ψ  += ck^2*exp(Sk)
            TL += ck^2*exp(Sk)* (sum(∇Sk.^2) + ∇2Sk)
        end
    end
    TL *= -1/(2Ψ)
    EL = TL + V(R)
end



# drift_i := 2∇_iΨ/Ψ = 2/Ψ* [ sum_k ck*exp(Sk) ∇_iSk  + (1<->2) ]
# ∇_iΨ =  sum_k ck*exp(Sk) ∇_iSk +  (1<->2)
# Careful not to mix ∇_1 Ψ and ∇_2 Ψ, the gradients are *not* supposed to be symmetrized
@inline function drift_multi(R ::MMatrix, wf_params ::Vector)
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    r12 = norm(R[:,1]-R[:,2])
    hatr1, hatr2, hatr12 =  get_unitvecs(R)
    
    ∇Ψ = zeros(MMatrix{3,2})
    Ψ = 0.0
    for sym = 1:2
        r1, r2 = r2, r1
        for k in 1:4:length(wf_params)
            ck, α1, α2, α12 = wf_params[k:k+3]
            Sk = -α1*r1 - α2*r2 - α12*r12
            Ψ += ck^2*exp(Sk)
            if sym==2
                ∇Ψ += ck*exp(Sk)* hcat(-α1*hatr1 - α12*hatr12, -α2*hatr2 + α12*hatr12)
            else
                ∇Ψ += ck*exp(Sk)* hcat(-α2*hatr1 - α12*hatr12, -α1*hatr2 + α12*hatr12)
            end
        end
    end

    
    drift = 2∇Ψ/Ψ 
    drift
end



# ψ = [  sum_k ck*exp(-α1*r1 - α2*r2 - α12*r12)  + (1<->2) ]
#   := [ sum_k ck*exp(Sk) + (1<->2) ]
#   Sk :=  -α1*r1 - α2*r2 - α12*r12 
@inline function psi_multi(R ::MMatrix, wf_params ::Vector)
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    r12 = norm(R[:,1]-R[:,2])
    Ψ = 0.0
    for sym = 1:2
        r1, r2 = r2, r1
        for k in 1:4:length(wf_params)
            ck, α1, α2, α12 = wf_params[k:k+3]
            Sk = -α1*r1 - α2*r2 - α12*r12 
            Ψ += ck^2*exp(Sk)
        end
    end
    Ψ
end

# ψ = [  sum_k ck*exp(-α1*r1 - α2*r2 -α12*r12)  + (1<->2) ]
#   := [ sum_k ck*exp(Sk) + (1<->2) ]
#   Sk :=  -α1*r1 - α2*r2 - α12*r12 
@inline function ln_psi2_multi(R ::MMatrix, wf_params ::Vector)
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    r12 = norm(R[:,1]-R[:,2])
    npara = length(wf_params)
    Ψ = 0.0
    for sym = 1:2
        r1, r2 = r2, r1
        for k in 1:4:npara
            ck, α1, α2, α12 = wf_params[k:k+3]
            if abs(ck)<1e-15 continue end
            Sk = -α1*r1 - α2*r2 - α12*r12
            Ψ += ck^2*exp(Sk)
        end
    end   
    2*log(Ψ) # will give error if Ψ<0, but boson ground state wf is non-negative 
end

# Ψ_i/Ψ 
@inline function Ψ_i_per_Ψ_multi(R ::MMatrix, wf_params ::Vector)
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    r12 = norm(R[:,1]-R[:,2])
    npara = length(wf_params)
    Ψ = 0.0    
    Ψi = zeros(npara)
    for sym = 1:2
        r1, r2 = r2, r1
        for k in 1:4:npara
            ck, α1, α2, α12 = wf_params[k:k+3]
            Sk = -α1*r1 - α2*r2 - α12*r12
            ee = exp(Sk)
            Ψ += ck^2*ee
            Ψi[k] += 2ck*ee
            Ψi[k+1] += -r1 * ck^2*ee
            Ψi[k+2] += -r2 * ck^2*ee
            Ψi[k+3] += -r12 * ck^2*ee
        end
    end
    Ψi/Ψ

end

end




