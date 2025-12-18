__precompile__(false)
module PIMC_Systems

using ForwardDiff
using DelimitedFiles

# 
# Submodules:
# ==============================================================================
module HarmonicOscillator
@inline norm(x) = sqrt(sum(abs2,x))

const name = "HO"
const Ntherm = 10000


function potential(r::AbstractArray{T}) where T<:Real
    """Harmonic oscillator confinement"""
    return 0.5*norm(r)^2 # 1/2*mω^2 r^2, units m=1 ω=1
end



function grad_potential(r::AbstractArray{T}) where T<:Real
    """∇ V(vecr), gradient of Harmonic oscillator confinement"""
    return r #  mω^2 vecr, units m=1 ω=1
end

function grad2_potential(r::AbstractArray{T}) where T<:Real
    """∇^V(vecr), Laplacian of Harmonic oscillator confinement"""
    # ∇⋅r = dim 
    return size(r,1) 
end

function E_exact(β::Float64, dim::Int64, N::Int64, λ::Float64, L::Float64)
    """<E> for distinquishable noninteracting boltzmannons per particle per dimension"""
    # E/N per dimension is ħω/2 * 1/tanh(β*ħω/2) in units ħω=1
    return 1/(2*tanh(β/2)) * dim
end


function E_exact_bosons(β::Float64, dim::Int64, N::Int64)
    E_exact_bose = 0.0 # default: unknown
    if dim == 1
        # One way to calculate 
        #q = exp(-β)   # exp(-β ħω)
        #s = 0
        #for j = 1:N
        #    s =  s + j* q^j/(1-q^j)
        #end
        #E_VA =  (N/2+s) / N
        
        #println("E_VA/N (Bose) = $E_VA")
        
        # Takahashi-Imada :
        # harm. osc. peculiarity: E=2*Ekin=2*Epot
        E_TI = 0
        for j = 1:N
            E_TI =  E_TI + j*0.5/tanh(j*β/2)   
        end 
        E_TI = 1/N*(E_TI - N*(N-1)/4) #  E/N

        #println("E_TI/N (Bose) = $E_TI")
        # use TI result as exact:
        E_exact_bose = E_TI
    end
    # Yet another way
    #if N==2
    # from Z2 = 1/2*(Z1(β)^2 + Z1(2β))
    #E2 = 1 + exp(-β)/(1-exp(-β)) + 2*exp(-2β)/(1-exp(-2β))
    #@show E2/2
    #end

    return E_exact_bose 
end

end

# -----------------------------------------------------------------

module Noninteracting

using ..ForwardDiff, ..DelimitedFiles

const name = "Nonint"


mutable struct t_exact
    E::Float64
    E_bose::Float64
end
const exact = t_exact(0.0, 0.0)




function Z1s(β, dim::Int64, N::Int64, λ::Float64, L::Float64)
    """Single-particle Z_1 for β, 2β, ..., Nβ in array Zs """
      
    if N>5
        println("Z1s won't work for N>5")
        return
    end
    ## TEST L = 10.0 * N^(1/dim)   # L chosen in PIMC_worm
    
    Zs  = zeros(eltype(β), N) # may be dual numbers for ForwardDiff

    # Cutoff energy
    # exp(-β*e1)<lim <=> e1 < -log(lim)/β
    lim = 1e-10
    ecut = -log(lim)/β
    # ecut = λ kcut^2 <=> kcut^2 = ecut/λ
    k2cut = ecut/λ
    ns = ceil(k2cut*(L/2π)^2) |> Int64
    
    for nx in -ns: ns
        kx2 = (nx*2π/L)^2
        kx2>k2cut && continue
        if dim==1
            e1 = λ*kx2 
            Zs .+= exp.([-n*β*e1 for n in 1:N])
            continue
        end
        for ny in -ns: ns
            ky2 = (ny*2π/L)^2
            ky2>k2cut && continue
            if dim==2
                e1 = λ*kx2 
                Zs .+= exp.([-n*β*e1 for n in 1:N])
                continue
            end
            for nz in -ns: ns
                kz2 = (nz*2π/L)^2
                kz2>k2cut && continue
                e1 = λ*(kx2 + ky2 + kz2)
                Zs .+= exp.([-n*β*e1 for n in 1:N])
            end
        end
    end
    
    return Zs    
end


function E_check(β::Float64, dim::Int64, N::Int64, λ::Float64, L::Float64)
    
    L = 10.0 * N^(1/dim)   # L chosen in PIMC_worm
    
    E = 0.0
    Z1x = 0.0
    # Cutoff energy
    # exp(-β*e1)<lim <=> e1 < -log(lim)/β
    lim = 1e-15
    ecut = -log(lim)/β
    # ecut = λ kcut^2 <=> kcut^2 = ecut/λ
    k2cut = ecut/λ
    ns = ceil(k2cut*(L/2π)^2) |> Int64
    for nx in -ns: ns
        kx2 = (nx*2π/L)^2
        e1x = λ*kx2 
        Z1x += exp(-β*e1x)
        E  += -e1x*exp(-β*e1x)
    end
    E = -2*dim/Z1x*E
end



function ZN(β, dim::Int64, N::Int64) 
    Z =  Z1s(β, dim, N, λ, L)
    Z1 = Z[1]^N
end


function E_exact(β, dim::Int64, N::Int64, λ::Float64, L::Float64)
    if dim==3 && N>5    
        return 3/(2*β) # E=3/2*N*kB*T 
    end
    if !isapprox(exact.E, 0.0)        
        return exact.E
    end
    Zβ(x) = ZN(x, dim, N, λ, L)
    E = -ForwardDiff.derivative(Zβ, β) / Zβ(β)
    println("Dist: $(E/N)")
    EE = E_check(β, dim, N, λ, L)
    println("Dist: $(EE/N) computed for checking")
    exact.E = E/N
    E/N
end

function ZN_bose(β, dim::Int64, N::Int64) 
    Z =  Z1s(β, dim, N)
    Z1 = Z[1]
    N==1 && (return Z1)
    Z2 = +Z[2]/2 + Z[1]^2/2
    N==2 && (return Z2)
    Z3 = +Z[2]*Z[1]/2 + Z[3]/3 + Z[1]^3/6
    N==3 && (return Z3)
    Z4 = Z[2]^2/8 + Z[2]*Z[1]^2/4 + Z[3]*Z[1]/3 + Z[4]/4 + Z[1]^4/24
    N==4 && (return Z4)
    Z5 = Z[2]^2*Z[1]/8 + Z[2]*Z[3]/6 + Z[2]*Z[1]^3/12 + Z[3]*Z[1]^2/6 + Z[4]*Z[1]/4 + Z[5]/5 + Z[1]^5/120
    N==5 && (return Z5)
    error("Explicit recursions only for N<=5")
end

function E_exact_bosons(β, dim::Int64, N::Int64)
    if !isapprox(exact.E_bose, 0.0)
        return exact.E_bose # pre-computed value
    end
    if dim==3 && N>5    
        println("Linear interpolation of Grand Canonical E(T) from file Enoninteracting_GC.dat")
        dat = readdlm("Nonint/Enoninteracting_GC.dat", comments=true, comment_char='#')
        Ts = dat[:, 1]  
        Es = dat[:, 2]
        # linear interpolation to find E(T)
        T = 1/β
        ind = findfirst(x -> x >= T, Ts)
        E = 0.0
        if ind!==nothing  
            E = ((Ts[ind]-T)*Es[ind-1] + (T-Ts[ind-1])*Es[ind])/(Ts[ind]-Ts[ind-1])
        end
        exact.E_bose = E
    else
        Zβ(x) = ZN_bose(x, dim, N)
        E = -ForwardDiff.derivative(Zβ, β) / Zβ(β)
        println("Bose: $(E/N)")
        exact.E_bose = E
    end
    exact.E_bose
end

end # module
# -----------------------------------------------------------------

module HeLiquid

const name = "He_liquid"

#  Aziz, Phys. Rev. Lett. 74, 1586 (1995)
const epsil = 10.9560000
const rm    = 2.96830000
const aa    = 1.86924404e5
const D     = 1.43800000
const alpha = 10.5717543
const beta  = -2.07758779  
const c6    = 1.35186623
const c8    = 0.41495143
const c10   = 0.17151143



function E_exact(β::Float64, dim::Int64, N::Int64, λ::Float64, L::Float64)
    return 0.0
end
function E_exact_bosons(β::Float64, dim::Int64, N::Int64)
    return 0.0
end


 
@inline function potential(r::Float64) 
    """Aziz He-He potential; Unit K"""
    x::Float64 = r/rm
    fx::Float64 = x < D ? exp(-(D/x-1)^2) : 1.0    
    V::Float64 = epsil*(aa*exp(-alpha*x+beta*x^2)-fx*(c6/x^6 + c8/x^8 + c10/x^10))    
end

# AD version
@inline function potential(r::T) where T<:Real # liberal argument type for AD 
    """Aziz He-He potential; Unit K"""
    x = r/rm
    fx = x < D ? exp(-(D/x-1)^2) : one(T)    # causes type instability, hence separate Float64 version
    V = epsil*(aa*exp(-alpha*x+beta*x^2)-fx*(c6/x^6 + c8/x^8 + c10/x^10))    
end



@inline function der_potential(r::Float64) 
    """Derivative V'(r) of Aziz He-He potential (discontinuous); Unit K/Ånsgröm"""        
    x::Float64   = r/rm
    x2::Float64  = x^2
    x6::Float64  = x2^3
    x8::Float64  = x2^4
    x10::Float64 = x2^5
    F::Float64   = 1.0
    dV::Float64  = 0.0    
    if x < D
        xD = D/x - 1.0
        F = exp(-(xD^2))
        dV = -2 * xD * D/x2* F *(c6/x6 + c8/x8 + c10/x10)
    end
    dV += aa*(-alpha + 2*beta * x) * exp(-alpha*x+beta*x2) + F*(6*c6/x6+8*c8/x8+10*c10/x10)/x
    dV *= epsil/rm
    return dV
end

@inline function der_potential(r::T) where T<:Real    
    """Derivative V'(r) of Aziz He-He potential (discontinuous); Unit K/Ånsgröm"""
    x   = r/rm
    x2  = x^2
    x6  = x2^3
    x8  = x2^4
    x10 = x2^5
    F   = one(T)
    dV = zero(T)
    if x < D
        xD = D/x - one(T)
        F = exp(-(xD^2))
        dV = -2 * xD * D/x2* F *(c6/x6 + c8/x8 + c10/x10)
    end
    dV += aa*(-alpha + 2*beta * x) * exp(-alpha*x+beta*x2) + F*(6*c6/x6+8*c8/x8+10*c10/x10)/x
    dV *= epsil/rm
    return dV
end



# tested with AD
@inline function der2_potential(r::Float64) 
    """Derivative V''(r) of Aziz He-He potential (discontinuous); Unit K/Ånsgröm^2"""       
    x::Float64   = r/rm
    F::Float64   = 1.0
    dF::Float64  = 0.0
    ddF::Float64 = 0.0
    xD::Float64  = D/x - 1.0
    if x < D  
        F = exp(-xD^2)
        dF = 2*D*xD*F/x^2
        ddF = 4*D^2*xD^2*F/x^4 - 2*D^2*F/x^4 - 4*D*xD*F/x^3
    end 
    expo::Float64 = exp(-alpha*x + beta*x^2)
    ddV::Float64 = epsil/rm^2*(2*aa*beta*expo + aa*expo*(-alpha + 2*beta*x)^2 
                      + (-110*c10/x^12 - 42*c6/x^8 - 72*c8/x^10)*F 
                      + 2*(10*c10/x^11 + 6*c6/x^7 + 8*c8/x^9)*dF 
                      + (-c10/x^10 - c6/x^6 - c8/x^8)*ddF)
    return ddV
end

@inline function der2_potential(r::T) where T<:Real
    """Derivative V''(r) of Aziz He-He potential (discontinuous); Unit K/Ånsgröm^2"""       
    x   = r/rm
    F   = one(T)
    dF  = zero(T)
    ddF = zero(T)
    xD = D/x - one(T)
    if x < D  
        F = exp(-xD^2)
        dF = 2*D*xD*F/x^2
        ddF = 4*D^2*xD^2*F/x^4 - 2*D^2*F/x^4 - 4*D*xD*F/x^3
    end 
    expo = exp(-alpha*x + beta*x^2)
    ddV = epsil/rm^2*(2*aa*beta*expo + aa*expo*(-alpha + 2*beta*x)^2 
                      + (-110*c10/x^12 - 42*c6/x^8 - 72*c8/x^10)*F 
                      + 2*(10*c10/x^11 + 6*c6/x^7 + 8*c8/x^9)*dF 
                      + (-c10/x^10 - c6/x^6 - c8/x^8)*ddF)
    return ddV
end


#
# Cutoff functions (not used in code, may ruin potential energy)
#
const rcut = Ref{Float64}(NaN) # placeholder
function init_V_cutoff(rcut_in::Float64)
    rcut[]=rcut_in 
end

@inline function Vcutoff(r::T) where T<:Real
    """Possible pair-potential cutoff - probably too shallow"""
    rc = rcut[]
    if r < rc
        x = r / rc
        return 1 - 10x^3 + 15x^4 - 6x^5
    else
        return 0.0
    end
end


function tail_correction(ρ::Float64, rc::Float64, L::Float64, dim::Int64, pbc::Bool) ::Float64
    """Potential energy tail correction, approximating g(r)=1 for r>rc"""
    
    if dim==1
        error("tail_correction not implemented for dim=1")
    end
    if pbc && rc>L/2
        error("rc > L/2 is no good for pbc") 
    end    
    
    n = 10001 # keep odd
    ws = [1; repeat([4,2], div(n-3,2)); [4, 1]] # Simpson rule weights 14242...4241
    rstep = 50.0/n # far enough so that the potential is approx zero
    Vtail = 0.0
    # Simpson rule integration 
    for i = 1:n
        r = rc + (i-1)*rstep 
        Vtail += ws[i]*r^(dim-1)*potential(r)
    end        
    Vtail *= (dim-1)*π*ρ *rstep/3 
    return Vtail
end

end
end

