__precompile__(false)
module PIMC_Common

using StaticArrays
using LinearAlgebra


include("PIMC_Systems.jl")
using .PIMC_Systems: HarmonicOscillator, HeLiquid, Noninteracting

# under construction:
using PIMC_Neighborlist: t_neighborlist


export λ, pbc, bose, Ntherm, system, τ_target, SystemPotentials, syspots
export has_potential
export action, results
export N, dim, worm_C, worm_limit, worm_K, N_slice_max, M_max
export restart, restart_with_data
export periodic!, distance!, dist, dist2
export pull!, wrapβ
export AbstractAction, PrimitiveAction, ChinAction
export worm_stats


abstract type AbstractAction end
struct PrimitiveAction <: AbstractAction end
struct ChinAction <: AbstractAction end

# All potential in one struct
struct SystemPotentials{F1, F2, F3, F4, F5, F6}
    pair_potential::F1    
    der_pair_potential::F2 
    der2_pair_potential::F3
    confinement_potential::F4
    grad_confinement_potential::F5  
    grad2_confinement_potential::F6
end

struct NoPotential end

(::NoPotential)(x...) = 0.0
(::NoPotential)(x::AbstractVector) = zeros(eltype(x), length(x))
(::NoPotential)(x::Real) = 0.0

has_potential(::NoPotential) = false
has_potential(::Any) = true

# don't change this
TEST = false # test vs. production code, PIMC_Common.TEST may change anywhere


#  EXPERIMENTAL 
const opt_chin = false #  Chin Action a1 and t0 parameter optimization
const use_loop_centroid = false # virial reference point is in the middle of a loop; may reduce variance

# ==========================
# CHOOSE SYSTEM AND ACTION (uncomment one case)
#
# const case = 1      # Harmonic oscillator in 1D, N=2
 const case = 216    # Liquid He4, N=16
# const case = 264    # Liquid He4, N=64
# const case = 2128   # Liquid He4, N=128
# const case = 2256   # Liquid He4, N=256 (pretty slow)
# const case = 3      # Noninteracting Bose fluid in 3D, N=256


# =================
# case definitions
# =================

if case==1
    sys = :HarmonicOscillator
    system = PIMC_Systems.HarmonicOscillator
    const syspots = SystemPotentials(
        NoPotential(), # no pair potential        
        NoPotential(),  
        NoPotential(),
        PIMC_Systems.HarmonicOscillator.potential, 
        PIMC_Systems.HarmonicOscillator.grad_potential,
        PIMC_Systems.HarmonicOscillator.grad2_potential,
    )
    restart = false
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 2
    const bose = true
    const dim = 1
    const pbc = false
    const λ = 0.5
    const τ = 0.005
    const action = ChinAction # PrimitiveAction 
    worm_limit = 1.e6
    Ntherm = 10000
    worm_C = 0.5
    worm_K = 5 
    optimize_worm_params = true
end

if case==216 
    const sys = :HeLiquid
    const system = PIMC_Systems.HeLiquid
    const syspots = SystemPotentials(
        PIMC_Systems.HeLiquid.potential,        
        PIMC_Systems.HeLiquid.der_potential,
        PIMC_Systems.HeLiquid.der2_potential,
        NoPotential(), # no confinement
        NoPotential(),  
        NoPotential(),
    )
    restart = true
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 16
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ = 0.01
    const action = ChinAction 
    worm_limit = 1e6
    Ntherm = 200000
    worm_C = 0.1
    worm_K = 20 
    optimize_worm_params = true
end

if case==264
    sys = :HeLiquid
    system = PIMC_Systems.HeLiquid
    const syspots = SystemPotentials(
        PIMC_Systems.HeLiquid.potential,       
        PIMC_Systems.HeLiquid.der_potential,
        PIMC_Systems.HeLiquid.der2_potential,
        NoPotential(), # no confinement
        NoPotential(),  
        NoPotential(),
    )
    restart = true
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 64
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ = 0.01
    const action = ChinAction  #  PrimitiveAction 
    worm_limit = 1e6
    Ntherm = 200000
    worm_C = 0.1
    worm_K = 20 
    optimize_worm_params = true
end

if case==264_0 # for timing tests
    sys = :HeLiquid
    system = PIMC_Systems.HeLiquid
    const syspots = SystemPotentials(
        PIMC_Systems.HeLiquid.potential,       
        PIMC_Systems.HeLiquid.der_potential,
        PIMC_Systems.HeLiquid.der2_potential,
        NoPotential(), # no confinement
        NoPotential(),  
        NoPotential(),
    )
    restart = true
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 64
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ = 0.01
    const action = ChinAction    
    worm_limit = 1e6
    Ntherm = 100
    worm_C = 0.1
    worm_K = 20 
    optimize_worm_params = false # true
end



if case==2128
    const sys = :HeLiquid
    const system = PIMC_Systems.HeLiquid
    const syspots = SystemPotentials(
        PIMC_Systems.HeLiquid.potential,       
        PIMC_Systems.HeLiquid.der_potential,
        PIMC_Systems.HeLiquid.der2_potential,
        NoPotential(), # no confinement
        NoPotential(),  
        NoPotential(),
    )
    restart = true
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 128
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ = 0.01
    const action = ChinAction 
    worm_limit = 1e6
    Ntherm = 200000
    worm_C = 0.1
    worm_K = 20 
    optimize_worm_params = true
end

if case==2256
    const sys = :HeLiquid
    const system = PIMC_Systems.HeLiquid
    const syspots = SystemPotentials(
        PIMC_Systems.HeLiquid.potential,       
        PIMC_Systems.HeLiquid.der_potential,
        PIMC_Systems.HeLiquid.der2_potential,
        NoPotential(), # no confinement
        NoPotential(),  
        NoPotential()
    )
    restart = true
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 256
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ = 0.01
    const action = ChinAction #PrimitiveAction 
    worm_limit = 1e6 #20.0
    Ntherm = 10000
    worm_C = 0.1
    worm_K = 5 
    optimize_worm_params = true
end

if case==3
    const sys = :Noninteracting
    const system = PIMC_Systems.Noninteracting
    const syspots = SystemPotentials(
        NoPotential(),
        NoPotential(),
        NoPotential(),
        NoPotential(),
        NoPotential(),  
        NoPotential()
    )
    restart = false
    if restart
        restart_with_data = false # continue collecting data or not
    end
    const N = 256
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 0.5
    const τ = 0.1
    const action = ChinAction #PrimitiveAction 
    const worm_limit = 1e6 
    Ntherm = 100000
    worm_C = 0.1
    worm_K = 2 
    optimize_worm_params = true
end

# ==============================================================



const N_slice_max = N+5 # extra space per slice
const M_max = 1000 # should be enough

mutable struct t_worm_stats
    N_open_try::Int 
    N_open_acc::Int 
    N_close_try::Int
    N_close_acc::Int
end
const worm_stats = t_worm_stats(0, 0, 0, 0)

# =====================================




results = Dict{String, Any}()
Vtail::Float64 = 0.0 # set in PIMC_main

# wrap times to [0, β), doesn't work if t>2β or t<-2β
@inline wrapβ(t, β) = t + (t < 0)*β - (t >= β)*β

@inline function periodic!(X::Matrix{Float64}, b::Int, L::Float64) 
    """Periodic shift to box [-L/2, L/2]"""
    @inbounds begin
        for d in 1:dim
            x = X[d, b]
            x -= L * floor((x + L/2) / L) # faster than x = mod(x + L/2, L) - L/2
            X[d, b] = x
        end
    end
end

@inline function periodic!(r::AbstractVector{Float64}, L::Float64) 
    """Periodic shift to box [-L/2, L/2]"""
    @inbounds for d in 1:dim
        x = r[d]
        x -= L * floor((x + L/2) / L)
        r[d] = x
    end
end


@inline function periodic!(r::AbstractVector{T}, L::Float64) where T<:Real
    """Periodic shift to box [-L/2, L/2]"""
    @inbounds for d in 1:dim
        r[d] -= L * floor((r[d] + L/2) / L)
    end
end


# bead-bead distance calculators
# ==============================

@inline function pbc_distance!(X::AbstractMatrix{Float64}, i::Int, j::Int, L::Float64, dr::MVector{dim, Float64})
    """Minimum image distance vector r1-r2 for pbc"""    
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j] 
        rij -= L * round(rij/L)
        dr[d] = rij
    end
end

@inline function non_pbc_distance!(X::AbstractMatrix{Float64}, i::Int, j::Int, L::Float64, dr::MVector{dim, Float64})
    """r1-r2"""    
    @inbounds for d in 1:dim
        dr[d] = X[d, i] - X[d, j] 
    end
end

@inline function pbc_distance!(r1::SVector{dim, Float64}, r2::SVector{dim, Float64}, L::Float64, dr::MVector{dim, Float64}) 
    """Minimum image distance vector r1-r2 for pbc"""
    @inbounds for d in 1:dim
        r12 = r1[d] - r2[d]
        r12 -= L * round(r12/L)
        dr[d] = r12
    end
end

@inline function non_pbc_distance!(r1::SVector{dim, Float64}, r2::SVector{dim, Float64}, L::Float64, dr::MVector{dim, Float64})
    """r1-r2"""    
    @inbounds for d in 1:dim
        dr[d] = r1[d] - r2[d]
    end
end

@inline function pbc_distance!(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64, dr::MVector{dim, Float64}) 
    """Minimum image distance vector r1-r2 for pbc"""    
    @inbounds for d in 1:dim
        r12 = r1[d] - r2[d]
        r12 -= L * round(r12/L)
        dr[d] = r12
    end
end

@inline function non_pbc_distance!(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64, dr::MVector{dim, Float64})
    """r1-r2"""    
    @inbounds for d in 1:dim
        dr[d] = r1[d] - r2[d]
    end
end


@inline function pbc_distance!(r1::AbstractVector{T}, r2::AbstractVector{T}, L::Float64, dr::AbstractVector{T}) where T<:Real
    """Minimum image distance vector r1-r2 for pbc"""    
    @inbounds for d in 1:dim
        r12 = r1[d] - r2[d]
        r12 -= L * round(r12/L)
        dr[d] = r12
    end
end

@inline function non_pbc_distance!(r1::AbstractVector{T}, r2::AbstractVector{T}, L::Float64, dr::AbstractVector{T}) where T<:Real
    """r1-r2"""    
    @inbounds for d in 1:dim
        dr[d] = r1[d] - r2[d]
    end
end

@inline function pbc_dist(X::AbstractMatrix{Float64}, i::Int, j::Int, L::Float64) ::Float64 
    """Minimum image distance |r_i-r_j| for pbc"""
    r::Float64 = 0.0
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j] 
        rij -= L * round(rij/L)
        r += rij*rij
    end
    r = sqrt(r)
end


@inline function non_pbc_dist(X::AbstractMatrix{Float64}, i::Int, j::Int, L::Float64) ::Float64 
    """Direct distance |r_i-r_j|; L is dummy"""
    r::float64 = 0.0
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j]
        r +=  rij*rij
    end  
    r = sqrt(r)
end



@inline function pbc_dist(X::Matrix{Float64}, i::Int, j::Int, L::Float64) ::Float64 
    """Minimum image distance |r_i-r_j| for pbc"""
    r::Float64 = 0.0
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j] 
        rij -= L * round(rij/L)
        r += rij*rij
    end
    r = sqrt(r)
end


@inline function non_pbc_dist(X::Matrix{Float64}, i::Int, j::Int, L::Float64) ::Float64 
    """Direct distance |r_i-r_j|; L is dummy"""
    r::float64 = 0.0
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j]
        r +=  rij*rij
    end  
    r = sqrt(r)
end


@inline function pbc_dist(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64) ::Float64 
    """Minimum image distance |r1-r2| for pbc"""
    r = 0.0
    @inbounds for d in 1:dim
        r12 = r1[d] - r2[d]    
        r12 -= L * round(r12/L)
        r += r12*r12
    end
    r = sqrt(r)
end


@inline function non_pbc_dist(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64) ::Float64 
    """Direct distance |r1-r2|; L is dummy"""
    r = 0.0
    @inbounds for d in 1:dim
        r +=  (r1[d] - r2[d])^2
    end  
    r = sqrt(r)
end


@inline function pbc_dist2(X::Matrix{Float64}, i::Int, j::Int, L::Float64) ::Float64
    """Minimum image distance squared |r_i-r_j|^2 for pbc"""
    r = 0.0
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j]
        rij -= L * round(rij/L)
        r += rij*rij
    end
    r 
end


@inline function non_pbc_dist2(X::Matrix{Float64}, i::Int, j::Int, L::Float64) ::Float64
    """Direct distance squared |r_i-r_i|^2; L is dummy"""
    r = 0.0
    @inbounds for d in 1:dim
        rij = X[d, i] - X[d, j]
        r += rij*rij
    end        
    r 
end

@inline function pbc_dist2(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64) ::Float64
    """Minimum image distance squared |r1-r2|^2 for pbc"""
    r = 0.0
    @inbounds for d in 1:dim
        r12 = r1[d] - r2[d]
        r12 -= L * round(r12/L)
        r += r12^2
    end
    r 
end


@inline function non_pbc_dist2(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64) ::Float64
    """Direct distance squared |r1-r2|^2; L is dummy"""
    r = 0.0
    @inbounds for d in 1:dim
        r += (r1[d] - r2[d])^2
    end        
    r 
end

# pick up minimum image or direct distance routines
const distance! = pbc ? pbc_distance! : non_pbc_distance!
const dist = pbc ? pbc_dist : non_pbc_dist
const dist2 = pbc ? pbc_dist2 : non_pbc_dist2


end

