__precompile__(false)
module PIMC_Common

using StaticArrays


include("PIMC_Systems.jl")
import .PIMC_Systems.HarmonicOscillator
import .PIMC_Systems.HeLiquid
import .PIMC_Systems.Noninteracting

export λ, pbc, bose, Ntherm, system, τ_target
export action, results
export N, dim, worm_C, worm_limit, worm_K, N_slice_max
export restart, restart_with_data
export periodic!, distance!, dist, dist2
export pull!
export AbstractAction, PrimitiveAction, ChinAction
export worm_stats


abstract type AbstractAction end
struct PrimitiveAction <: AbstractAction end
struct ChinAction <: AbstractAction end

TEST = false # test vs. production code, TEST may change anywhere
opt_chin = false # EXPERIMENTAL force Chin a1 ad t0 parameter optimization

# ==========================
# CHOOSE SYSTEM AND ACTION

case = 264

if case==1
    sys = :HarmonicOscillator
    restart = true
    if restart
        restart_with_data = true # continue collecting data or not
    end
    const N = 2
    const bose = true
    const dim = 1
    const pbc = false
    const λ = 0.5
    const τ = 0.01
    const action = ChinAction # PrimitiveAction 
    worm_limit = 1.e6
    const Ntherm = 10000
    worm_C = 0.5
    optimize_worm_params = true    
end

if case==216 
    sys = :HeLiquid
    restart = true
    if restart
        restart_with_data = false # continue collecting data or not
    end
    const N = 16
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ = 0.01
    const action = ChinAction
    worm_limit = 100.0
    const Ntherm = 100
    worm_C = 0.1
    optimize_worm_params = false   
end


if case==264
    sys = :HeLiquid
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
    const action = ChinAction #PrimitiveAction 
    worm_limit = 30.0
    const Ntherm = 10000
    worm_C = 0.1
    optimize_worm_params = true  
end


worm_K = 30 #  a better value given in PIMC_main based on M, optimized if optimize_worm_params = true
const N_slice_max = N+5 # extra space per slice

mutable struct t_worm_stats
    N_open_try::Int64 
    N_open_acc::Int64 
    N_close_try::Int64
    N_close_acc::Int64
end
const worm_stats = t_worm_stats(0, 0, 0, 0)

# =====================================

const system = eval(:(PIMC_Systems.$sys))

results = Dict{String, Any}()
Vtail = 0.0 # set in PIMC_main

# Lightweight buffer, *not* thread-safe
const vec_buffer  = MVector{dim, Float64}(undef)



#@inline pull!(v::Vector{Int64}, i::Int64) = filter!(j -> j ≠ i, v)
# faster, unordered version
@inline function pull!(v::Vector{Int64}, i::Int64)
    idx = findfirst(==(i), v)
    if idx !== nothing
        v[idx] = v[end]
        pop!(v)
    end
end

@inline function periodic!(r::AbstractVector{T}, L::Float64) where T<:Number
    """Periodic shift to box [-L/2, L/2]"""
    # same as @. r = mod(r+L/2, L) - L/2, but faster:
    @inbounds for d in eachindex(r)
        r[d] = mod(r[d] + L/2, L) - L/2
    end
    r
end

function distance!(r1::AbstractVector{T}, r2::AbstractVector{T}, L::Float64, dr::AbstractVector{T}) where T<:Real
    """Minimum image distance vector r1-r2 for pbc, r1-r2 for non-pbc"""
    
    if pbc
        @inbounds for d in eachindex(dr)
            dr[d] = r1[d] - r2[d]
            dr[d] -= L * round(dr[d]/L)
        end
    else
        @inbounds for d in eachindex(dr)
            dr[d] = r1[d] - r2[d]
        end
    end
end


function dist(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64) 
    """Minimum image distance |r1-r2| for pbc, distance for non-pbc"""
    r12 = vec_buffer 
    @inbounds for d in eachindex(r12)
        r12[d] = r1[d] - r2[d]
    end
    r = 0.0
    if pbc
        @inbounds for d in eachindex(r12)
            r12[d] -= L * round(r12[d]/L)
            r += r12[d]^2
        end
    else
        @inbounds for d in eachindex(r12)
            r += r12[d]^2
        end        
    end  
    r = sqrt(r)
end



function dist2(r1::AbstractVector{Float64}, r2::AbstractVector{Float64}, L::Float64)
    """Minimum image distance |r1-r2| for pbc, distance for non-pbc"""
    r12 = vec_buffer
    @inbounds for d in eachindex(r12)
        r12[d] = r1[d] - r2[d]
    end
    r = 0.0
    if pbc
        @inbounds for d in eachindex(r12)
            r12[d] -= L * round(r12[d]/L)
            r += r12[d]^2
        end
    else
        @inbounds for d in eachindex(r12)
            r += r12[d]^2
        end        
    end  
    r 
end


end

