__precompile__(false)
module PIMC_Common


include("PIMC_Systems.jl")
import .PIMC_Systems.HarmonicOscillator
import .PIMC_Systems.HeLiquid
import .PIMC_Systems.Noninteracting

export λ, pbc, bose, Ntherm, system, τ_target
export action, results
export N, dim, worm_K, worm_C, worm_limit, swap_limit
export restart
export periodic!, distance, distance!, dist, dist2
export pull!
export AbstractAction, PrimitiveAction, ChinAction

# ==========================
# CHOOSE SYSTEM AND ACTION

abstract type AbstractAction end
struct PrimitiveAction <: AbstractAction end
struct ChinAction <: AbstractAction end


case = 1

if case==1
    sys = :HarmonicOscillator
    restart = false
    const N = 2
    const bose = true
    const dim = 1
    const pbc = false
    const λ = 0.5
    const τ_target = 0.05
    const action = ChinAction # PrimitiveAction 
    const worm_limit = 1.e6
    const swap_limit = 1.e6
    const Ntherm = 10000
    const worm_K = 8
    const worm_C = 0.5
end

if case==216 
    sys = :HeLiquid
    restart = true
    const N = 16
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ_target = 0.01
    const action = ChinAction
    const worm_limit = 100.0
    const swap_limit = 1e6
    const Ntherm = 10000 
    const worm_K = 90
    const worm_C = 0.6
end


if case==232
    sys = :HeLiquid
    restart = true 
    const N = 32
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ_target = 0.01
    const action = ChinAction
    const worm_limit = 100.0
    const swap_limit = 1e6
    const Ntherm = 10000
    const worm_K = 90  
    const worm_C = 0.6
end

if case==264
    sys = :HeLiquid
    restart = true
    const N = 64
    const bose = true 
    const dim = 3
    const pbc = true
    const λ = 6.0612686
    const τ_target = 0.01
    const action = ChinAction
    const worm_limit = 200.0
    const swap_limit = 1e6
    const Ntherm = 10000
    const worm_K = 130 
    const worm_C = 0.6
end


if case==3
    sys = :Noninteracting
    restart = false
    const N = 64
    const bose = true
    const dim = 3
    const λ = 0.5
    const pbc = true
    const τ_target = 0.1 # not too large, or large winding causes min. image errors  
    const action = ChinAction
    const worm_limit = 100.0
    const swap_limit = 1.e6 
    const Ntherm = 1000    
    const worm_K = 10
    const worm_C = 0.3
end



# =====================================

const system = eval(:(PIMC_Systems.$sys))

results = Dict{String, Any}()
Vtail = 0.0 # set in PIMC_main



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

# benchmarked against LoopVectorization and @tturbo: copying to Vector's takes longer 
function distance(r1::SubArray{T}, r2::SubArray{T}, L::Float64) where T<:Number
    """Minimum image distance vector r1-r2 for pbc, r1-r2 for non-pbc"""
    error("use distance! ")
    vecr12 = similar(r1)
    @inbounds for d in eachindex(vecr12)
        vecr12[d] = r1[d] - r2[d]
    end
    if pbc
        @inbounds for d in eachindex(vecr12)
            vecr12[d] -= L * round(vecr12[d]/L)
        end
    end  
    vecr12
end

# avoid creating new vector
function distance!(r1::AbstractVector{T}, r2::AbstractVector{T}, L::Float64, dr::AbstractVector{T}) where T<:Number
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


function dist(r1::SubArray{T}, r2::SubArray{T}, L::Float64) where T<:Number
    """Minimum image distance |r1-r2| for pbc, distance for non-pbc"""
    vecr12 = similar(r1)
    @inbounds for d in eachindex(vecr12)
        vecr12[d] = r1[d] - r2[d]
    end
    r = 0.0
    if pbc
        @inbounds for d in eachindex(vecr12)
            vecr12[d] -= L * round(vecr12[d]/L)
            r += vecr12[d]^2
        end
    else
        @inbounds for d in eachindex(vecr12)
            r += vecr12[d]^2
        end        
    end  
    r = sqrt(r)
end


function dist2(r1::SubArray{T}, r2::SubArray{T}, L::Float64) where T<:Number
    """Minimum image distance |r1-r2| for pbc, distance for non-pbc"""
    vecr12 = similar(r1)
    @inbounds for d in eachindex(vecr12)
        vecr12[d] = r1[d] - r2[d]
    end
    r = 0.0
    if pbc
        @inbounds for d in eachindex(vecr12)
            vecr12[d] -= L * round(vecr12[d]/L)
            r += vecr12[d]^2
        end
    else
        @inbounds for d in eachindex(vecr12)
            r += vecr12[d]^2
        end        
    end  
    r 
end


end

