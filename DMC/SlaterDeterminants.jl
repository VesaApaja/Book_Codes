# Slater Determinant functions

module SlaterDeterminants

using ForwardDiff: Dual

using Common
import Common: AtomData, dim
using StaticArrays
import LinearAlgebra: det, ⋅
    
import HydrogenicOrbitals: orb as H_orb, ∇orb as ∇H_orb, ∇2orb as ∇2H_orb
import STOOrbitals: orb as STO_orb, ∇orb as ∇STO_orb, ∇2orb as ∇2STO_orb
export D_up, D_do, ∇2_up, ∇2_do, F_up, F_do

#export use_orb, set_orb

# Function pointers
const orb_ref   = Ref{Function}(x -> println("Function orb is not set"))
const ∇orb_ref  = Ref{Function}(x -> println("Function ∇orb is not set"))
const ∇2orb_ref = Ref{Function}(x -> println("Function ∇2orb is not set"))

# Function to call orb directly
@inline function orb(x ::MVector, ao ::AtomicOrbital)
    orb_ref[](x, ao)
end
@inline function ∇orb(x ::MVector, ao ::AtomicOrbital)
    ∇orb_ref[](x, ao)
end
@inline function ∇2orb(x ::MVector, ao ::AtomicOrbital)
    ∇2orb_ref[](x, ao)
end


# Set the orb function to use
function set_orb(orb::Function, ∇orb ::Function, ∇2orb ::Function)
    orb_ref[] = orb    
    ∇orb_ref[] = ∇orb
    ∇2orb_ref[] = ∇2orb
    @show(orb,∇orb,∇2orb)
end

function init(basis_set::String)
    if basis_set == "Hydrogenic"
        set_orb(H_orb, ∇H_orb, ∇2H_orb)

    elseif startswith(basis_set, "STO")
        set_orb(STO_orb, ∇STO_orb, ∇2STO_orb)

    else
        error("Unknown orbital basis: $basis_set")
    end
end



# Slater determinant primitives

@inline function Slater_up_old(R ::MMatrix{dim,N,Float64}, ups) where {dim, N} 
    Nup = length(ups)

    dtype = eltype(orb(R[:,1], ups[1]))
    
    d = Matrix{dtype}(undef,Nup,Nup)
    for (k,ao) in enumerate(ups)
        for i ∈ 1:Nup
            d[k,i] = orb(R[:,i], ao)
        end
    end
    d
end

@inline function Slater_up(R::MMatrix{dim,N,Float64}, ups) where {dim,N}
    T = Float64

    for ao in ups
        T = promote_type(T, eltype(ao.exponents), eltype(ao.coefficients))
    end

    Slater_up_kernel(R, ups, T, orb_ref[])
end

function Slater_up_kernel(R::MMatrix{dim,N,Float64}, ups, ::Type{T}, orbital_function::F) where {dim,N,T,F}

    Nup = length(ups)
    d = Matrix{T}(undef, Nup, Nup)

    @inbounds for (k,ao) in enumerate(ups)
        for i in 1:Nup
            d[k,i] = orbital_function(R[:,i], ao)
        end
    end

    return d
end

@inline function Slater_do(R::MMatrix{dim,N,Float64}, ups) where {dim,N}
    T = Float64

    for ao in ups
        T = promote_type(T, eltype(ao.exponents), eltype(ao.coefficients))
    end

    Slater_do_kernel(R, ups, T, orb_ref[])
end


function Slater_do_kernel(
    R::MMatrix{dim,N,Float64},
    dos,
    ::Type{T},
    orbital_function::F
) where {dim,N,T,F}

    Ndo = length(dos)
    Nup = N-Ndo
    d = Matrix{T}(undef, Ndo, Ndo)

    @inbounds for (k,ao) in enumerate(dos)
        for i in 1:Ndo
            d[k,i] = orbital_function(R[:,i+Nup], ao)
        end
    end
    return d
end


@inline function Slater_do_old(R ::MMatrix{dim,N,Float64}, dos) where {dim, N} 

    Ndo = length(dos)
    Nup = N-Ndo
    
        
    dtype = eltype(orb(R[:,1+Nup], dos[1]))
    d = Matrix{dtype}(undef,Ndo,Ndo)
    for (k,ao) in enumerate(dos)
        for i ∈ 1:Ndo # *not* a coordinate index
            d[k,i] = orb(R[:,i+Nup], ao)
        end
    end
    d
end


@inline function D_up(R ::MMatrix{dim,N,Float64}, ups) where {dim, N} 
    Nup = length(ups)
    if Nup==0
        D_up = 1.0 
        return D_up
    end
    d = Slater_up(R, ups)
    if Nup==1
        D_up = d[1,1]
    elseif Nup==2
        D_up = d[1,1]*d[2,2]-d[2,1]*d[1,2]        
    else        
        D_up = det(d)
    end
    D_up
end

@inline function D_do(R ::MMatrix{dim,N,Float64}, dos) where {dim, N} 
    
    Ndo = length(dos)
    if Ndo==0
        D_do = 1.0 
        return D_do
    end
    d = Slater_do(R, dos)
    if Ndo==1
        D_do = d[1,1]
    elseif Ndo==2
        D_do = d[1,1]*d[2,2]-d[2,1]*d[1,2]
    else
        D_do = det(d)
    end
    D_do
end



# 2*nabla|D|/|D| spin up

@inline function F_up(R::MMatrix{dim,N,Float64}, ups) where {dim,N}

    Dinv = get_D_up_inv(R, ups)

    return F_up_kernel(R, ups, Dinv, ∇orb_ref[])
end


function F_up_kernel(R::MMatrix{dim,N,Float64}, ups, Dinv::AbstractMatrix{T}, gradient_function::G) where {dim,N,T,G}

    Nup = length(ups)
    F = zeros(T, dim, N)

    @inbounds for i in 1:Nup
        ri = R[:,i]

        for (k,ao) in enumerate(ups)
            g = gradient_function(ri, ao)
            weight = 2 * Dinv[i,k]

            for d in 1:dim
                F[d,i] += weight * g[d]
            end
        end
    end

    return F
end

# 2*nabla|D|/|D| spin down 
@inline function F_do(R::MMatrix{dim,N,Float64}, dos) where {dim,N}

    Dinv = get_D_do_inv(R, dos)

    return F_do_kernel(R, dos, Dinv, ∇orb_ref[])
end

function F_do_kernel(R::MMatrix{dim,N,Float64}, dos, Dinv::AbstractMatrix{T}, gradient_function::G) where {dim,N,T,G}

    Ndo = length(dos)
    Nup = N-Ndo
    F = zeros(T, dim, N)

    @inbounds for i in Nup+1:N
        ri = R[:,i]

        for (k,ao) in enumerate(dos)
            g = gradient_function(ri, ao)
            weight = 2 * Dinv[i-Nup,k]
            for d in 1:dim
                F[d,i] += weight * g[d]
            end
        end
    end

    return F
end


@inline function get_D_up_inv(R::MMatrix{dim,N,Float64}, ups) where {dim,N}

    d = Slater_up(R, ups)
    Nup = length(ups)

    if Nup == 1
        dinv = similar(d)
        dinv[1,1] = inv(d[1,1])

    elseif Nup == 2
        a = d[1,1]
        b = d[1,2]
        c = d[2,1]
        e = d[2,2]

        det = a*e - b*c

        dinv = similar(d)
        dinv[1,1] =  e/det
        dinv[1,2] = -b/det
        dinv[2,1] = -c/det
        dinv[2,2] =  a/det

    else
        dinv = inv(d)
    end

    return dinv
end

@inline function get_D_do_inv(R::MMatrix{dim,N,Float64}, dos) where {dim,N}

    d = Slater_do(R, dos)
    Ndo = length(dos)
    Nup = N-Ndo
    
    if Ndo == 1
        dinv = similar(d)
        dinv[1,1] = inv(d[1,1])

    elseif Ndo == 2
        a = d[1,1]
        b = d[1,2]
        c = d[2,1]
        e = d[2,2]

        det = a*e - b*c

        dinv = similar(d)
        dinv[1,1] =  e/det
        dinv[1,2] = -b/det
        dinv[2,1] = -c/det
        dinv[2,2] =  a/det

    else
        dinv = inv(d)
    end

    return dinv
end



@inline function ∇2_up(
    R::MMatrix{dim,N,Float64}, ups
) where {dim,N}

    Dinv = get_D_up_inv(R, ups)

    return ∇2_up_kernel(R, ups, Dinv, ∇2orb_ref[])
end


function ∇2_up_kernel(
    R::MMatrix{dim,N,Float64},
    ups,
    Dinv::AbstractMatrix{T},
    lap_function::F
) where {dim,N,T,F}

    lap = zero(T)

    @inbounds for i in 1:length(ups)
        ri = R[:,i]

        for k in eachindex(ups)
            lap += lap_function(ri, ups[k]) * Dinv[i,k]
        end
    end

    return lap
end


# ∇^2 |D_up| / |D_up|
@inline function ∇2_up_old(R ::MMatrix{dim,N,Float64}, ups) where {dim, N} 
   
    Nup = length(ups)   
    D_up_inv = get_D_up_inv(R, ups)
    ∇2 = 0.0
    for i ∈ 1:Nup
        for (k,ao) in enumerate(ups)
            ∇2 += ∇2orb(R[:,i], ao) * D_up_inv[i,k]
        end
    end
    ∇2
end

# ∇^2 |D_do| / |D_do|
@inline function ∇2_do(R ::MMatrix{dim,N,Float64}, dos) where {dim, N} 

    Ndo = length(dos)
    Nup = N-Ndo
    D_do_inv = get_D_do_inv(R, dos)
    ∇2 = 0.0
    for i ∈ Nup+1:N # coordinate index
        for (k,ao) in enumerate(dos)
            ∇2 += ∇2orb(R[:,i], ao) * D_do_inv[i-Nup,k]
        end
    end
    ∇2
end


# =================================================
#=
# |D^new| /|D^old| *not squared*,  if only i:th particle has moved 
function get_D_ratio(Rold ::MyArrType, Rnew ::MyArrType, i ::Int64) :: Float64
    ratio = 0.0 ::Float64
    if i <= Nup
        for (c,orbs) ∈ zip(coefflist,orblist)
            D_up_inv = get_D_up_inv(orbs, atom_data, Rold)
            for j = 1:Nup
                ratio += c*orb(j, Rnew[:,i]) * D_up_inv[j,i]
            end
        end            
    else
        for (c,orbs) ∈ zip(coefflist,orblist)
            D_do_inv = get_D_do_inv(orbs,Rold) 
            for j = 1: Ndo
                ratio += c*orb(j, Rnew[:,i]) * D_do_inv[j,i-Nup]
            end
        end
    end
    ratio
end
=#

end


