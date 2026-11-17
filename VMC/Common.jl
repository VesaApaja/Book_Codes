# Common to many QMC programs

module Common
using StaticArrays
using ForwardDiff
# using ReverseDiff
using DelimitedFiles: readdlm
using CurveFit

import Base: one, zero

# slightly faster norm 
@inline norm(x) = sqrt(sum(abs2,x))

export Walker, init_walkers, VMC_Params, spinfactor
export branch!

export AtomicOrbital, LinearCombination, JastrowParameters
export dim, D
export V

export parse_atom_data, parse_basis_set_data, parse_Jastrow_data, parse_STO, parse_coeffients_data
export WfParams
export update_wf_params, copy_wf_params!
export det_coeff


const dim = 3    # space dimension
const D = 0.5    # hbar^2/2m in a.u. 


struct AtomData
    atom::String
    N::Int
    Nup::Int
    Ndo::Int
    Z::Int
    Eexact::Float64
end

# Use two Number types, because AD derives exponents and coefficients separately
# ForwardDiff uses Dual type, ReverseDiff uses TrackedArray type
struct AtomicOrbital{A<:AbstractString, B<:AbstractString, T<:Number, Z<:Number}
    atom         ::A
    orbital_type ::B
    n            ::Int
    l            ::Int
    exponents    ::Vector{T}
    coefficients ::Vector{Z}
    spin         ::A
end

struct LinearCombination{T,A}
    orbitals::A
    csf_coeffs::Vector{T}
    det_csf::Vector{Int}
    det_weight::Vector{Float64}
end

function DD(ao::AtomicOrbital, R::MMatrix{dim,N,Float64}) where {dim, N}
    DD = 0.0    
    for (i,aos) in enumerate(lc.orbitals)
        ups = aos[lc_ups[i]]
        dos = aos[lc_dos[i]]
        DD += lc.coeffs[i] * D_up(R, ups) * D_do(R, dos)
    end
end

struct JastrowParameters{T<:Number}
    param ::Vector{T}
end

struct WfParams
    lc ::LinearCombination
    jpara ::JastrowParameters
end

function one(::Type{AtomicOrbital})
    return AtomicOrbital("", "", 0,0,ones(Float64, 0), ones(Float64, 0), 0)
end


function zero(::Type{AtomicOrbital})
    return AtomicOrbital("", "", zeros(Float64, 0), zeros(Float64, 0), 0)
end

import Base: +

function +(ao::AtomicOrbital, b::AtomicOrbital) 
        AtomicOrbital(ao.atom, ao.orbital_type, ao.n, ao.l, ao.exponents .+ b.exponents, ao.coefficients .+ b.coefficients, ao.spin)
end

# get the actual determinant coefficient
@inline det_coeff(lc::LinearCombination, i::Int) =
    lc.csf_coeffs[lc.det_csf[i]] * lc.det_weight[i]

# In case of ReverseDiff, define + between TrackedArray and StaticArray types
# commented out, was a workaround for older version of ReverseDiff
# function +(x::ReverseDiff.TrackedArray, y::StaticArray)
#    return x .+ y  # Element-wise broadcasting
#end
# function +(x::StaticArray, y::ReverseDiff.TrackedArray)
#    return x .+ y  # Element-wise broadcasting
#end

# define WfParams + Δα
# Notice that Δα is in a specific order
#  
  

function update_wf_params(a::WfParams, x::AbstractVector)
    lc = a.lc
    pos = 1

    # ---------------------------------------------
    # Update independent CSF coefficients
    # Keep the first CSF coefficient fixed
    # ---------------------------------------------

    ncsf = length(lc.csf_coeffs)
    nfree = ncsf - 1

    new_csf_coeffs = vcat(
        lc.csf_coeffs[1:1],
        lc.csf_coeffs[2:end] .+ x[pos:pos+nfree-1]
    )
    pos += nfree

    # ---------------------------------------------
    # Update unique orbital parameters
    # ---------------------------------------------

    P = promote_type(eltype(x), Float64)

    new_params = Dict{
        Tuple{Int,Int},
        Tuple{Vector{P},Vector{P}}
    }()

    for aos in lc.orbitals
        for ao in aos
            key = (ao.n, ao.l)

            if !haskey(new_params, key)

                len = length(ao.exponents)
                @assert length(ao.coefficients) == len

                # Limit exponent changes to +/- 10%
                new_exponents = [
                    begin
                        ζ = ao.exponents[i]
                        dd = x[pos+i-1]
                        ζ + clamp(dd, -0.10*ζ, 0.10*ζ)
                    end
                    for i in 1:len
                ]

                if !all(z -> isfinite(z) && z > 0.0, new_exponents)
                    println("update_wf_params: invalid orbital exponent")
                    return a
                end

                pos += len

                new_coefficients =
                    ao.coefficients .+ x[pos:pos+len-1]

                pos += len

                new_params[key] =
                    (new_exponents, new_coefficients)
            end
        end
    end

    # ---------------------------------------------
    # Reconstruct orbitals, preserving sharing
    # ---------------------------------------------

    orbitals = Vector{AtomicOrbital}[]

    for aos in lc.orbitals
        new_aolist = AtomicOrbital[]

        for ao in aos
            exponents, coefficients = new_params[(ao.n, ao.l)]

            new_ao = AtomicOrbital(
                ao.atom,
                ao.orbital_type,
                ao.n,
                ao.l,
                exponents,
                coefficients,
                ao.spin
            )

            push!(new_aolist, new_ao)
        end

        push!(orbitals, new_aolist)
    end

    # Preserve the CSF -> determinant mapping
    new_lc = LinearCombination(
        orbitals,
        new_csf_coeffs,
        lc.det_csf,
        lc.det_weight
    )

    # ---------------------------------------------
    # Update Jastrow parameters
    # ---------------------------------------------

    nj = length(a.jpara.param)

    @assert pos + nj - 1 == length(x)

    param = a.jpara.param .+ x[pos:end]

    if !all(isfinite, param)
        error("update_wf_params: invalid Jastrow parameters")
        @show param
        return a
    end
    # keep Jastrow β1 and β2 > 0, use a lower limit (only if they were nonzero to begin with)
    if a.jpara.param[2]>0 ||  a.jpara.param[3]>0
        if param[2]<0.001 || param[3]<0.001
            @info "Jastrow β1 or β2 too low (<0.001), force min values"
            param[2] = max(param[2], 0.001)
            param[3] = max(param[3], 0.001)
        end
    end

    new_jpara = JastrowParameters(param)

    return WfParams(new_lc, new_jpara)
end



function copy_wf_params!(dest::WfParams, src::WfParams)

    # CSF / determinant coefficients
    copyto!(dest.lc.coeffs, src.lc.coeffs)

    # Atomic orbital parameters
    @assert length(dest.lc.orbitals) == length(src.lc.orbitals)

    for (aos_dest, aos_src) in
        zip(dest.lc.orbitals, src.lc.orbitals)

        @assert length(aos_dest) == length(aos_src)

        for (ao_dest, ao_src) in zip(aos_dest, aos_src)
            @assert (ao_dest.n, ao_dest.l, ao_dest.spin) ==
                    (ao_src.n, ao_src.l, ao_src.spin)

            copyto!(ao_dest.exponents, ao_src.exponents)
            copyto!(ao_dest.coefficients, ao_src.coefficients)
        end
    end

    # Jastrow parameters
    copyto!(dest.jpara.param, src.jpara.param)

    return dest
end



mutable struct VMC_Params
    Ntry :: Int64
    Naccept :: Int64
    step :: Float64
end 


mutable struct Walker
    R     :: MMatrix
    alive :: Int64
    Ψ     :: Float64    
    E     :: Float64
    age   :: Float64
    weight :: Float64
    pure_data ::Union{Nothing, T} where T
    measurements :: Dict{Symbol, Any} # other measured quantities
    # constructor
    Walker(; R, alive, Ψ, E, age, weight, pure_data = nothing) =
        new(R, alive, Ψ,  E, age, weight, pure_data, Dict{Symbol, Any}())
end


# 
# Initialize walkers
# ==================
# Simplified version
# 
function init_walkers(Nw ::Int64, dim ::Int64, N ::Int64) ::Array{Walker,1}

    println("Init $Nw walkers  - note: most values set to zero")
    MMzeros = @MMatrix zeros(dim,N)
    # take copy of MMzeros, else the R's reference the same memory location
    walker = [Walker(R = copy(MMzeros), alive=0, Ψ=0.0, E=0.0, age=0.0,
                     weight=1.0) for i in 1:Nw]

    walker
end

# Initialization with trial wave function
function init_walkers(Nw ::Int64,
                      Nwx ::Int64,
                      dim ::Int64,
                      N ::Int64;
                      φ_T  ::Function = dummy,
                      scale ::Float64 = 1.0,                      
                      ) ::Array{Walker,1}
    #
    println("generating $Nw walkers, storage for $Nwx walkers")
    MMzeros = @MMatrix zeros(dim,N)
    # take copy of MMzeros, else the R's reference the same memory location
    walker = [Walker(R = copy(MMzeros), alive=0, Ψ=0.0, E=0.0, age=0.0, weight=1.0) for i in 1:Nwx]

    # set Nw walkers
    for w ∈ walker[1:Nw]
        w.alive = 1
        while true
            R = @MMatrix rand(dim,N)
            R .*= scale # scale a bit
            Ψ = φ_T(R)
            if Ψ > 0
                copyto!(w.R, R)  
                w.Ψ = Ψ
                break
            end
        end        
    end
    walker
end

    

# Walker to Walker copy
@inline function copy!(dest::Walker, src::Walker)
    if dest.R === src.R
        error("dest and src R's reference the same memory location.")
    end
    dest.R .= src.R    # broadcast
    dest.alive = src.alive
    dest.Ψ = src.Ψ    
    dest.E = src.E
    dest.age = src.age
    dest.weight = src.weight
    if src.pure_data !== nothing
        dest.pure_data = deepcopy(src.pure_data) 
    else
        dest.pure_data = nothing  
    end
    dest.measurements =  deepcopy(src.measurements)
end



@inline function V(R ::MMatrix, N ::Int64, Z ::Int64) ::Float64
    V = 0.0 ::Float64
    for k in 1:N
        rk  = norm(R[:,k])
        V += -Z/rk
        for i in k+1:N
            rki = norm(R[:,k]-R[:,i]) 
            V += 1/rki
        end
    end
    V
end

# ====================

# spin factor
function spinfactor(Nup, Ndo)
    N = Nup+Ndo
    s = zeros(Int64,N,N)
    for i ∈ 1:N
        for j ∈ 1:N
            s[i,j] = spins(i, j, Nup, Ndo)
        end
    end
    s
end

function spins(i ::Int64, j ::Int64, Nup ::Int64, Ndo ::Int64) ::Int64    
    if (i<= Nup && j>Nup) || (i> Nup && j<=Nup)
        # anti-parallel spins                
        fac = 2 ::Int64    
    else
        # parallel spins
        fac = 4 ::Int64    
    end
    fac
end

function print_basis_set(basis_set_data)
    for ao in basis_set_data
        println("Orbital Type: ", ao.orbital_type)
        try
            println("Exponents: ", [ζ.value for ζ in ao.exponents])
            println("Coefficients: ", [c.value for c in ao.coefficients])
        catch
            println("Exponents: ", ao.exponents)
            println("Coefficients: ", ao.coefficients)
        end
    end
end


function dummy(x ::Any)
    return 1.0
end






# Common DMC functions
# ====================


function branch!(walker ::Vector{Walker}, copies ::MVector, alives ::Set{Int64})
    Nwx = length(walker)
    # list initially dead walkers that can we can branch to
    deads = findall(iw -> iw ∉ alives, 1:Nwx)
    idead = 1
    orig_alives = copy(alives) # modify alives in loop
    @inbounds for iw in orig_alives
        if copies[iw]==1
            continue        # one copy is already there
        elseif copies[iw]==0
            # walker dies
            walker[iw].alive = 0
            delete!(alives, iw)
            continue
        end
        # copies[iw]>1
        # copy the walker to empty slots 
        @inbounds for inew in 1:copies[iw]-1   # with copies=3 this is 1,2
            # use free slot at idead
            if idead > length(deads)
                @show length(alives)
                error("No free walkers available; too large τ ?")
            end
            iw2 = deads[idead]
            copy!(walker[iw2], walker[iw]) # walker to walker copy
            push!(alives, iw2)             
            idead += 1 # take next free slot 
        end
    end
    return nothing
end


# ================
# Parsers
# =================





function parse_basis_set_data(atom::String, basis_set ::String)
    STOfile = "atom_data/$(atom)_$(basis_set)"
    if isfile(STOfile)
        basis_set_data = parse_STO(STOfile)
    else
        error("File not found: $STOfile")
    end
    #
    #    # Fall back to STO-3g data and fit them to STO
    #    println("Warning: Failed to read data from $STOfile, falling back to STO-3g")
    #    error("not tested")
    #    STO3gfile = "atom_data/"*atom*"_STO3g"
    #    basis_set_data = parse_turbomole_sto3g(STO3gfile)
    #    basis_set_data, Spara, nonlin = fit_STO3g_to_STO(basis_set_data)
    #    Norb = 0 # not needed
    #end
    basis_set_data
end

function parse_Jastrow_data(atom::String, basis_set ::String) ::JastrowParameters
    filename = "atom_data/"*atom*"_"*basis_set
    println("Reading Jastrow wf parameters from file $filename")
    α, β1, β2, α12 = 0.0, 0.0, 0.0, 0.0
    open(filename, "r") do file
        while !eof(file)
            line = strip(readline(file))
            # Remove comments
            line = split(line, "#")[1]
            line = strip(line)
            if startswith(line, "α =")
                α = parse(Float64, strip(split(line, "=")[2]))
            elseif startswith(line, "β1 =")
                β1 = parse(Float64, strip(split(line, "=")[2]))
            elseif startswith(line, "β2 =")
                β2 = parse(Float64, strip(split(line, "=")[2]))
            elseif startswith(line, "α12 =")
                α12 = parse(Float64, strip(split(line, "=")[2]))
            end
        end
    end
    para = [α, β1, β2, α12]
    return JastrowParameters(para)
end


function parse_coeffients_data(atom::String, basis_set ::String)
    filename = "atom_data/"*atom*"_"*basis_set
    println("Reading linear combination coefficients from file $filename")
    cnls=[]
    open(filename, "r") do file
        while !eof(file)
            line = strip(readline(file))
            # Remove comments
            line = split(line, "#")[1]
            line = strip(line)
            if startswith(line, "coeff =")                
                cc = parse(Float64, strip(split(line, "=")[2]))
                append!(cnls,cc)
            end
            nl = Int[]
            if startswith(line, "orbitals =")
                parts = split(line)               
                for p in parts[3:end]
                    snl = string(parse(Int,p))
                    n = parse(Int, snl[1])
                    l = parse(Int, snl[2])
                    append!(nl,n,l)
                end
                append!(cnls,[nl])  
            end
        end
    end
    # output looks like
    # cnls = Any[1.0, [1, 0, 1, 0, 2, 0, 2, 0], -0.4464125065, [1, 0, 1, 0, 2, 1, 2, 1],
    #            -0.4464125065, [1, 0, 1, 0, 2, 1, 2, 1], -0.4464125065, [1, 0, 1, 0, 2, 1, 2, 1]]

    return cnls
end



# Function to parse the data from the file
function parse_atom_data(atom::String)
    N = 0
    Nup = 0
    Ndo = 0
    Z = 0
    Eexact = 0.0
    

    filename = "atom_data/"*atom
    println("reading atom data from file $filename")
    open(filename, "r") do file
        while !eof(file)
            line = strip(readline(file))
            # Remove comments
            line = split(line, "#")[1]
            line = strip(line)
            
            if isempty(line)
                continue
            end
            if startswith(line, "N =")
                N = parse(Int, strip(split(line, "=")[2]))
            elseif startswith(line, "Nup =")
                Nup = parse(Int, strip(split(line, "=")[2]))
            elseif startswith(line, "Ndo =")
                Ndo = parse(Int, strip(split(line, "=")[2]))
            elseif startswith(line, "Z =")
                Z = parse(Int, strip(split(line, "=")[2]))
            elseif startswith(line, "Eexact =")
                Eexact = parse(Float64, strip(split(line, "=")[2]))               
            end
        end
    end
    
    return AtomData(atom, N, Nup, Ndo, Z, Eexact)
end

# Function to parse the atomic orbital data from STO file  
function parse_STO(filename ::String)
    l_dict  = Dict("s"=>0, "p"=>1, "d"=>2, "e"=>3, "f"=>4, "g"=>5, "h"=>6, "i"=>7, "j"=>8)
    fill_order = [1,2,2,3,3,4,3,4,5,4,5,6,4,5,6]
    
    
    println("Reading atomic orbital data from file $filename")
    atom = ""  
    orbitals = AtomicOrbital[]
    i = 0
    println("="^80)
    println("Known orbitals and parameters:") 
    open(filename, "r") do file
        while !eof(file)
            line = strip(readline(file))
            
            if startswith(line, "#")
                # Skip comment lines
                continue
            elseif startswith(line, "*")
                # End of a section
                continue
            elseif isempty(atom)
                # First non-comment line is the atom
                atom = string(split(line)[1]) #just to avoid being a substring
            else
                # Parsing the orbital data
                parts = split(line)
                
                if length(parts) == 2 && parse(Int, parts[1]) > 0
                    # Start of a new orbital section
                    num_primitives = parse(Int, parts[1])
                    orbital_type = parts[2]
                    
                    
                    exponents = Float64[]
                    coefficients = Float64[]
                    
                    for _ in 1:num_primitives
                        line = strip(readline(file))
                        values = split(line)
                        push!(exponents, parse(Float64, values[1]))
                        push!(coefficients, parse(Float64, values[2]))
                    end
                    i += 1
                    n = fill_order[i]
                    l = l_dict[orbital_type]
                    s = string("$n"*orbital_type)
                    orbital =  AtomicOrbital(atom, s, n, l, exponents, coefficients, "spin_not_set")
                    @show orbital
                    push!(orbitals, orbital)
                end
            end 
        end
    end
    return orbitals 
end


# Function to parse the atomic orbital data from the Turbomole file
function parse_turbomole_sto3g(STO3gfile::String)
    orbitals = []
    filename = STO3gfile
    println("Reading atomic orbital data from file $filename")
    atom = ""
    
    open(filename, "r") do file
        while !eof(file)
            line = strip(readline(file))
            
            if startswith(line, "#")
                # Skip comment lines
                continue
            elseif startswith(line, "*")
                # End of a section
                continue
            elseif isempty(atom)
                # First non-comment line is the atom
                atom = split(line)[1]
            else
                # Parsing the orbital data
                parts = split(line)
                
                if length(parts) == 2 && parse(Int, parts[1]) > 0
                    # Start of a new orbital section
                    num_primitives = parse(Int, parts[1])
                    orbital_type = parts[2]
                    
                    exponents = Float64[]
                    coefficients = Float64[]
                    
                    for _ in 1:num_primitives
                        line = strip(readline(file))
                        values = split(line)
                        push!(exponents, parse(Float64, replace(values[1], "D" => "E")))
                        push!(coefficients, parse(Float64, replace(values[2], "D" => "E")))
                    end
                    
                    orbital =  AtomicOrbital(atom, orbital_type, exponents, coefficients)
                    push!(orbitals, orbital)
                end
            end
        end
    end
    
    return orbitals
end
#=
# Example usage

orbitals = parse_turbomole_sto3g(filename)

# Print the parsed data
for orbital in orbitals
println("Atom: ", orbital.atom)
println("Orbital Type: ", orbital.orbital_type)
println("Exponents: ", orbital.exponents)
println("Coefficients: ", orbital.coefficients)
println()
end
=#


end

