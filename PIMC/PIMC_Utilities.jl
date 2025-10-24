__precompile__(false)
module PIMC_Utilities

using LinearAlgebra
using StaticArrays
push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Structs:t_beads, t_links, t_pimc

export distance_check
export store_X!, restore_X!, storage
export rho_0, active_beads_on_slice, bisection_segment!
export generate_path!
export backup_state!, restore_state!
export boson_virial_exchange_energy, E_pot, E_pot_bead
export save_paths
export argparse



# =====================================
# ArgParse would do this, but I hate to load yet another package

function argparse(known_args ::Vector{String})
    if length(ARGS)==0; return nothing ; end
    
    got_args = Dict() 
    for arg in ARGS
        if occursin("=",arg)
            parts = split(arg, "=", keepempty=false)
            var = parts[1]
            if var ∉ known_args
                @show(known_args)
                println("Error reading command line arguments, variable $var is unknown")
                exit(1)
            end
            try
                value = parts[2]                
                if occursin(".",value)
                    value = parse(Float64,value)
                else
                    try
                        value = parse(Int64,value)
                    catch
                        # assume it's a string
                        value = String(value)
                    end
                end
                #println("command line: ",var,"=",value)
                got_args[var]=value
            catch
                println("Error reading command line arguments, use var=val")
                exit(1)
            end
        end
    end
    return got_args
end
# ======================================




@inline function active_beads_on_slice(beads::t_beads, m::Int64)
    mask = beads.active[beads.at_t[m]]  
    return beads.at_t[m][mask]      
end

# Temporary storage for a few bead positions in worm moves and in bisection

struct t_storage
    buffer::Matrix{Float64} # Preallocated storage
    k::Base.RefValue{Int64}   # number of stored values
end

function t_storage(K::Int64)
    t_storage(zeros(dim, K),  Ref(0))
end
storage = t_storage(max(10, 10000)) # should be enough  


function store_X!(storage::t_storage, beads::t_beads, idlist::AbstractVector{Int64})
    k = length(idlist)
    k > size(storage.buffer, 2) && error("Storage too small")
    storage.k[] = k           # set number of stored elements
    @inbounds for i = 1:k, d = 1:dim
        storage.buffer[d, i] = beads.X[d, idlist[i]] 
    end
end

# Restore values if move is rejected
function restore_X!(storage::t_storage, beads::t_beads,  idlist::AbstractVector{Int64})
    k = storage.k[]   # get number of stored elements
    @inbounds for i = 1:k, d = 1:dim
        beads.X[d, idlist[i]] = storage.buffer[d, i]
    end
end


function backup_state!(beads_backup::t_beads, links_backup::t_links, 
                       beads::t_beads, links::t_links)
    # these are already fixed: beads.ids, beads.ts, beads.at_t, beads.times       
    @inbounds beads_backup.X .= beads.X
    @inbounds beads_backup.active .= beads.active
    @inbounds links_backup.next .= links.next
    @inbounds links_backup.prev .= links.prev
end

function restore_state!(beads::t_beads, links::t_links, 
                        beads_backup::t_beads, links_backup::t_links)
    @inbounds beads.X .= beads_backup.X
    @inbounds beads.active .= beads_backup.active 
    @inbounds links.next .= links_backup.next
    @inbounds links.prev .= links_backup.prev    
end


@inline function bisection_segment!(PIMC::t_pimc, beads::t_beads, links::t_links, idlist::AbstractVector{Int64})
    """Bisection of path between known beads idlist[1] and idlist[end] for any number of slices"""
    n = length(idlist)
    if n==2
        # no beads to generate, just link'm
        links.next[idlist[1]] = idlist[2]
        links.prev[idlist[2]] = idlist[1]
        return
    end
    if n<2
        error("Can't bisect path without two end points")
    end
    stack = [(1, n)] # known indices in bead list idlist
    β = PIMC.β
    L = PIMC.L
   
    while !isempty(stack)
        (i, j) = pop!(stack)
        if j - i <= 1
            continue  # no points needed between beads idlist[i] and idlist[j]
        end
        m = div(i + j, 2)

        τ_ij = mod(beads.times[idlist[j]] - beads.times[idlist[i]], β)
        τ_im = mod(beads.times[idlist[m]] - beads.times[idlist[i]], β)
        τ_mj = mod(beads.times[idlist[j]] - beads.times[idlist[m]], β) 
        τ = τ_im * τ_mj / τ_ij
        X = @view beads.X[:, idlist[m]]
        @inbounds for d in 1:dim
            xi = beads.X[d, idlist[i]]
            xj = beads.X[d, idlist[j]]
            dx = xj - xi
            if pbc
                dx -= L * round(dx/L)
            end
            # interpolated midpoint
            mean = xi + dx * (τ_im/τ_ij)            
            # without pbc, mean = (τ_mj * xi + τ_im * xj) / τ_ij
            # with pbc, replace xj with xi+dx 
            # (τ_mj * xi + τ_im * xj)/τ_ij ->  (τ_mj * xi + τ_im * (xi+dx))/τ_ij = xi + dx * (τ_im/τ_ij)  
            X[d] = mean + sqrt(2λ*τ) * randn()
            if pbc
                X[d] = mod(X[d] + L/2, L) - L/2
            end
        end
        # push left and right halves to stack 
        push!(stack, (i, m))
        push!(stack, (m, j))
    end
    for i in 1:length(idlist)-1
        links.next[idlist[i]] = idlist[i+1]
        links.prev[idlist[i+1]] = idlist[i]
    end
end


function E_pot_all(PIMC::t_pimc, beads::t_beads)
    """Potential energy of the whole system"""

    Epot = 0.0
    ms = get_ms(PIMC.M)
    for m in ms
        # all active beads on slice m
        bs = active_beads_on_slice(beads, m)
        Xv = @view beads.X[:, bs]        
        if PIMC.confinement_potential !== nothing
            Epot += sum(PIMC.confinement_potential(Xv))            
        end
        if PIMC.pair_potential !== nothing
            V = PIMC.pair_potential
            L = PIMC.L
            #t1 = @elapsed begin            
            nbs = length(bs)               
            if pbc
                # beads on different world lines, use min. image dist for pbc                
                # cutoff at L/2
                dXs = Float64[]  
                @inbounds for i in 1:nbs-1
                    @inbounds for j in i+1:nbs
                        d = dist(view(Xv, :, i), view(Xv, :, j), L)
                        if d ≤ L/2
                            push!(dXs, d)
                        end
                    end
                end
            else
                dXs  = [norm(Xv[:, i]-Xv[:, j]) for i in 1:nbs-1 for j in i+1:nbs]                
            end
            Epot += sum(V.(dXs))
            #=
            #end
            #t2 = @elapsed begin
            # non-vectorized form, almost always slower
            nbs = length(bs)
            Etest = 0
            id_pairs = [(bs[i], bs[j]) for i in 1:nbs-1 for j in i+1:nbs]
            for (id1, id2) in id_pairs        
                r1 = @view beads.X[:, id1]
                r2 = @view beads.X[:, id2]
                # beads on different world lines, use min. image dist for pbc
                r12 = dist(r1, r2, L)
                if r12 ≤ L/2
                    Etest += V(r12)
                end
            end
            Epot += Etest
        
            #@show Etest/N, sum(V.(dXs))# , t1, t2, t1<t2
            =#
            
        end          
    end
    if PIMC.pair_potential !== nothing
        Epot += PIMC_Common.Vtail*N # add tail correction, if any
    end
    return Epot/length(ms) # works for both prim and chin
end

#
# Generic energy computations
# ===========================
#
function E_pot_bead(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Potential energy of bead id"""    
    Epot::Float64 = 0.0 
    # an inactive bead has no potential energy
    beads.active[id]==false && (return Epot)

    r1 = @view beads.X[:, id]
    if PIMC.confinement_potential !== nothing        
        Epot += PIMC.confinement_potential(r1)
    end
    if PIMC.pair_potential !== nothing
        V = PIMC.pair_potential
        L = PIMC.L
        t = beads.ts[id]        
        # active beads on slice t not equal to id
        bs = filter(x -> x ≠ id, active_beads_on_slice(beads, t))
        #
        # vectorized form
        #    nbs = length(bs)
        #    Xv = @view beads.X[:, bs]
        #    # beads on different world lines, use min. image dist for pbc
        #    dXs  = [dist(Xv[:, i], r1, L) for i in 1:nbs]
        #    Epot += sum(Vs(dXs))            
        
        # non-vectorized form is a bit faster
        if pbc
            @inbounds for id2 in bs
                # beads on different world lines, use min. image dist for pbc
                r = dist(r1, view(beads.X, :, id2), L)
                r > L/2 && continue #cutoff
                Epot += V(r)
            end
            Epot += PIMC_Common.Vtail # add tail correction, if any
        else
            @inbounds for id2 in bs                   
                Epot += V(norm(r1 - view(beads.X, :, id2)))
            end
        end
        
    end
    return Epot
end
function boson_virial_exchange_energy(PIMC::t_pimc, beads::t_beads, links::t_links)
    """Boson virial estimator exchange energy term for co-moving centroid reference."""

    if !bose
        error("Boson virial estimator works only for bosons :)")
    end
    M = PIMC.M
    L = PIMC.L
    β = PIMC.β
    τ = PIMC.τ
    
    # exchange energy term
    
    dx_long = zeros(dim)  # pre-allocate 
    dx_short = zeros(dim) 
    r = zeros(dim)
    
    Δτ = 0.0 
    Eexc = 0.0
    dr = zeros(dim)
    for idm in beads.ids[beads.active]
        r_m = @view beads.X[:,idm] # X_m 
        r .= r_m  # displacements are relative to r (updated along the way)
        id = idm
        for m in 1:M
            id = links.next[id]
            distance!(view(beads.X, :, id), view(r,:), L, dr)
            if m==M-1
                # X_M+m - X_M+m-1, short distance vector
                dx_short .= dr
                Δτ = mod(beads.times[id] - beads.times[links.prev[id]], β)        
            end
            r .+= dr
        end
        r_Mm = r # X_M+m
        # long continuous path vector (on same slice), *not* min. image distance         
        dx_long .= r_Mm - r_m  # X_M+m - X_m
        Eexc += dx_long ⋅ dx_short/Δτ  
    end
    Eexc *= -1/(4*λ*M*β)
    Eexc
end

# just for generate_path:
const vec_gene1 = MVector{dim, Float64}(undef)
const vec_gene2 = MVector{dim, Float64}(undef)
const vec_gene3 = MVector{dim, Float64}(undef)
const vec_gene4 = MVector{dim, Float64}(undef)
const vec_gene5 = MVector{dim, Float64}(undef)

function generate_path!(PIMC::t_pimc, beads::t_beads, links::t_links, idlist::AbstractVector{Int64})
    """Generates a free-particle path between known beads idlist[1] and idlist[end] using staging; updates links."""
    #
    # Generated beads are not activated! 
    #
    M = PIMC.M
    τ = PIMC.τ
    β = PIMC.β
    L = PIMC.L
    #
    start_bead = idlist[1]
    end_bead   = idlist[end]
    t_start = beads.ts[start_bead]
    t_end   = beads.ts[end_bead]

    K = length(idlist)
    
    if K==2
        #      1        K=2
        # start_bead  end_bead
        # No path to generate, just close the link
        links.next[start_bead] = end_bead
        links.prev[end_bead]   = start_bead
        return nothing
    end

    #      1      2 3 ...    K-1    K
    # start_bead   new beads      end_bead
    # Generate new beads using staging

    r_left = vec_gene5
    @inbounds @views r_left .= beads.X[:, start_bead] # r_left changes in iteration, beads.X[:, start_bead] is not changed
    r_end = @view beads.X[:, end_bead]   # fixed
    t_end = beads.times[end_bead]        # fixed

    r_m = vec_gene1
    r_m_star = vec_gene2
    r_right = vec_gene3    
    dr = vec_gene4
    
    id_left = start_bead
    @inbounds begin
        for m in 2:K-1        
            id = idlist[m] 
            #        
            τ_left = mod(beads.times[id] -  beads.times[id_left], β)
            τ_right = mod(t_end -  beads.times[id], β)
            τ = 1/(1/τ_left + 1/τ_right) 
            σ = sqrt(2λ*τ)
            #
            distance!(r_end, view(r_left, :), L, dr) # dr = r_end - r_left periodically
            
            for d in 1:dim
                r_right[d] = r_left[d] + dr[d]
                r_m_star[d] = (τ_right*r_left[d] + τ_left*r_right[d])/(τ_left + τ_right)         
                # new bead position        
                r_m[d] = r_m_star[d] +  σ * randn()
                if pbc 
                    r_m[d] = mod(r_m[d] + L/2, L) - L/2
                end            
                beads.X[d, id] = r_m[d]
            end
            # link previous bead to new bead
            links.next[id_left] = id
            links.prev[id] = id_left
            # assign r_m as the known bead r_left; careful not to set them same forever!
            id_left = id
            @views r_left .= r_m        
        end
    end
    
    # link last generated bead to end_bead
    id = idlist[K-1]
    links.next[id] = end_bead
    links.prev[end_bead] = id
    return nothing
end


@inline function rho_0(r1::SubArray{Float64}, r2::SubArray{Float64}, λ::Float64, τ12::Float64, L::Float64)    
    """Free-particle density matrix ⟨r1 | e^(-τ T) | r2⟩ for one particle"""
    Δr2 = dist2(r1, r2, L)    
    return (4π*λ*τ12)^(-dim/2) * exp(-Δr2 /(4*λ*τ12))  
end


function distance_check(PIMC::t_pimc, beads::t_beads, links::t_links)
    for id in beads.active
        next = links.next[id]
        if next>0
            Δr = dist(beads.X[:, id], beads.X[:, next], PIMC.L)
            r  = norm(beads.X[:, id] - beads.X[:, next])
            if !isapprox(Δr, r)
                @show id, next, Δr, r
                error("min. image and euclidian distances are not the same for bead and next bead")
            end
        end
    end
end




function save_paths(PIMC::t_pimc, beads::t_beads, links::t_links, col::Int64=1)

    L = PIMC.L
    
    mode = col==1 ? "w" : "a"
    open("paths", mode) do f
        # full world lines x(τ) in 1D, just coordinates in 2D and 3D
        for id in beads.active
            #if dim==1
                println(f,  beads.X[1, id], " ", beads.ts[id], " ", col)
            #else
            #    println(f,  beads.X[1, id], " ", beads.X[2, id], " ",  col)
            #end
            
            id_next = links.next[id]            
            if id_next > 0
                if beads.ts[id_next]==1
                    println(f," ")
                end
                #if dim==1
                    println(f,  beads.X[1, id_next], " ", beads.ts[id_next], " ",  col)
                #else
                #    println(f,  beads.X[1, id_next], " ",  beads.X[2, id_next], " ",  col)
                #end
            end
            println(f," ")            
        end
    end
    #if col==2
    #    println("saved paths (press enter if stopped)")
    #    readline()
    #end
end


end
