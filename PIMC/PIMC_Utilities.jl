__precompile__(false)
module PIMC_Utilities

using LinearAlgebra
using StaticArrays


push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Structs:t_beads, t_links, t_pimc, has_potential

export store_X!, restore_X!
export rho_0, bisection_path_unfolded!
export generate_path!, generate_path_unfolded! 
export backup_state!, restore_state!
export boson_virial_exchange_energy
export E_pot, E_pot_all, E_pot_bead
export save_paths
export argparse
export pull!, activate_beads!, deactivate_beads!
export loop_counter, centroid_ref!

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
                        value = parse(Int,value)
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

@inline function pull!(v::Vector{Int}, x::Int)
    @inbounds begin
        for i in eachindex(v)
            if v[i] == x
                v[i] = v[end]
                pop!(v)
                return true
            end
        end
    end
    return false   # not found
end


#@inline function active_beads_on_slice(beads::t_beads, m::Int)
#    error("old, don't use")
#    idxs = beads.at_t[m]
#    mask = @view beads.active[idxs]
#    return @view idxs[mask]
#end


@inline function activate_beads!(beads::t_beads, idlist::AbstractVector{Int})
    @inbounds begin
        @views beads.active[idlist] .= true
        for b in idlist
            m = beads.ts[b]
            ok = pull!(beads.inactive_at_t[m], b)
            #@assert ok "pull! failed in activate_beads!"            
            push!(beads.active_at_t[m], b)
        end
    end
end

@inline function deactivate_beads!(beads::t_beads, idlist::AbstractVector{Int})
    @inbounds begin
        @views beads.active[idlist] .= false
        for b in idlist
            m = beads.ts[b]
            ok = pull!(beads.active_at_t[m], b)
            #@assert ok "pull! failed in deactivate_beads!"
            push!(beads.inactive_at_t[m], b)
        end
    end
end

    
function loop_counter(beads::t_beads, links::t_links)::Int
    """counts number of loops"""
    bs = beads.active_at_t[1]
    done = falses(maximum(bs))
    
    Nloops::Int = 0
    @inbounds for id1 in bs
        done[id1] && continue
        done[id1] = true
        id = id1
        while true
            id = links.next[id]
            id == id1 && break
            if beads.ts[id]==1
                done[id] = true
            end           
        end
        Nloops += 1
    end
    return Nloops
end

# Temporary storage for a few bead positions 

mutable struct t_storage
    X::Matrix{Float64} 
    k::Int  
end

const storage_ref = Ref(t_storage(Matrix{Float64}(undef, dim, N_slice_max*M_max), 0))

function get_storage()    
    return storage_ref[]
end

# single bead
function store_X!(beads::t_beads, id::Int,
                  storage::t_storage = get_storage()    
                  ) ::t_storage
    storage.k = 1           # set number of stored elements
    @inbounds for d = 1:dim
        storage.X[d, 1] = beads.X[d, id]
    end
    return storage
end

# two beads
function store_X!(beads::t_beads, id1::Int, id2::Int,
                  storage::t_storage = get_storage()    
                  ) ::t_storage
    storage.k = 2           # set number of stored elements
    @inbounds for d = 1:dim
        storage.X[d, 1] = beads.X[d, id1]
        storage.X[d, 2] = beads.X[d, id2]
    end
    return storage
end


# vector of beads
function store_X!(beads::t_beads, idlist::AbstractVector{Int},
                  storage::t_storage = get_storage()    
                  ) ::t_storage
    k = length(idlist)
    k > size(storage.X, 2) && error("Storage too small")
    storage.k = k           
    @inbounds for i = 1:k, d = 1:dim
        storage.X[d, i] = beads.X[d, idlist[i]]
    end
    return storage
end

# Restore values if move is rejected
# single bead 
function restore_X!(beads::t_beads, id::Int,
                    storage::t_storage = get_storage()    
                    )
    #@assert storage.k==1 "restore_X! for a single bead failed"   
    @inbounds for d = 1:dim
        beads.X[d, id] = storage.X[d, 1]
    end
end

# two beads
function restore_X!(beads::t_beads, id1::Int, id2::Int,
                    storage::t_storage = get_storage()    
                    )
    #@assert storage.k==2 "restore_X! for a two beads failed"   
    @inbounds for d = 1:dim
        beads.X[d, id1] = storage.X[d, 1]
        beads.X[d, id2] = storage.X[d, 2]
    end
end

# vector of beads
function restore_X!(beads::t_beads,  idlist::AbstractVector{Int},
                    storage::t_storage = get_storage()    
                    )
    k = storage.k   
    @inbounds for i = 1:k, d = 1:dim
        beads.X[d, idlist[i]] = storage.X[d, i]
    end
end



function backup_state!(beads_backup::t_beads, links_backup::t_links, 
                       beads::t_beads, links::t_links)
    # these are already fixed: beads.ids, beads.ts, beads.at_t, beads.times
    @inbounds @views begin
        beads_backup.X .= beads.X
        beads_backup.active .= beads.active       
        links_backup.next .= links.next
        links_backup.prev .= links.prev
        for m in eachindex(beads.at_t)
            empty!(beads_backup.active_at_t[m])
            append!(beads_backup.active_at_t[m], beads.active_at_t[m])
            empty!(beads_backup.inactive_at_t[m])
            append!(beads_backup.inactive_at_t[m], beads.inactive_at_t[m])
        end
        
    end
end

function restore_state!(beads::t_beads, links::t_links, 
                        beads_backup::t_beads, links_backup::t_links)
    
    @inbounds @views  begin
        beads.X .= beads_backup.X
        beads.active .= beads_backup.active
        links.next .= links_backup.next
        links.prev .= links_backup.prev
        for m in eachindex(beads.at_t)
            beads.active_at_t[m] = Int[]
            for b in beads_backup.active_at_t[m]
                push!(beads.active_at_t[m], b)
            end
            beads.inactive_at_t[m] = Int[]
            for b in beads_backup.inactive_at_t[m]
                push!(beads.inactive_at_t[m], b)
            end
        end
    end
end

# NB: No folding to PBC positions
@inline function bisection_path_unfolded!(PIMC::t_pimc, beads::t_beads, links::t_links, idlist::AbstractVector{Int})
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
   
    while !isempty(stack)
        (i, j) = pop!(stack)
        if j - i <= 1
            continue  # no points needed between beads idlist[i] and idlist[j]
        end
        m = div(i + j, 2)
        τ_ij = wrapβ(beads.times[idlist[j]] - beads.times[idlist[i]], β)
        τ_im = wrapβ(beads.times[idlist[m]] - beads.times[idlist[i]], β)
        τ_mj = wrapβ(beads.times[idlist[j]] - beads.times[idlist[m]], β)
        #@assert isapprox(τ_im + τ_mj, τ_ij) "bisection_path_unfolded! time error"
        τ = τ_im * τ_mj / τ_ij
        σ = sqrt(2λ*τ)
        X = @view beads.X[:, idlist[m]]        
        @inbounds for d in 1:dim
            xi = beads.X[d, idlist[i]]
            xj = beads.X[d, idlist[j]]
            # interpolated midpoint
            xmid = (τ_mj * xi + τ_im * xj) / τ_ij
            X[d] = xmid +  σ * randn()            
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

# Compile-time dispatch
@inline function get_ms(::PrimitiveAction, M::Int)
    return 1:M
end
@inline function get_ms(::ChinAction, M::Int)
    return 2:3:M
end
@inline function get_ms(M::Int)
    A = PIMC_Common.action
    return get_ms(A(), M)
end


function E_pot_all(PIMC::t_pimc, beads::t_beads)
    """Potential energy of the whole system"""

    rc = PIMC.rc
    Epot = 0.0
    ms = get_ms(PIMC.M)
    for m in ms
        # all active beads on slice m
        bs = beads.active_at_t[m]
        Xv = @view beads.X[:, bs]        
        if has_potential(PIMC.syspots.confinement_potential)
            Epot += sum(PIMC.syspots.confinement_potential(Xv))            
        end
        if has_potential(PIMC.syspots.pair_potential)
            V = PIMC.syspots.pair_potential
            L = PIMC.L
            #t1 = @elapsed begin            
            nbs = length(bs)               
            # beads on different world lines, use min. image dist for pbc                
            # cutoff at rc (may be Inf)
            dXs = Float64[]  
            @inbounds for i in 1:nbs-1
                @inbounds for j in i+1:nbs
                    d = dist(view(Xv, :, i), view(Xv, :, j), L)
                    if d ≤ rc
                        push!(dXs, d)
                    end
                end
            end
            Epot += sum(V.(dXs))
        end          
    end
    if has_potential(PIMC.syspots.pair_potential)
        Epot += PIMC_Common.Vtail*N # add tail correction, if any
    end
    return Epot/length(ms) # works for both prim and chin
end


# Co-moving centroid and loop centroid references for virial estimators
# =====================================================================
# rref storage 
mutable struct t_rref_status
    ipimc::Int
    iworm::Int    
end
const rref_status = t_rref_status(0, 0) 
const rref_buffer = Matrix{Float64}(undef, dim, N_slice_max*M_max) # overkill 


const dr_cr_buffer = MVector{dim, Float64}(undef)
const rup_cr_buffer = MVector{dim, Float64}(undef)
const rdo_cr_buffer = MVector{dim, Float64}(undef)



# Co-moving centroid reference for all active beads
# NB: updates rref_buffer 
function centroid_ref!(PIMC::t_pimc, beads::t_beads, links::t_links,                       
                       centroid_M::Int = 0, 
                       dr::MVector{dim, Float64} =  dr_cr_buffer,
                       rup::MVector{dim, Float64} = rup_cr_buffer,
                       rdo::MVector{dim, Float64} = rdo_cr_buffer,
                       rref::Matrix{Float64} = rref_buffer,
                       rref_status::t_rref_status = rref_status
                       ) ::Matrix{Float64} 
    """ co-moving centroid reference using centroid_M (often M) beads up and down"""
        
    if PIMC.ipimc == rref_status.ipimc && PIMC.iworm == rref_status.iworm
        return rref
    end
    
    M = PIMC.M
    L = PIMC.L
    # Choose either centroid reference or full-loop centroid reference   
    if !PIMC_Common.use_loop_centroid
        # pick centroid window from argument or fix it 
        centroid_M::Int = centroid_M==0 ? M : centroid_M
        fac::Float64 = 1/(2*centroid_M)
        # take fixed window (centroid_M) average
        @inbounds @views begin
            for m in 1:M
                bs = beads.active_at_t[m]
                rref[:, bs] .= 2*beads.X[:, bs]
                for b in bs
                    # steps up and down from bead b
                    bup = b 
                    bdo = b 
                    # bead X[:, b] defines the continuous path
                    rup .= beads.X[:, b] # path up
                    rdo .= rup           # path down
                    for _ in 1:centroid_M-1
                        bup = links.next[bup]
                        bdo = links.prev[bdo]
                        distance!(beads.X[:, bup], rup, L, dr)
                        rup .+=  dr
                        rref[:, b] .+= rup
                        distance!(beads.X[:, bdo], rdo, L, dr)
                        rdo .+= dr
                        rref[:, b] .+= rdo
                    end
                end
            end
            rref .*= fac
        end # inbounds
    else
        # full-loop centroid
        done = falses(maximum(beads.ids)) 
        @inbounds @views begin
            bs = beads.active_at_t[1]
            for b in bs
                done[b] && continue                 
                # steps up from bead b until loop closes
                bup = b 
                # bead X[:, b] defines the continuous path
                rup .= beads.X[:, b]
                lp = 1 # counts beads in this loop := beads added to rup
                while true
                    bup = links.next[bup]
                    bup == b && break # loop closes
                    distance!(beads.X[:, bup], rup, L, dr)
                    rup .+=  dr
                    lp += 1
                    done[bup] = true # mark done
                end
                rup ./= lp
                rref[:, b] .= rup
                # set same rref (rup) to all beads in this loop
                bup = b
                while true
                    bup = links.next[bup]
                    bup == b && break
                    rref[:, bup] .= rup
                end
            end
        end # inbounds
    end
    
    rref_status.ipimc = PIMC.ipimc
    rref_status.iworm = PIMC.iworm
    return rref
end

#
# Generic energy computations
# ===========================
#
function E_pot_bead(PIMC::t_pimc, beads::t_beads, id::Int)
    """Potential energy of bead id"""

    rc = PIMC.rc
    Epot::Float64 = 0.0 
    # an inactive bead has no potential energy
    beads.active[id]==false && (return Epot)

    r1 = @view beads.X[:, id]
    if has_potential(PIMC.syspots.confinement_potential)
        Epot += PIMC.syspots.confinement_potential(r1)
    end
    if has_potential(PIMC.syspots.pair_potential)
        V = PIMC.syspots.pair_potential
        L = PIMC.L
        t = beads.ts[id]        
        # active beads on slice t not equal to id
        bs = filter(x -> x ≠ id, beads.active_at_t[t])
        # non-vectorized form is a bit faster
        @inbounds for id2 in bs
            # beads on different world lines, use min. image dist for pbc
            r = dist(r1, view(beads.X, :, id2), L)
            if r < rc  # rc can be Inf
                Epot += V(r)
            end
        end
        Epot += PIMC_Common.Vtail # add tail correction, if any
    end
    return Epot
end

const dr_buffer = MVector{dim, Float64}(undef)
const r_buffer = MVector{dim, Float64}(undef)



# Exchange energy estimator storage
mutable struct t_Eexc_store
    ipimc::Int
    iworm::Int
    Eexc::Float64
end

const Eexc_store_ref = Ref{Union{Nothing, t_Eexc_store}}(nothing)

function get_Eexc_stored()
    # lazy init stored
    if Eexc_store_ref[] === nothing 
        Eexc_store_ref[] = t_Eexc_store(0, 0, 0.0)
    end    
    return Eexc_store_ref[]
end

# for future use
@inline function get_Eexc_bead(PIMC::t_pimc, beads::t_beads, links::t_links,
                               r::MVector{dim, Float64}, dr::MVector{dim, Float64},                                               
                               idm::Int):: Float64
    
    M = PIMC.M
    L = PIMC.L
    β = PIMC.β
    id = idm
    r_m = @view beads.X[:, idm] # X_m 
    r .= r_m   # displacements are relative to r (updated along the way)
    Eexc::Float64 = 0.0
    @inbounds begin
        for m in 1:M
            id = links.next[id]
            distance!(view(beads.X, :, id), r, L, dr)                
            r .+= dr
        end
        # now dr is X_M+m - X_M+m-1, short distance vector
        Δτ::Float64 = wrapβ(beads.times[id] - beads.times[links.prev[id]], β)        
        #
        #  r is X_M+m
        # long continuous path vector (on same slice), *not* min. image distance         
        Eexc += dot(r - r_m, dr)/Δτ
    end
    Eexc *= -1/(4*λ*M*β)
    return Eexc
end

function boson_virial_exchange_energy(PIMC::t_pimc, beads::t_beads, links::t_links,
                                      r::MVector{dim, Float64} = r_buffer,
                                      dr::MVector{dim, Float64} = dr_buffer,                                      
                                      Eexc_store::t_Eexc_store = get_Eexc_stored()
                                      ) ::Float64
    """Boson exchange energy for centroid reference"""

    if !bose
        error("Boson virial estimator works only for bosons :)")
    end
    # see if stored value can be used
    if PIMC.ipimc == Eexc_store.ipimc && PIMC.iworm == Eexc_store.iworm
        return Eexc_store.Eexc
    end

    
    M = PIMC.M
    L = PIMC.L
    β = PIMC.β
    τ = PIMC.τ
    
    exchange_beads = Set{Int}()
    @inbounds begin
        for idm in beads.ids[beads.active]
            id = idm
            for m in 1:M
                id = links.next[id]
            end
            if id != idm
                id = idm
                for m in 1:M
                    push!(exchange_beads, id)
                    id = links.next[id]
                end
            end
        end
    end
    exchange_beads = collect(exchange_beads)
    Eexc::Float64 = 0.0
    @inbounds begin
        for idm in exchange_beads            
            r_m = @view beads.X[:, idm] # X_m 
            r .= r_m  # displacements are relative to r (updated along the way)
            id = idm
            for m in 1:M
                id = links.next[id]
                distance!(view(beads.X, :, id), r, L, dr)                
                r .+= dr
            end
            # the last dr is X_M+m - X_M+m-1, short distance vector, the corresponding Δτ is 
            Δτ = wrapβ(beads.times[id] - beads.times[links.prev[id]], β)        
            #
            # long continuous path vector (on same slice), *not* min. image distance         
            # r - r_m  is  X_M+m - X_m
            Eexc += (r - r_m) ⋅ dr/Δτ
        end
    end# inbounds
    Eexc *= -1/(4*λ*M*β)
      
    # update stored value
    Eexc_store.ipimc = PIMC.ipimc
    Eexc_store.iworm = PIMC.iworm
    Eexc_store.Eexc = Eexc
    return Eexc
end



# just for generate_path:
const vec_gene1 = MVector{dim, Float64}(undef)
const vec_gene2 = MVector{dim, Float64}(undef)
const vec_gene3 = MVector{dim, Float64}(undef)
const vec_gene4 = MVector{dim, Float64}(undef)
const vec_gene5 = MVector{dim, Float64}(undef)

function generate_path!(PIMC::t_pimc, beads::t_beads, links::t_links, idlist::AbstractVector{Int},
                        r_m::MVector{dim, Float64} = vec_gene1,
                        r_left::MVector{dim, Float64} = vec_gene2
                        )
    """Generates a free-particle path between known beads idlist[1] and idlist[end] using staging; updates links."""
    #
    # NB: Generated beads are not activated!
    # Returns a path idlist[2:end-1] folded to box [-L/2, L/2] 
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

    
    @inbounds @views r_left .= beads.X[:, start_bead] # r_left changes in iteration, beads.X[:, start_bead] is not changed
    r_end = SVector{dim, Float64}(beads.X[:, end_bead])   # fixed
    t_end = beads.times[end_bead]        # fixed

    id_left = start_bead
    @inbounds begin
        for m in 2:K-1        
            id = idlist[m] 
            #        
            Δτ_left = wrapβ(beads.times[id] -  beads.times[id_left], β)
            Δτ_right = wrapβ(t_end -  beads.times[id], β)
            Δτ_left_right = Δτ_left + Δτ_right
            Δτ = 1/(1/Δτ_left + 1/Δτ_right) 
            σ = sqrt(2λ*Δτ)
            #
            ta = Δτ_right/Δτ_left_right
            tb = Δτ_left/Δτ_left_right 
            if pbc
                # ok to continue from folded coordinate x,
                # because dr always points toward the minimum image of r_end                
                for d in 1:dim
                    dr = r_end[d] - r_left[d]
                    dr -= L * round(dr/L)
                    r_m_star = ta*r_left[d] + tb*(r_left[d] + dr)
                    # new bead position        
                    x  = r_m_star +  σ * randn()
                    x -= L * floor((x + L/2) / L)
                    r_m[d] = x
                    beads.X[d, id] = x
                end
            else
                for d in 1:dim
                    dr = r_end[d] - r_left[d]
                    r_m_star = ta*r_left[d] + tb*(r_left[d] + dr)
                    # new bead position        
                    x  = r_m_star +  σ * randn()
                    r_m[d] = x
                    beads.X[d, id] = x
                end
            end
            # link previous bead to new bead
            links.next[id_left] = id
            links.prev[id] = id_left
            # assign r_m as the known bead r_left; careful not to set them same forever!
            id_left = id
            for d in 1:dim
                r_left[d] = r_m[d]
            end
        end
    end    
    # link last generated bead to end_bead
    id = idlist[K-1]
    links.next[id] = end_bead
    links.prev[end_bead] = id
    return nothing
end

@inline function rho_0(X::Matrix{Float64}, i::Int, j::Int, λ::Float64, τ12::Float64, L::Float64)    
    """Free-particle density matrix ⟨r_i | e^(-τ T) | r_j⟩ for one particle"""
    Δr2 = dist2(X, i, j, L)    
    return (4π*λ*τ12)^(-dim/2) * exp(-Δr2 /(4λ*τ12))  
end

@inline function rho_0(r1::AbstractArray{Float64}, r2::AbstractArray{Float64}, λ::Float64, τ12::Float64, L::Float64)    
    """Free-particle density matrix ⟨r1 | e^(-τ T) | r2⟩ for one particle"""
    Δr2 = dist2(r1, r2, L)    
    return (4π*λ*τ12)^(-dim/2) * exp(-Δr2 /(4λ*τ12))  
end



# generate path using staging *without* pbc folding 
function generate_path_unfolded!(PIMC::t_pimc, beads::t_beads, links::t_links, idlist::AbstractVector{Int},
                        r_m::MVector{dim, Float64} = vec_gene1,
                        r_m_star::MVector{dim, Float64} = vec_gene2,
                        r_right::MVector{dim, Float64} = vec_gene3,    
                        dr::MVector{dim, Float64} = vec_gene4,
                        r_left::MVector{dim, Float64} = vec_gene5
                        )
    """Generates a free-particle path between known beads idlist[1] and idlist[end] using staging; updates links."""
    #
    # NB: Generated beads are not activated and the bead positions are not pbc-folded  

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

    
    @inbounds @views r_left .= beads.X[:, start_bead] # r_left changes in iteration, beads.X[:, start_bead] is not changed
    r_end = @view beads.X[:, end_bead]   # fixed
    t_end = beads.times[end_bead]        # fixed
    id_left = start_bead
    @inbounds begin
        for m in 2:K-1        
            id = idlist[m] 
            #        
            τ_left = wrapβ(beads.times[id] -  beads.times[id_left], β)
            τ_end = wrapβ(t_end -  beads.times[id], β)
            τ = 1/(1/τ_left + 1/τ_end) 
            σ = sqrt(2λ*τ)
            #
            for d in 1:dim
                r_m_star[d] = (τ_end*r_left[d] + τ_left*r_end[d])/(τ_left + τ_end)         
                # new bead position        
                r_m[d] = r_m_star[d] +  σ * randn()                     
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


function save_paths(PIMC::t_pimc, beads::t_beads, links::t_links, col::Int=1)

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
