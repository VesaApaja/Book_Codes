__precompile__(false)
module PIMC_Moves

using BenchmarkTools
using Random, LinearAlgebra
BLAS.set_num_threads(1) 
using Distributions: Categorical, Geometric, Exponential, Truncated
using Printf
using InteractiveUtils # for @code_warntype profiling
using StaticArrays

# local modules:
push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Structs
using PIMC_Utilities
using QMC_Statistics

using PIMC_Primitive_Action: U as U_prim, K as K_prim, U_update as U_update_prim, U_stored as U_stored_prim
using PIMC_Primitive_Action:update_stored_slice_data as update_stored_slice_data_prim
using PIMC_Primitive_Action: init_stored as init_stored_prim

using PIMC_Chin_Action: update_stored_slice_data as update_stored_slice_data_chin, U as U_chin, K as K_chin
using PIMC_Chin_Action: U_update as U_update_chin, U_stored as U_stored_chin
using PIMC_Chin_Action: init_stored as init_stored_chin
using PIMC_Chin_Action: opt_chin_a1_chin_t0

using PIMC_Systems
using PIMC_Measurements: meas_superfluid_fraction, meas_cycle_count

export add_move!  
export bead_move!, rigid_move!, bisection_move!, worm_move!
export init_stored


using PIMC_Chin_Action: meas_E_vir #TEST
    
# compile-time dispatch:
# ======================

@inline function init_stored(::PrimitiveAction, PIMC::t_pimc, beads::t_beads)
    return init_stored_prim(PIMC, beads)
end
@inline function init_stored(::ChinAction, PIMC::t_pimc, beads::t_beads)
    return init_stored_chin(PIMC, beads)
end
@inline function init_stored(PIMC::t_pimc, beads::t_beads)
    """user interface"""
    A = PIMC_Common.action
    return init_stored(A(), PIMC, beads)
end

# -------------------
# Light-weight buffers, *not* thread-safe
const vec_buffer  = MVector{dim, Float64}(undef)
const vec_buffer2 = MVector{dim, Float64}(undef)
const vec_buffer3 = MVector{dim, Float64}(undef)

# just for generate_path:
const vec_gene1 = MVector{dim, Float64}(undef)
const vec_gene2 = MVector{dim, Float64}(undef)
const vec_gene3 = MVector{dim, Float64}(undef)
const vec_gene4 = MVector{dim, Float64}(undef)
const vec_gene5 = MVector{dim, Float64}(undef)

const idlist_buffer = Vector{Int64}(undef, 1000) # should be enough




function add_move!(PIMC::t_pimc, frequency::Int64, name::Symbol, exe::Function)
    move = t_move(frequency=frequency, name=name, exe=exe, ntries=0, nacc=0)
    push!(PIMC.moves, move)
    println("Added move: ",name,",  frequency ",frequency)
end



@inline function metro(ΔS::Float64)
    """Metropolis question for action change ΔS"""
    accept = false
    if -ΔS > 0.0  # means exp(-ΔS)>1
        accept = true        
    else
        if exp(-ΔS) > rand()
            accept = true
        end
    end
    accept
end


# compile-time dynamical dispatch:
# ================================

@inline function U(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return U_prim(PIMC, beads, id)
end
@inline function U(::ChinAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return U_chin(PIMC, beads)
end
@inline function U(PIMC::t_pimc, beads::t_beads)
    """user interface"""
    A = PIMC_Common.action
    return U(A(), PIMC, beads)
end

@inline function U_stored(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, id::Int64)    
    return U_stored_prim(PIMC, beads, id)
end
@inline function U_stored(::ChinAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return U_stored_chin(PIMC, beads, id)
end
@inline function U_stored(PIMC::t_pimc, beads::t_beads, id::Int64)
    """user interface"""
    A = PIMC_Common.action
    return U_stored(A(), PIMC, beads, id)
end

@inline function U_update(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, Xold::AbstractArray{Float64}, id::Int64, act::Symbol; fake::Bool)    
    return U_update_prim(PIMC, beads, Xold, id ,act; fake=fake)
end
@inline function U_update(::ChinAction, PIMC::t_pimc, beads::t_beads, Xold::AbstractArray{Float64}, id::Int64, act::Symbol; fake::Bool)   
    return U_update_chin(PIMC, beads, Xold, id, act; fake=fake)
end
@inline function U_update(PIMC::t_pimc, beads::t_beads, Xold::AbstractArray{Float64}, id::Int64, act::Symbol; fake::Bool=false)
    """user interface"""
    A = PIMC_Common.action
    return U_update(A(), PIMC, beads, Xold, id, act; fake)
end

@inline function K(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    return K_prim(PIMC, beads, links, id)
end
@inline function K(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    return K_chin(PIMC, beads, links, id)
end
@inline function K(PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    """user interface"""
    A = PIMC_Common.action
    return K(A(), PIMC, beads, links, id)
end

@inline function update_stored_slice_data(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return update_stored_slice_data_prim(PIMC, beads, id)
end
@inline function update_stored_slice_data(::ChinAction, PIMC::t_pimc, beads::t_beads, id::Int64)
    return update_stored_slice_data_chin(PIMC, beads, id)
end
@inline function update_stored_slice_data(PIMC::t_pimc, beads::t_beads, id::Int64)
    """user interface"""
    A = PIMC_Common.action
    return update_stored_slice_data(A(), PIMC, beads, id)
end
# ==================================

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

# =======
# Moves 
# =======


mutable struct Bead_Move_Params
    ntry::Int64
    nacc::Int64
    step::Float64
end
const DEFAULT_BEAD_MOVE_PAR = Bead_Move_Params(0,0,0.2)

function bead_move!(PIMC::t_pimc, beads::t_beads, links::t_links, par::Bead_Move_Params=DEFAULT_BEAD_MOVE_PAR)
    """Moves a single bead"""
   
        
    M = PIMC.M
    L = PIMC.L
    # random bead    
    id = rand(beads.ids[beads.active])
        
    xold = copy(beads.X[:, id])

    #Uold = U(PIMC, beads, id)    
    Uold = U_stored(PIMC, beads, id)    
    Kold = K(PIMC, beads, links, id)
    Sold = Uold + Kold
    
   
    
    beads.X[:, id] .+= par.step .* (rand(dim) .- 0.5)
    pbc && periodic!(view(beads.X, :, id), L)
            
    Unew = U_update(PIMC, beads, view(xold,:), id, :move)    
    Knew = K(PIMC, beads, links, id)
    Snew = Unew + Knew
    
  
    ΔS = Snew - Sold
    # Metropolis    
    accept = metro(ΔS)
    par.ntry += 1
    if accept
        par.nacc += 1
        update_stored_slice_data(PIMC, beads, id)
    else
        beads.X[:, id] .= xold
    end
   
    # adjust step in thermalization
    if PIMC.ipimc<0 && par.ntry%100 == 0
        accep = par.nacc/par.ntry
        if accep>0.6
            par.step *= 1.1
        end
        if accep<0.4
            par.step *= 0.9
        end
        # step ∈ [0.1, 5.0]
        par.step = clamp(par.step, 0.1, 5.0)
        #@printf("bead move acceptance %8.3f %% step %8.3f\n", accep*100.0, par.step)
    end
    return 100*par.nacc/par.ntry
    
    
end

mutable struct Rigid_Move_Params
    ntry::Int64
    nacc::Int64
    step::Float64
end
const DEFAULT_RIGID_MOVE_PAR = Rigid_Move_Params(0, 0, 0.2)

function rigid_move!(PIMC::t_pimc, beads::t_beads, links::t_links,
                     gauss::MVector{dim, Float64}=vec_buffer,
                     shift::MVector{dim, Float64}=vec_buffer2,
                     idlist_buffer::Vector{Int64}=idlist_buffer,
                     par::Rigid_Move_Params=DEFAULT_RIGID_MOVE_PAR)
    """Moves a whole loop, but not exchange loops"""
    par.ntry += 1    
    L = PIMC.L
    M = PIMC.M
    # start from random bead
    id1 = rand(beads.ids[beads.active])
    idlist = @view idlist_buffer[1:M] # in this version only M beads move
    id = id1
    @inbounds for i in 1:M
        idlist[i] = id
        id = links.next[id]            
        if id<0
            # not for worm
            return 100*par.nacc/par.ntry
        end
    end
    if id != id1
        # give up on exchange loops (U_update gets too messy)
        return 100*par.nacc/par.ntry
    end

       # same shift for all
    randn!(gauss)
    shift .= par.step.*gauss
    # move a whole loop
    Uold = 0.0 
    
    @inbounds for id in idlist
        Uold += U_stored(PIMC, beads, id)    #U(PIMC, beads, id)
    end
    xold = copy(beads.X[:, idlist])   
    #
    # In an exchange loop move there can be more than one bead moving on a single slice
    # The value of beads.X[:, id1] has already changed when the multi-M loop
    # moves another bead on the same slice as id1. Gave up earlier on exchange loops.
    #
    Unew = 0.0 
    @inbounds for  (i, id) in enumerate(idlist)
        @inbounds for d in 1:dim
            beads.X[d, id] += shift[d]     # safe to move here, no exchange loops moved
            pbc && (beads.X[d, id] = mod(beads.X[d, id] + L/2, L) - L/2)
        end
        Unew += U_update(PIMC, beads, view(xold,:,i), id, :move)
    end
    ΔS = Unew - Uold # no change in kinetic action
    # Metropolis
    accept = metro(ΔS)

    
    if accept
        par.nacc += 1
        @inbounds for id in idlist
            update_stored_slice_data(PIMC, beads, id)
        end
    else
        beads.X[:, idlist] .= xold
    end

    # adjust step in thermalization
    if PIMC.ipimc<0 && par.ntry%100 == 0
        accep = par.nacc/par.ntry
        if accep>0.6
            par.step *= 1.1
        end
        if accep<0.4
            par.step*= 0.9
        end
        par.step = clamp(par.step, 0.1, 5.0)
        #@printf("rigid move acceptance %8.3f %% step %8.3f\n", accep*100.0, par.step)
    end
    return 100*par.nacc/par.ntry
    
end


mutable struct Bisection_Move_Params
    ntry::Int64
    nacc::Int64
    Nlevels::Int64
end
# for 3 levels:
# fixed end points 1, 9
#           1 2 3 4 5 6 7 8 9
# level 1           5
# level 2       3       7
# level 3     2   4   6   8
#

# start with 3 levels, with pair interaction acceptance decreases with more levels. 
const DEFAULT_BISECTION_MOVE_PAR = Bisection_Move_Params(0, 0, 4)

function bisection_move!(PIMC::t_pimc, beads::t_beads, links::t_links,
                         dr::MVector{dim, Float64}=vec_buffer,
                         xold::MVector{dim, Float64}=vec_buffer2,
                         idlist_buffer::Vector{Int64}=idlist_buffer,
                         par::Bisection_Move_Params=DEFAULT_BISECTION_MOVE_PAR)
    """Bisection move of a path segment"""
    M = PIMC.M
    L = PIMC.L  
    
    par.ntry+=1
    while M<2^par.Nlevels 
        # too small M for default Nlevels bisections
        par.Nlevels -= 1        
        par.Nlevels == 0 &&  error("Nlevels = 0, probably bad M=$M")        
    end

    K = 2^par.Nlevels 
    β = PIMC.β
    
    # keep 1 and K+1 fixed, generate points 2:K
    idlist = @view idlist_buffer[1:K+1]    
    # random bead 1
    id_beg = rand(beads.ids[beads.active])
    # store path   
    id = id_beg
    @inbounds for k in 1:K+1  # list also end point
        idlist[k] = id        
        id = links.next[id]
        if id < 0
            # worm head, give up
            return 100*par.nacc/par.ntry
        end
    end
    # store midpoints
    store_X!(storage, beads, idlist[2:end-1])     
    accept = true
    coun = 1
    @inbounds for lev in 1:par.Nlevels
        # for evenly spaced time slices:
        #τ_lev = K/(2^(lev+1)) * τ 
        #σ_lev = sqrt(2*λ*τ_lev)       
        Uold = 0.0
        Unew = 0.0
        np = 2^(par.Nlevels-lev)
        t_mid = np + 1
        @inbounds for k in 1:coun
            t_prev = t_mid - np
            t_next = t_mid + np            
                        
            prev = idlist[t_prev]
            next = idlist[t_next]
            r_prev = @view beads.X[:, prev]
            r_next = @view beads.X[:, next]
            id = idlist[t_mid]
            

            τ_prev_id = mod(beads.times[id] - beads.times[prev], β)
            τ_id_next = mod(beads.times[next] - beads.times[id], β)
            τ_prev_next = mod(beads.times[next] - beads.times[prev], β)
            
            # ---            
            Uold += U_stored(PIMC, beads, id)
            # ---
            
            #
            # new bead position
            τ_lev = 1/(1/τ_prev_id + 1/τ_id_next)
            σ_lev =sqrt(2*λ*τ_lev) 
                
            # with pbc, compute minimum image distance dr = r_next-r_prev
            # and from that r_next = r_prev + 
            distance!(r_next, r_prev, L, dr)
            @inbounds for d in 1:dim
                rmid = (τ_id_next * r_prev[d] + τ_prev_id * (r_prev[d] + dr[d]))/τ_prev_next
                xold[d] = beads.X[d, id]
                beads.X[d, id] = rmid + σ_lev * randn()
                pbc && (beads.X[d, id] = mod(beads.X[d, id] + L/2, L) - L/2)
            end
            # ---
            Unew += U_update(PIMC, beads, view(xold,:), id, :move)            
            # ---
            t_mid += div(2^par.Nlevels + 1, coun)
        end
        coun *= 2
        # Metropolis
        accept = metro(Unew - Uold)
        !accept && break # bust       
    end
    
    if accept
        par.nacc += 1
        for id in idlist[2:end-1]
            update_stored_slice_data(PIMC, beads, id)
        end
    else
        restore_X!(storage, beads, idlist[2:end-1])        
    end

    # adjust step in thermalization
    if PIMC.ipimc<0 && par.ntry%100 == 0
        accep = par.nacc/par.ntry
        if accep>0.8
            par.Nlevels += 1
        end
        if accep<0.4
            par.Nlevels -= 1
        end
        if M<2^par.Nlevels
            par.Nlevels -= 1
        end
        par.Nlevels = par.Nlevels<1 ? 1 : par.Nlevels
        #@printf("bisection acceptance %8.3f %% step %8.3f\n", accep*100.0, par.Nlevels)
    end
    
    return 100*par.nacc/par.ntry
    
  
end


# Worm moves
# ==========
worm_acc = zeros(8) 
const name_to_num = Dict(:close=>1, :swap=>2, :move_head=>3, :advance=>4,
                         :recede=>5, :bisection=>6)

# Worm moves and their relative weights
_moves = [:close,  :bisection, :move_head, :advance, :recede]
_probs = [0.2, 0.1, 0.3, 0.1, 0.1]


if bose && N>1   # bosons and something to swap
    push!(_moves, :swap)
    push!(_probs, 0.5)
end
_probs ./= sum(_probs) # probabilities  
const moves = _moves
const probs = _probs

# sanity checks
if length(moves) != length(probs)
    error("Worm moves and probs must have same length")
end
iadv = findfirst(==( :advance ), moves)
irec = findfirst(==( :recede  ), moves)
if probs[iadv] != probs[irec] 
    error("It would be safer to have advance/recede tried with the same probability")
end


const distrib = Categorical(probs)

mutable struct t_worm_C_store
    Csum::Float64
    n::Int64
end
const worm_C_store = t_worm_C_store(0.0, 0)

# non-standard move, with backup and worm_measurement
function worm_move!(PIMC::t_pimc, beads::t_beads, links::t_links, beads_backup::t_beads, links_backup::t_links,
                    worm_measurements::Vector{t_measurement},
                    distrib::Categorical = distrib,
                    worm_C_store::t_worm_C_store = worm_C_store)
    """Tries to open a worm, and if succeeds, does worm updates"""
    
    backup_state!(beads_backup, links_backup, beads, links)

   
        
    # ===OPEN========
    worm_stats.N_open_try += 1
    is_open, open_acceptance = worm_open!(PIMC, beads, links)
    # ===============
    if is_open
        worm_stats.N_open_acc += 1
    end
    
    worm_close_count = 0 # to detect worm closing problems
    
   
    while is_open
        PIMC.iworm += 1
        # worm measurements, if any
        if PIMC.ipimc>0
            for meas in worm_measurements
                if (PIMC.ipimc+PIMC.iworm)%meas.frequency==0
                    meas.exe(PIMC, beads, links, meas)
                end
            end
        end
        
        # pick a move 
        move = moves[rand(distrib)]
        ok = true
        if move == :close
            worm_stats.N_close_try += 1
            ok, worm_acc[name_to_num[move]] = worm_close!(PIMC, beads, links)
            if ok
                # worm closes
                worm_stats.N_close_acc += 1
                break
            end
            
            worm_close_count += 1
            if worm_close_count>10000
                println("worm_close_count = $worm_close_count, worm won't close")
                restore_state!(beads, links, beads_backup, links_backup)
                init_stored(PIMC, beads) # must re-calculate action-specific storage after restore
                PIMC.head = -1
                PIMC.tail = -1
                break
            end        
        elseif move == :move_head
            worm_acc[name_to_num[move]] =  worm_move_head!(PIMC, beads, links)
        elseif move == :bisection 
            worm_acc[name_to_num[move]] = bisection_move!(PIMC, beads, links)            
        elseif move == :swap            
            worm_acc[name_to_num[move]] = worm_swap!(PIMC, beads, links)
        elseif move == :advance
            worm_acc[name_to_num[move]] = worm_advance!(PIMC, beads, links)
        elseif move == :recede
            worm_acc[name_to_num[move]] = worm_recede!(PIMC, beads, links)                    
        end
        #PIMC_Common.TEST = true
        #check_links(PIMC, beads, links)
    end

    # Adjust worm_C and worm_K
    # Don't change them after thermalization or obdm normalization will be wrong    
    if PIMC_Common.optimize_worm_params && worm_stats.N_close_try>0 
        if PIMC.ipimc<0 && PIMC.ipimc%10==0  
            #
            close_acceptance = worm_acc[name_to_num[:close]]
            open_close_ratio = open_acceptance/close_acceptance
            worm_C_store.Csum += PIMC_Common.worm_C
            worm_C_store.n += 1
            ave_C = worm_C_store.Csum/worm_C_store.n
            PIMC_Common.worm_C = ave_C + 0.1 * log(3.0/open_close_ratio)  # target ratio 3.0
            PIMC_Common.worm_C = clamp(PIMC_Common.worm_C, 0.001, 3.0)
            @printf("open_close_ratio = %-8.2f  worm_C = %-15.5f\n", open_close_ratio, worm_C)
            #
            
            if open_acceptance < 20.0
                PIMC_Common.worm_K -= 1
            end
            if open_acceptance > 30.0
                PIMC_Common.worm_K += 1
            end
            PIMC_Common.worm_K = clamp(PIMC_Common.worm_K, Int64(0.1*PIMC.M), Int64(0.3*PIMC.M))
            @printf("open_acceptance  = %-8.2f  worm_K = %-15d\n", open_acceptance, worm_K)
        end    
    end


    if PIMC_Common.opt_chin == true
        # EXPERIMENTAL optimize Chin Action parameters chin_a1 and chin_t0
        if PIMC.ipimc<0 && PIMC.ipimc%100==0
            opt_chin_a1_chin_t0(PIMC, beads, links)
        end
    end

   
    # Report acceptances now and then
    if (PIMC.ipimc<0 && PIMC.ipimc%10==0) || PIMC.ipimc%1000 == 0        
        println("")        
        @printf("ipimc = %d\n", PIMC.ipimc)
        @printf("Worm: %12s %8.2f %% \n", "open", open_acceptance)
        for move in moves
            @printf("Worm: %12s %8.2f %%  ", move, worm_acc[name_to_num[move]])
            if move == :swap
                println("swapcount = $(PIMC.swapcount)") 
                print(" \n swap reject reasons: \n")
                s = sum(no_swap_reason)
                for (rn, r) in zip(no_swap_reason_names, no_swap_reason)
                    @printf("  %-30s %-8.2f %%\n", rn, r/max(1.0, s)*100)
                end                
            end
            println()
        end
        if bose && pbc
            meas_superfluid_fraction(PIMC, beads, links)
            meas_cycle_count(PIMC, beads, links)
        end
        println()
    end
    return open_acceptance
    
end


@inline function enforce_worm_limit(PIMC::t_pimc, beads::t_beads, head::Int64, tail::Int64)
    # head may be not set as PIMC.head when coming here
    r_H = @view beads.X[:, head]
    r_T = @view beads.X[:, tail]
    τ_HT = mod(beads.times[tail] - beads.times[head], PIMC.β)
    Δr2 = dist2(r_H, r_T,  PIMC.L) 
    return Δr2/(4*λ*τ_HT) > worm_limit    
end

no_swap_reason = zeros(6)
no_swap_reason_names = ["swap bead is tail", "forward small weights", "met tail", "head-tail>limit",
                        "backward small weights", "Metropolis"]

mutable struct Worm_Swap_Params
    ntry::Int64
    nacc::Int64
end
const DEFAULT_WORM_SWAP_PAR = Worm_Swap_Params(0,0)

const W_swap_old = Vector{Float64}(undef, N_slice_max)
const W_swap_new = Vector{Float64}(undef, N_slice_max)
const swaps = Vector{Int64}(undef, N_slice_max)

mutable struct t_swapdist
    done::Bool
    swapdist::Categorical
end

swapd = t_swapdist(false, Categorical(1))

function sample_swap_k(M, swapd::t_swapdist=swapd)
    if swapd.done
        return rand(swapd.swapdist)
    end
    p = [exp(-0.05 * k) for k in 1:M-1]
    p ./= sum(p)
    swapd.swapdist = Categorical(p)
    swapd.done = true
    return rand(swapd.swapdist)    
end


# SWAP
function worm_swap!(PIMC::t_pimc, beads::t_beads, links::t_links,
                    swaps::Vector{Int64}=swaps,
                    W_swap_old::Vector{Float64}=W_swap_old,
                    W_swap_new::Vector{Float64}=W_swap_new,
                    idlist_buffer::Vector{Int64}=idlist_buffer,
                    par::Worm_Swap_Params=DEFAULT_WORM_SWAP_PAR)    
    """Worm swap for identical particles, with bose==true and N>1"""
    par.ntry += 1
    
    M = PIMC.M
    β = PIMC.β
    L = PIMC.L
    
    t_H = beads.ts[PIMC.head]

    # swap_beads k ∈ U[1:worm_K] above Head
    #k = rand(1:worm_K)
    k = sample_swap_k(M)
    
    t_swap_beads = mod1(t_H + k, M) # slice of swap beads 
    swap_beads = active_beads_on_slice(beads, t_swap_beads) # swap candidates
    #if PIMC.tail in swap_beads
    #    no_swap_reason[1] +=1  
    #    return 100*par.nacc/par.ntry
    #end
   
    #   
    # Weights Head->swap_beads are stored in W_swap_new
    # The naming logic is that there's no path Head->swap_bead in the old config, so it's a "new" path.
    # Similarly, the path new Head-> swap_bead is an "old" path.
    #  
    
    r_H = @view beads.X[:, PIMC.head]
    τ_Hswap = mod(beads.times[swap_beads[1]] - beads.times[PIMC.head], β) 
    nswap = 0    
    @inbounds for id in swap_beads
        r_swap = @view beads.X[:, id]
        nswap += 1
        swaps[nswap] = id       
        W_swap_new[nswap] = rho_0(r_H, r_swap, λ, τ_Hswap, L) 
    end    
    sum_W_swap_new = sum(W_swap_new[1:nswap])
    if nswap==0 || sum_W_swap_new<1e-50 
        no_swap_reason[2] += 1       
        return 100*par.nacc/par.ntry
    end

    #    
    swap_bead = 0
    if nswap == 1
        swap_bead = swaps[1]
    else
        distrib = Categorical(W_swap_new[1:nswap]/sum_W_swap_new)
        swap_bead =  swaps[rand(distrib)]
    end
   
    if swap_bead == PIMC.tail 
        no_swap_reason[1] +=1  
        return 100*par.nacc/par.ntry
    end
    
    # find bead newH k steps below swap_bead
    # abort if tail is met    
    #
    ok = true
    id = swap_bead
    @inbounds for _ in 1:k
        id = links.prev[id]
        if id == PIMC.tail
            # reject move
            ok = false
            break
        end        
    end
    if !ok
        no_swap_reason[3] += 1
        return 100*par.nacc/par.ntry
    end
    newH = id    
    
    
    #
    # Hard limit new head -> tail, make sure worm can close
    #
    if enforce_worm_limit(PIMC, beads, newH, PIMC.tail)
        # reject move
        no_swap_reason[4] += 1
        return 100*par.nacc/par.ntry
    end

    r_T = @view beads.X[:, PIMC.tail]
    τ_newHT = mod(beads.times[PIMC.tail] - beads.times[newH], β)
    
    # The reverse move weight is from newH to one of the swap_beads
    nswap = 0
    r_newH = @view beads.X[:, newH]
    @inbounds for id in swap_beads       
        r_swap = @view beads.X[:, id]
        nswap += 1        
        W_swap_old[nswap] = rho_0(r_newH, r_swap, λ, τ_Hswap, L) 
    end
    sum_W_swap_old = sum(W_swap_old[1:nswap])
    if sum_W_swap_old < 1e-50
        no_swap_reason[5] += 1
        return 100*par.nacc/par.ntry
    end
    
    # beads newH+1 -> swap_bead-1 will be moved
        
    idlist = @view idlist_buffer[1:k+1] # new head -> swap_bead, ends included
    moved_bead_indices = 2:k
    
    Uold = 0.0    
    id = newH
    @inbounds for i in moved_bead_indices 
        id = links.next[id]        
        Uold += U_stored(PIMC, beads, id)  
        idlist[i] = id  
    end
    
    store_X!(storage, beads, idlist[moved_bead_indices]) 

    idlist[1] = PIMC.head # new path is head->swap_bead
    idlist[k+1] = swap_bead
    
    # staging:
    generate_path!(PIMC, beads, links, idlist)
    
    
    # bisection:    
    #bisection_segment!(PIMC, beads, idlist)
    
    Unew = 0.0    
    for i in moved_bead_indices 
        id = idlist[i]
        # storage buffer runs 1 step behind       
        Unew += U_update(PIMC, beads, view(storage.buffer, :, i-1), id, :move)        
    end
    
    ΔU = Unew-Uold
    ratio = exp(-ΔU) * sum_W_swap_new/sum_W_swap_old
    ΔS = -log(ratio)
    # Metropolis
    accept = metro(ΔS)
    
    if accept
        par.nacc += 1
        # set new Head bead 
        PIMC.head = newH
        links.next[PIMC.head] = -1
        for id in idlist[moved_bead_indices] 
            update_stored_slice_data(PIMC, beads, id)
        end
        PIMC.swapcount += 1
    else        
        links.next[PIMC.head] = -1
        # restore link up from newH 
        links.prev[idlist[2]] = newH
        links.next[newH] = idlist[2]
        # restore bead positions; only moved beads where stored
        restore_X!(storage, beads, idlist[moved_bead_indices])
        no_swap_reason[6] += 1
    end
    return 100*par.nacc/par.ntry

end


mutable struct Worm_Open_Params
    ntry::Int64
    nacc::Int64
end
const DEFAULT_WORM_OPEN_PAR = Worm_Open_Params(0,0)


# OPEN
function worm_open!(PIMC::t_pimc, beads::t_beads, links::t_links,
                    idlist_buffer::Vector{Int64}=idlist_buffer,
                    par::Worm_Open_Params=DEFAULT_WORM_OPEN_PAR)
    """Open worm, pick random Head, and Tail k∈U[1,K] above Head"""
    par.ntry += 1
    
    M::Int64 = PIMC.M
    C::Float64 = worm_C
    K::Int64 = worm_K
    μ::Float64 = PIMC.μ    
    β::Float64 = PIMC.β
    L::Float64 = PIMC.L
    # Head to random bead
    PIMC.head = rand(beads.ids[beads.active])
    # find Tail k steps *above* Head
    k = rand(1:K)
    id = PIMC.head    
    @inbounds for m in 1:k
        id = links.next[id]
    end
    PIMC.tail = id
    #
   
    
    # Hard limit
    if enforce_worm_limit(PIMC, beads, PIMC.head, PIMC.tail)
        PIMC.head = -1
        PIMC.tail = -1
        return false, 100*par.nacc/par.ntry        
    end

    idlist = @view idlist_buffer[1:k-1]
    id = PIMC.head
    @inbounds for i in 1:k-1  # k is Tail
        id = links.next[id]
        idlist[i] = id
    end
    
    Uold = 0.0
    Unew = 0.0
    @inbounds @views beads.active[idlist] .= false # U_update needs the beads to be temporarily deactivated 
    @inbounds for id in idlist
        Uold += U_stored(PIMC, beads, id)
        Unew += U_update(PIMC, beads, view(beads.X, :, id), id, :remove) # nothing moved, just deactivated
    end
    @inbounds @views beads.active[idlist] .= true 
    
    r_H = @view beads.X[:, PIMC.head]
    r_T = @view beads.X[:, PIMC.tail]
    τ_HT::Float64 = mod(beads.times[PIMC.tail] - beads.times[PIMC.head], β) # k*τ in PA

    ΔU = Unew-Uold   
    ratio = C*exp(-ΔU-μ*τ_HT)/rho_0(r_H, r_T, λ, τ_HT, L)
    ΔS = -log(ratio)
    # Metropolis
    accept = metro(ΔS)
           
    if accept
        par.nacc += 1        
        #deactivate beads Head->Tail 
        @inbounds @views beads.active[idlist] .= false
        for id in idlist
            update_stored_slice_data(PIMC, beads, id)
        end
        # break connection next to Head and prev to Tail        
        links.next[PIMC.head] = -1
        links.prev[PIMC.tail] = -1        
    else
        PIMC.head = -1
        PIMC.tail = -1
    end
    
    return accept, 100*par.nacc/par.ntry
end


mutable struct Worm_Close_Params
    ntry::Int64
    nacc::Int64
end
const DEFAULT_WORM_CLOSE_PAR = Worm_Close_Params(0,0)

# CLOSE
function worm_close!(PIMC::t_pimc, beads::t_beads, links::t_links,
                     idlist_buffer::Vector{Int64}=idlist_buffer,
                     par=DEFAULT_WORM_CLOSE_PAR)
    """Tries to close the worm by generating path between Head and Tail"""

    par.ntry += 1
    #
    M = PIMC.M
    C = worm_C
    K = worm_K
    μ = PIMC.μ
    τ = PIMC.τ
    β = PIMC.β
    L = PIMC.L
    #
    if enforce_worm_limit(PIMC, beads, PIMC.head, PIMC.tail)
        return false, 100*par.nacc/par.ntry        
    end
    
    # Generate path Head->Tail 
    k = mod(beads.ts[PIMC.tail] - beads.ts[PIMC.head], M)
    idlist = @view idlist_buffer[1:k+1]
    new_bead_indices = 2:k

    m = beads.ts[PIMC.head]
    for i in new_bead_indices
        m = mod1(m+1, M)
        ind = findfirst(b -> !beads.active[b], beads.at_t[m]) # find inactive bead on slice m
        id =  beads.at_t[m][ind]
        idlist[i] = id
    end    
    idlist[1] = PIMC.head
    idlist[end] = PIMC.tail


   
    # staging    
    generate_path!(PIMC, beads, links, idlist)
    
    # bisection
    #bisection_segment!(PIMC, beads, links,  idlist)

    Uold = 0.0
    Unew = 0.0
    @inbounds @views  beads.active[idlist[new_bead_indices]] .= true
    for id in idlist[new_bead_indices]
        Uold += U_stored(PIMC, beads, id)
        Unew += U_update(PIMC, beads, view(beads.X, :, id), id, :add) # dummy old positions
    end
    @inbounds @views beads.active[idlist[new_bead_indices]] .= false
      
    
    r_H = @view beads.X[:, PIMC.head]
    r_T = @view beads.X[:, PIMC.tail]   
    τ_HT = mod(beads.times[PIMC.tail] - beads.times[PIMC.head], β)
    ΔU = Unew - Uold    
    ratio = rho_0(r_H, r_T, λ, τ_HT, L)/(C*exp(ΔU-μ*τ_HT))
    ΔS = -log(ratio)
    # Metropolis
    accept = metro(ΔS)
    
    if accept
        par.nacc += 1        
        @inbounds @views beads.active[idlist[new_bead_indices]] .= true
        for id in idlist[new_bead_indices]
            update_stored_slice_data(PIMC, beads, id)
        end       
        PIMC.head = -1
        PIMC.tail = -1
    else       
        # break links 
        links.next[PIMC.head] = -1
        links.prev[PIMC.tail] = -1        
    end
    return accept, 100*par.nacc/par.ntry
end


mutable struct Worm_Move_Head_Params
    ntry::Int64
    nacc::Int64
    step::Float64
end
const DEFAULT_WORM_MOVE_HEAD_PAR = Worm_Move_Head_Params(0,0,0.5)

# MOVE_HEAD
function worm_move_head!(PIMC::t_pimc, beads::t_beads, links::t_links,
                         gauss::MVector{dim, Float64} = vec_buffer,   
                         idlist_buffer::Vector{Int64}=idlist_buffer,
                         par::Worm_Move_Head_Params=DEFAULT_WORM_MOVE_HEAD_PAR)
    """Moves worm head """

    par.ntry += 1    
    M = PIMC.M    
    β = PIMC.β
    L = PIMC.L    
    
    # Start from base k slices below head    
    k = rand(1:worm_K)
    idlist = @view idlist_buffer[1:k+1]
    new_bead_indices = 2:k+1 # also head will be new
    
    ok = true
    base = PIMC.head    
    @inbounds for _ in 1:k
        base = links.prev[base]
        if base == PIMC.tail            
            ok = false # met tail
            break
        end
    end
    if !ok
        return 100*par.nacc/par.ntry
    end
       
    Uold = 0.0    
    id = base
    @inbounds for i in new_bead_indices
        id = links.next[id]
        Uold += U_stored(PIMC, beads, id)        
        idlist[i] = id 
    end
    idlist[1] = base 

    # store beads *before* head is moved
    store_X!(storage, beads, idlist)
        
    # sample new Head
    randn!(gauss)
    @inbounds for d in 1:dim        
        beads.X[d, PIMC.head] += par.step*gauss[d]
        if pbc
            beads.X[d, PIMC.head] = mod(beads.X[d, PIMC.head] + L/2, L) - L/2
        end
    end
   
    if enforce_worm_limit(PIMC, beads, PIMC.head, PIMC.tail)
        # move head back to old position (other beads weren't moved yet)
        @inbounds @views beads.X[:, PIMC.head] .= storage.buffer[:, length(idlist)]
        return 100*par.nacc/par.ntry
    end
    
    # generate new path base -> new head
    # staging
    generate_path!(PIMC, beads, links, idlist)

    Unew = 0.0
    @inbounds for i in new_bead_indices # contains head
        id = idlist[i]
        # idlist index is storage.buffer index
        Unew += U_update(PIMC, beads, view(storage.buffer, :, i), id, :move)
    end

    # long-jump rho_0 factors of new and old head
    r_new_H = @view beads.X[:, PIMC.head]
    r_old_H = @view storage.buffer[:, length(idlist)]
    r_base = @view beads.X[:, base]
    τ_baseH = mod(beads.times[PIMC.head] - beads.times[base], β)
    d2new = dist2(r_base, r_new_H, L)
    d2old = dist2(r_base, r_old_H, L)
    
    rhoratio = (d2new - d2old)/(4λ*τ_baseH)
    ΔS = Unew - Uold  + rhoratio
    # Metropolis
    accept = metro(ΔS)
    
    if accept
        par.nacc += 1
        for id in idlist[new_bead_indices]
            update_stored_slice_data(PIMC, beads, id)
        end
    else
        restore_X!(storage, beads, idlist)
    end

    # adjust step in thermalization; favor large steps at the expense of acceptance     
    if PIMC.ipimc<0 && par.ntry%10 == 0
        accep = par.nacc/par.ntry
        if accep>0.30
            par.step *= 1.5
        end
        if accep<0.10 
            par.step *= 0.5
        end        
        par.step = clamp(par.step, 0.001, 3.0)
    end
    #
    return 100*par.nacc/par.ntry
end


    
mutable struct Worm_Advance_Params
    ntry::Int64
    nacc::Int64
end
const DEFAULT_WORM_ADVANCE_PAR = Worm_Advance_Params(0,0)


#exp_dist = Exponential(0.5)
#expo =  Exponential(4.0)

function truncated_distribution(p, Kmax)
    #k = ceil(Int64, rand(Truncated(exp_dist, 1, Kmax)))
    #return k
    #return ceil(Int64, rand(expo))
             
    # CDF of Geometric(p) is F(k) = 1 - (1 - p)^k
    # inverse sampling:
    u = rand()
    q = 1 - p
    k = ceil(Int64, log(1 - u * (1 - q^Kmax)) / log(q))
    return k
end


# ADVANCE
function worm_advance!(PIMC::t_pimc, beads::t_beads, links::t_links,
                       gauss::MVector{dim, Float64} = vec_buffer,
                       idlist_buffer::Vector{Int64}=idlist_buffer,
                       par::Worm_Advance_Params=DEFAULT_WORM_ADVANCE_PAR)
    """Advances worm head"""
    par.ntry += 1
    
    M = PIMC.M
    μ = PIMC.μ
    β = PIMC.β
    L = PIMC.L
    
    k =  1 #truncated_distribution(0.5, M)# worm_K)
    
    t_H = beads.ts[PIMC.head]
    t_T = beads.ts[PIMC.tail]
    
    if PIMC.canonical
        # don't let head advance past tail
        m = t_H
        @inbounds for i = 1:k
            m = mod1(m+1, M)
            if m==t_T
                return 100*par.nacc/par.ntry
            end
        end
    end
    
    # pick new head k slices above current Head (no links to follow)    
    idlist = @view idlist_buffer[1:k+1] # from head to new head
    new_bead_indices = 2:k+1
    
    Uold = 0.0
    m = t_H
    idlist[1] = PIMC.head
    @inbounds for i in new_bead_indices         
        m = mod1(m+1, M)
        ind = findfirst(b -> !beads.active[b], beads.at_t[m]) # find inactive bead on slice m        
        id = beads.at_t[m][ind] #  beads.at_t[m] is a Vector{Int} and findfirst gives index to that Vector
        idlist[i] = id
        Uold += U_stored(PIMC, beads, id)         
    end
    newH = idlist[end]    
    Δτ = mod(beads.times[newH] - beads.times[PIMC.head], β)
    # sample new Head position from free-particle Green's function rho_0
    
    randn!(gauss)
    σ = sqrt(2λ*Δτ)
    
    @inbounds for d in 1:dim        
        beads.X[d, newH] = beads.X[d, PIMC.head] + σ*gauss[d]
        if pbc
            beads.X[d, newH] = mod(beads.X[d, newH] + L/2, L) - L/2
        end
    end
    
    if enforce_worm_limit(PIMC, beads, newH, PIMC.tail)
        # newH is still inactive, no need to reset it 
        return 100*par.nacc/par.ntry
    end

    # staging 
    # generate new path segment PIMC.head -> id_newH
    generate_path!(PIMC, beads, links, idlist)

    #
    # bisection
    #bisection_segment!(PIMC, beads, links, idlist)
    
    Unew = 0.0   
    @inbounds @views beads.active[idlist[new_bead_indices]] .= true # beads must be active to get correct U     
    for i in new_bead_indices
        id = idlist[i]
        Unew += U_update(PIMC, beads, gauss, id, :add) # dummy old position
    end
    @inbounds @views beads.active[idlist[new_bead_indices]] .= false 
    ΔU = Unew - Uold
    
    # Metropolis
    ΔS = ΔU + μ*Δτ 
    accept = metro(ΔS)

    if accept
        par.nacc += 1        
        PIMC.head = newH        
        links.next[newH] = -1
        # activate new beads 
        @inbounds @views beads.active[idlist[new_bead_indices]] .= true
        for i in new_bead_indices
            id = idlist[i]
            update_stored_slice_data(PIMC, beads, id)
        end
    else        
        # break link
        links.next[PIMC.head] = -1
    end
    
    return 100*par.nacc/par.ntry
end


mutable struct Worm_Recede_Params
    ntry::Int64
    nacc::Int64
end
const DEFAULT_WORM_RECEDE_PAR = Worm_Recede_Params(0,0)


# RECEDE
function worm_recede!(PIMC::t_pimc, beads::t_beads, links::t_links,
                      idlist_buffer::Vector{Int64}=idlist_buffer,
                      par::Worm_Recede_Params=DEFAULT_WORM_RECEDE_PAR)
    """Recedes worm head """
    par.ntry += 1
        
    M = PIMC.M   
    μ = PIMC.μ
    β = PIMC.β
    L = PIMC.L

    t_H = beads.ts[PIMC.head]
    t_T = beads.ts[PIMC.tail]
        
    k = 1 # truncated_distribution(0.5, M) #worm_K)  
    

    if PIMC.canonical
        # don't let head recede past tail
        m = t_H
        @inbounds for i = 1:k
            m = mod1(m-1, M)
            if m==t_T
                return 100*par.nacc/par.ntry
            end
        end
    end
    
    idlist = @view idlist_buffer[1:k] # head, head-1, ..., head-k+1
    # all k beads in idlist may be removed
    id = PIMC.head
    @inbounds for i in 1:k
        idlist[i] = id
        id = links.prev[id]
    end
    newH = id
 
    # worm_limit for open/close and advance/reject detailed balance
    if enforce_worm_limit(PIMC, beads, newH, PIMC.tail)
        return 100*par.nacc/par.ntry
    end
    
    Uold = 0.0
    @inbounds for id in idlist
        Uold += U_stored(PIMC, beads, id) 
    end       
   
    
  
    
    @inbounds @views beads.active[idlist] .= false
    Unew = 0.0
    for id in idlist
        Unew += U_update(PIMC, beads, view(beads.X, :, id), id, :remove)
    end
    @inbounds @views beads.active[idlist] .= true
        
    ΔU = Unew - Uold
    
    Δτ = mod(beads.times[PIMC.head] - beads.times[newH], β)
    # Metropolis
    ΔS = ΔU + μ*Δτ 

    
    accept = metro(ΔS)
    if accept
        par.nacc += 1
        PIMC.head = newH
        links.next[PIMC.head] = -1
        # deactivate removed beads       
        @inbounds @views beads.active[idlist] .= false
        for id in idlist
            update_stored_slice_data(PIMC, beads, id)
        end
        
        
    end
    return 100*par.nacc/par.ntry
    
end

end
