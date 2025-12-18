__precompile__(false)
module PIMC_Moves

using BenchmarkTools
using Random, LinearAlgebra
BLAS.set_num_threads(1) 
using Distributions: Categorical, Geometric, Exponential, Truncated
using Printf
using InteractiveUtils # for @code_warntype profiling
using StaticArrays


using TimerOutputs
const to = TimerOutput()


# local modules:
push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Structs
using PIMC_Utilities
using QMC_Statistics

using PIMC_Action_Interface
using PIMC_Chin_Action: U

using PIMC_Systems
using PIMC_Measurements: meas_superfluid_fraction, meas_cycle_count

export add_move!  
export bead_move!, rigid_move!, bisection_move!, worm_move!
export worm_close!, worm_swap!, worm_move_head!, worm_advance!,  worm_recede! 


    
# -------------------
# Light-weight buffers, *not* thread-safe

const gauss_buffer = MVector{dim, Float64}(undef)
const Δr_buffer = MVector{dim, Float64}(undef)
const idlist_buffer = Vector{Int}(undef, 1000) # should be enough

function add_move!(PIMC::t_pimc, frequency::Int, name::Symbol, exe::Function)
    sname = String(name)
    move = t_move(frequency=frequency, name=name, sname=sname, exe=exe, ntries=0, nacc=0, acc=0.0)
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



# =======
# Moves 
# =======


mutable struct Bead_Move_Params
    ntry::Int
    nacc::Int
    step::Float64
end
const DEFAULT_BEAD_MOVE_PAR = Bead_Move_Params(0,0,0.2)

function bead_move!(PIMC::t_pimc, beads::t_beads, links::t_links, par::Bead_Move_Params=DEFAULT_BEAD_MOVE_PAR)
    """Moves a single bead"""
   
    L = PIMC.L
    # random bead    
    id = rand(beads.ids[beads.active])
    storage = store_X!(beads, id)
    xold = @view storage.X[:, 1]

    Uold = U_stored(PIMC, beads, id)    
    Kold = K(PIMC, beads, links, id)
    Sold = Uold + Kold

    # --------
    if pbc 
        for d in 1:dim
            x = beads.X[d, id] 
            x  += par.step * (rand() - 0.5)
            x  -= L * floor((x + L/2) / L)
            beads.X[d, id] = x
        end
    else
        for d in 1:dim
            beads.X[d, id] += par.step * (rand() - 0.5)
        end
    end
    
    Unew = U_update(PIMC, beads, xold, id, :move)
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
        restore_X!(beads, id)
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
    ntry::Int
    nacc::Int
    step::Float64
end
const DEFAULT_RIGID_MOVE_PAR = Rigid_Move_Params(0, 0, 0.2)

function rigid_move!(PIMC::t_pimc, beads::t_beads, links::t_links,
                     gauss::MVector{dim, Float64}=gauss_buffer,
                     Δr::MVector{dim, Float64}=Δr_buffer,
                     idlist_buf::Vector{Int}=idlist_buffer,
                     par::Rigid_Move_Params=DEFAULT_RIGID_MOVE_PAR)
    """Moves a whole loop, but not exchange loops"""
    par.ntry += 1    
    L = PIMC.L
    M = PIMC.M
    # start from random bead
    id1 = rand(beads.ids[beads.active])
    idlist = @view idlist_buf[1:M] # move M beads 
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
    
    storage = store_X!(beads, idlist)
    Uold = 0.0     
    @inbounds for id in idlist
        Uold += U_stored(PIMC, beads, id)   
    end

    # same Δr for all
    randn!(gauss)
    @inbounds Δr .= par.step .* gauss
    
    
    #
    # In an exchange loop move there can be more than one bead moving on a single slice
    # The value of beads.X[:, id1] has already changed when the multi-M loop
    # moves another bead on the same slice as id1. Gave up earlier on exchange loops.
    #
    # move the loop
    
    
    @inbounds begin
        for id in idlist, d in 1:dim
            x = beads.X[d, id]
            x += Δr[d]
            if pbc
                x  -= L * floor((x + L/2) / L)
            end
            beads.X[d, id] = x
        end
    end
    
    Unew = 0.0
    @inbounds @views begin
        for (i, id) in enumerate(idlist)
            Unew += U_update(PIMC, beads, storage.X[:, i], id, :move)
        end
    end
    

    ΔU = Unew - Uold # no change in kinetic action
    # Metropolis
    accept = metro(ΔU)
    if accept
        par.nacc += 1
        @inbounds for id in idlist
            update_stored_slice_data(PIMC, beads, id)
        end        
    else
        restore_X!(beads, idlist)
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
    ntry::Int
    nacc::Int
    Nlevels::Int
end
# for 3 levels:
# fixed end points 1, 9
#           1 2 3 4 5 6 7 8 9
# level 1           5
# level 2       3       7
# level 3     2   4   6   8
#

const DEFAULT_BISECTION_MOVE_PAR = Bisection_Move_Params(0, 0, 4)

const bis_xold_buffer = MVector{dim, Float64}(undef) 

function bisection_move!(PIMC::t_pimc, beads::t_beads, links::t_links,
                         xold::MVector{dim, Float64}=bis_xold_buffer,
                         idlist_buf::Vector{Int}=idlist_buffer,
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
    idlist = @view idlist_buf[1:K+1]    
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
    # store midbeads
    midbeads = @view idlist[2:end-1]
    store_X!(beads, midbeads)     
    accept = true
    coun = 1
    @inbounds for lev in 1:par.Nlevels              
        Uold = 0.0
        Unew = 0.0
        np = 2^(par.Nlevels-lev)
        t_mid = np + 1
        @inbounds for k in 1:coun
            t_prev = t_mid - np
            t_next = t_mid + np            
                        
            prev = idlist[t_prev]
            next = idlist[t_next]
            id = idlist[t_mid]

            t_id = beads.times[id]
            t_prev = beads.times[prev]
            t_next = beads.times[next]
            Δτ_prev_id = wrapβ(t_id - t_prev, β)
            Δτ_id_next = wrapβ(t_next - t_id, β)
            Δτ_prev_next = wrapβ(t_next - t_prev, β)
            
            # ---
            Uold += U_stored(PIMC, beads, id)
            # ---
            
            #
            # new bead position
            Δτ_lev = 1/(1/Δτ_prev_id + 1/Δτ_id_next)
            σ_lev = sqrt(2λ*Δτ_lev) 
                
            # with pbc, compute minimum image distance dr = r_next-r_prev
            # and from that r_next = r_prev + dr
            if pbc
                # ok to continue from folded coordinate x
                # because dr is always toward the minimum image of r_end                
                @inbounds for d in 1:dim
                    dr = beads.X[d, next] - beads.X[d, prev]
                    dr -= L * round(dr/L)
                    x_prev = beads.X[d, prev] 
                    rmid = (Δτ_id_next * x_prev + Δτ_prev_id * (x_prev + dr))/Δτ_prev_next
                    xold[d] = beads.X[d, id] # store
                    x = beads.X[d, id]
                    x = rmid + σ_lev * randn()
                    x -= L * floor((x + L/2) / L) 
                    beads.X[d, id] = x
                end                
            else
                @inbounds for d in 1:dim
                    dr = beads.X[d, next] - beads.X[d, prev]
                    x_prev = beads.X[d, prev] 
                    rmid = (Δτ_id_next * x_prev + Δτ_prev_id * (x_prev + dr))/Δτ_prev_next
                    xold[d] = beads.X[d, id] # store
                    beads.X[d, id] = rmid + σ_lev * randn()
                end
            end                           
            # ---
            Unew += U_update(PIMC, beads, xold, id, :move)            
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
        for id in midbeads
            update_stored_slice_data(PIMC, beads, id)
        end
    else
        restore_X!(beads, midbeads)         
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
        if  2^par.Nlevels>M
            par.Nlevels -= 1
        end
        # @printf("bisection acceptance %8.3f %% level %8.3f\n", accep*100.0, par.Nlevels)
    end
    
    return 100*par.nacc/par.ntry
    
  
end

mutable struct t_worm_C_store
    Csum::Float64
    n::Int
end
const worm_C_store = t_worm_C_store(0.0, 0)

mutable struct t_worm_acc
    acc::Vector{Float64}
end


# non-standard move, with backup and worm_measurement
function worm_move!(PIMC::t_pimc, beads::t_beads, links::t_links, beads_backup::t_beads, links_backup::t_links,
                    worm_measurements::Vector{t_measurement}, 
                    worm_moves::Vector{t_move},                    
                    distrib::Categorical,
                    to::TimerOutput,
                    worm_C_store::t_worm_C_store = worm_C_store)
    """Tries to open a worm, and if succeeds, does worm updates and off-diagonal measurements"""
    
    backup_state!(beads_backup, links_backup, beads, links)

    # ===OPEN========
    worm_stats.N_open_try += 1
    @timeit to "moves" begin
        @timeit to "worm_open" begin
            is_open, open_acceptance = worm_open!(PIMC, beads, links)
        end
    end
    # ===============
    if is_open
        worm_stats.N_open_acc += 1
    end     
    
    worm_close_count = 0   # detect worm closing problems
    
    while is_open
        PIMC.iworm += 1
        #
        # worm measurements, if any
        # -------------------------
        if PIMC.ipimc>0            
            for meas in worm_measurements
                if PIMC.iworm%meas.frequency==0
                    @timeit to "measurements" begin
                        @timeit to meas.sname begin                                                               
                            meas.exe(PIMC, beads, links, meas)
                        end                        
                    end
                end
            end
        end
        
        # pick a move
        # -----------
        move = worm_moves[rand(distrib)]
        @timeit to "moves" begin
            @timeit to move.sname begin
                ok = true
                if move.name === :worm_close
                    worm_stats.N_close_try += 1
                    ok, move.acc = worm_close!(PIMC, beads, links)
                    if ok
                        # worm closes
                        worm_stats.N_close_acc += 1
                        break
                    end
                    # detect non-closing worm
                    worm_close_count += 1
                    if worm_close_count>10000
                        println("worm_close_count = $worm_close_count, worm won't close")
                        restore_state!(beads, links, beads_backup, links_backup)
                        init_stored(PIMC, beads) # re-calculate action-specific storage after restore                
                        PIMC.head = -1
                        PIMC.tail = -1
                        break
                    end        
                else                    
                    move.acc = move.exe(PIMC, beads, links)
                end
            end            
        end
        #PIMC_Common.TEST = true
        #check_links(PIMC, beads, links)
    end # is_open

    for move in worm_moves
        PIMC.acceptance[move] = move.acc
    end
    
    # Adjust worm_C and worm_K
    # Don't change them after thermalization or obdm normalization will be wrong    
    if PIMC_Common.optimize_worm_params && worm_stats.N_close_try>0
        close_acceptance = 0.0 # just to capture
        for move in worm_moves
            if move.name===:worm_close
                close_acceptance = move.acc
                break
            end
        end

        if PIMC.ipimc<0 && PIMC.ipimc%100==0  
            #
            open_close_ratio = open_acceptance/close_acceptance
            worm_C_store.Csum += PIMC_Common.worm_C
            worm_C_store.n += 1
            ave_C = worm_C_store.Csum/worm_C_store.n
            PIMC_Common.worm_C = ave_C + 0.1 * log(2.0/open_close_ratio)  # target ratio 2.0
            PIMC_Common.worm_C = clamp(PIMC_Common.worm_C, 0.001, 3.0)
            @printf("open_close_ratio = %-8.2f  worm_C = %-15.5f\n", open_close_ratio, worm_C)
            #
            
            if open_acceptance < 40.0
                PIMC_Common.worm_K -= 1
            end
            if open_acceptance > 60.0
                PIMC_Common.worm_K += 1
            end
            PIMC_Common.worm_K = clamp(PIMC_Common.worm_K, ceil(Int, 0.05* PIMC.M), ceil(Int, 0.3 * PIMC.M))
            PIMC_Common.worm_K = max(PIMC_Common.worm_K, 3) 
            @printf("open_acceptance  = %-8.2f  worm_K = %-15d\n", open_acceptance, PIMC_Common.worm_K)
        end    
    end


    if PIMC_Common.opt_chin == true
        # EXPERIMENTAL optimize Chin Action parameters chin_a1 and chin_t0
        if PIMC.ipimc<0 && PIMC.ipimc%5000==0 
            opt_chin_a1_chin_t0(PIMC, beads, links)
        end
    end

   
    # Report swap acceptance details, superfluid fraction and cycle count
    if (PIMC.ipimc<0 && PIMC.ipimc%100==0) || PIMC.ipimc%10000 == 0        
        @printf("ipimc = %d\n", PIMC.ipimc)
        for move in worm_moves            
            if move.name === :worm_swap
                println("swapcount = $(PIMC.swapcount)") 
                print(" \n swap reject reasons: \n")
                s = sum(no_swap_reason)
                for (rn, r) in zip(no_swap_reason_names, no_swap_reason)
                    if r>0.0
                        @printf("  %-30s %-8.2f %%\n", rn, r/max(1.0, s)*100)
                    end
                end
                break
            end
        end
        if bose && pbc
            meas_superfluid_fraction(PIMC, beads, links)
            meas_cycle_count(PIMC, beads, links)
        end
    end
    return open_acceptance
    
end


@inline function enforce_worm_limit(PIMC::t_pimc, beads::t_beads, head::Int, tail::Int)
    # head may be not be set as PIMC.head when coming here; same with tail
    return false #always accept
    
    # NB: uses minimum image distances
    #Δτ = wrapβ(beads.times[tail] - beads.times[head], PIMC.β)
    #Δr2 = dist2(beads.X, head, tail, PIMC.L)
    #return Δr2/(4*λ*Δτ) > worm_limit
    #return rho_0(beads.X, head. tail, λ, Δτ, PIMC.L)<1e-14 # skip small 
end

no_swap_reason = zeros(6)
no_swap_reason_names = ["swap bead is tail", "forward small weights", "met tail", "head-tail>limit",
                        "backward small weights", "Metropolis"]

mutable struct Worm_Swap_Params
    ntry::Int
    nacc::Int
end
const DEFAULT_WORM_SWAP_PAR = Worm_Swap_Params(0,0)

const W_swap_old_buffer = Vector{Float64}(undef, N_slice_max)
const W_swap_new_buffer = Vector{Float64}(undef, N_slice_max)

mutable struct t_swapdist
    done::Bool
    swapdist::Categorical
end

const swapd = t_swapdist(false, Categorical(1))

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
                    W_old_buf::Vector{Float64}=W_swap_old_buffer,
                    W_new_buf::Vector{Float64}=W_swap_new_buffer,
                    idlist_buf::Vector{Int}=idlist_buffer,
                    par::Worm_Swap_Params=DEFAULT_WORM_SWAP_PAR)    
    """Worm swap for identical particles, with bose==true and N>1"""
    # Choose swap bead from k steps above head using minimum image distances
    par.ntry += 1
    
    M = PIMC.M
    β = PIMC.β
    L = PIMC.L
    
    t_H = beads.ts[PIMC.head]

    # swap_beads are k steps above head
    # option: k = rand(1:worm_K)
    k = sample_swap_k(M)
    
    t_swap_beads = mod1(t_H + k, M) # slice of swap beads 
    swap_beads = beads.active_at_t[t_swap_beads] # swap candidates
    n_swap_beads = length(swap_beads)
    #   
    # Weights old head->swap_beads are stored in W_new
    # Weights new head->swap_beads are stored in W_old
    # The naming logic is that there's no path head -> swap_bead in the old config, so it's a "new" path.
    #
    # old head - swap bead time distance
    τ_Hswap = wrapβ(beads.times[swap_beads[1]] - beads.times[PIMC.head], β)
    
    # weights head -> swap_beads[:]
    W_new = @view W_new_buf[1:n_swap_beads]
    @inbounds @views begin
        for (i,b) in enumerate(swap_beads)
            W_new[i] = rho_0(beads.X, PIMC.head, b, λ, τ_Hswap, L) 
        end
    end
    sum_W_new = sum(W_new)
    if sum_W_new<1e-50 
        no_swap_reason[2] += 1
        return 100*par.nacc/par.ntry
    end

    distrib = Categorical(W_new/sum_W_new)
    swap_bead = swap_beads[rand(distrib)]
   
    if swap_bead == PIMC.tail 
        no_swap_reason[1] +=1
        return 100*par.nacc/par.ntry
    end
    #
    # find new head (newH) k steps below swap_bead
    # abort if tail is met    
    #
    ok = true
    b = swap_bead
    @inbounds for _ in 1:k
        b = links.prev[b]
        if b == PIMC.tail
            # reject move
            no_swap_reason[3] += 1
            return 100*par.nacc/par.ntry
        end        
    end
    # found new head candidate
    newH = b      
    #
    # Hard limit new head -> tail, make sure worm can close
    if enforce_worm_limit(PIMC, beads, newH, PIMC.tail)
        # reject move
        no_swap_reason[4] += 1
        return 100*par.nacc/par.ntry
    end

    # newH  newH+1  ... swap_bead-1  swap_bead 
    #   1     2          k            k+1
    # old   move         move         old 
    idlist = @view idlist_buf[1:k+1] 
    Uold = 0.0    
    b = newH
    @inbounds for i in 2:k
        b = links.next[b]        
        Uold += U_stored(PIMC, beads, b)  
        idlist[i] = b  
    end
    # store moving beads
    moved_beads = @view idlist[2:end-1]
    storage = store_X!(beads, moved_beads) 

    # calculate reverse move weights newH -> swap_beads[:]
    W_old = @view W_old_buf[1:n_swap_beads]
    @inbounds @views begin
        for (i,b) in enumerate(swap_beads)
            W_old[i] = rho_0(beads.X, newH, b, λ, τ_Hswap, L) 
        end
    end
    sum_W_old = sum(W_old)
    if sum_W_old < 1e-50
        no_swap_reason[5] += 1
        return 100*par.nacc/par.ntry
    end

    idlist[1]   = PIMC.head # new path is old head->swap_bead
    idlist[end] = swap_bead

    # -----------------------------------------
    generate_path!(PIMC, beads, links, idlist)
    # -----------------------------------------
    
    Unew = 0.0
    @inbounds @views begin
        for (i,b) in enumerate(moved_beads)
            Unew += U_update(PIMC, beads, storage.X[:, i], b, :move)
        end
    end
    
    ΔU = Unew - Uold
    ΔS = ΔU - log(sum_W_new/sum_W_old)
    # Metropolis
    accept = metro(ΔS)
    
    if accept
        par.nacc += 1
        # set new head bead 
        PIMC.head = newH
        links.next[PIMC.head] = -1
        for b in moved_beads
            update_stored_slice_data(PIMC, beads, b)
        end
        PIMC.swapcount += 1
    else        
        links.next[PIMC.head] = -1
        # restore link up from newH 
        links.prev[idlist[2]] = newH
        links.next[newH] = idlist[2]        
        # restore moved bead positions 
        restore_X!(beads, moved_beads) 
        no_swap_reason[6] += 1
    end
    
    return 100*par.nacc/par.ntry
end


mutable struct Worm_Open_Params
    ntry::Int
    nacc::Int
end
const DEFAULT_WORM_OPEN_PAR = Worm_Open_Params(0,0)


# OPEN
function worm_open!(PIMC::t_pimc, beads::t_beads, links::t_links,
                    idlist_buf::Vector{Int}=idlist_buffer,
                    par::Worm_Open_Params=DEFAULT_WORM_OPEN_PAR)
    """Open worm, pick random Head, and Tail k∈U[1,K] above Head"""
    par.ntry += 1
    
    M::Int = PIMC.M
    C::Float64 = worm_C
    K::Int = worm_K
    μ::Float64 = PIMC.μ    
    β::Float64 = PIMC.β
    L::Float64 = PIMC.L
    
    if K > M
        error("Your starting worm_K > M (may happen if M is very small)")
    end

    # pick head and tail 
    k = 0 
    ok = false
    ms = get_ms(M)
    ks = get_ms(K)
    for tries in 1:10
        # Head to random bead (on physical slice in range ms)
        m = rand(ms)
        bs = beads.active_at_t[m]
        PIMC.head = rand(bs)
        # find tail k steps *above* head
        k = rand(ks) + 1 # tail on a physical slice 2:3:M in Chin action 
        id = PIMC.head    
        @inbounds for m in 1:k
            id = links.next[id]
        end
        PIMC.tail = id
        if !enforce_worm_limit(PIMC, beads, PIMC.head, PIMC.tail)
            # Hard limit
            ok = true
            break
        end
    end
    if !ok
        PIMC.head = -1
        PIMC.tail = -1
        return false, 100*par.nacc/par.ntry        
    end

    idlist = @view idlist_buf[1:k-1] # beads to be removed 
    id = PIMC.head
    @inbounds for i in 1:k-1  # k is tail
        id = links.next[id]
        idlist[i] = id
    end
    
    Uold = 0.0
    Unew = 0.0
    deactivate_beads!(beads, idlist)  # U_update needs the beads to be temporarily deactivated 
    @inbounds @views for id in idlist
        Uold += U_stored(PIMC, beads, id)
        Unew += U_update(PIMC, beads, beads.X[:, id], id, :remove) # nothing moved, just deactivated
    end
    activate_beads!(beads, idlist)  # restore activation 
    
    Δτ = wrapβ(beads.times[PIMC.tail] - beads.times[PIMC.head], β) 

    ΔU = Unew-Uold   
    ratio = C*exp(-ΔU-μ*Δτ)/rho_0(beads.X, PIMC.head, PIMC.tail, λ, Δτ, L)
   
    ΔS = -log(ratio)
    # Metropolis
    accept = metro(ΔS)
    if accept
        par.nacc += 1        
        #deactivate beads head->tail 
        deactivate_beads!(beads, idlist)
        # update stored U
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
    ntry::Int
    nacc::Int
end
const DEFAULT_WORM_CLOSE_PAR = Worm_Close_Params(0,0)

# CLOSE
function worm_close!(PIMC::t_pimc, beads::t_beads, links::t_links,
                     idlist_buf::Vector{Int}=idlist_buffer,
                     par=DEFAULT_WORM_CLOSE_PAR)
    """Tries to close the worm by generating path between Head and Tail"""
    # In PBC, keep closing attempts local, between minimum images.
    # Else you always connect specific images of head and tail, which suppresses winding.    
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
    #
    # Generate path Head->Tail (in pbc, to minimum image tail)
    #
    k = mod(beads.ts[PIMC.tail] - beads.ts[PIMC.head], M) # NB: use mod, not mod1
    idlist = @view idlist_buf[1:k+1]
    new_bead_indices = 2:k

    m = beads.ts[PIMC.head]
    @inbounds begin
        for i in new_bead_indices
            m = mod1(m+1, M)
            idlist[i] = beads.inactive_at_t[m][1]
        end
        idlist[1] = PIMC.head
        idlist[end] = PIMC.tail
    end

    # -----------------------------------------
    generate_path!(PIMC, beads, links,  idlist)
    # -----------------------------------------    
    
    Uold = 0.0
    Unew = 0.0
    @inbounds begin
        activate_beads!(beads, view(idlist, new_bead_indices))
        for b in idlist[new_bead_indices]
            Uold += U_stored(PIMC, beads, b)
            Unew += U_update(PIMC, beads, view(beads.X, :, b), b, :add) # dummy old positions
        end
        deactivate_beads!(beads, view(idlist, new_bead_indices))
    end
    Δτ = wrapβ(beads.times[PIMC.tail] - beads.times[PIMC.head], β)    
    ΔU = Unew - Uold
    ΔU = clamp(ΔU, -100.0, 100.0) # avoid overflow
    ratio = rho_0(beads.X, PIMC.head, PIMC.tail, λ, Δτ, L)/(C*exp(ΔU-μ*Δτ))
    
    ΔS = -log(ratio)
    # Metropolis
    accept = metro(ΔS)

    if accept
        par.nacc += 1
        @inbounds begin
            activate_beads!(beads, view(idlist, new_bead_indices))
            for id in idlist[new_bead_indices]
                update_stored_slice_data(PIMC, beads, id)
            end
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
    ntry::Int
    nacc::Int
    step::Float64
end
const DEFAULT_WORM_MOVE_HEAD_PAR = Worm_Move_Head_Params(0,0,1.3)


# MOVE_HEAD
function worm_move_head!(PIMC::t_pimc, beads::t_beads, links::t_links,
                         gauss::MVector{dim, Float64} = gauss_buffer,
                         idlist_buf::Vector{Int}=idlist_buffer,
                         par::Worm_Move_Head_Params=DEFAULT_WORM_MOVE_HEAD_PAR)
    """Moves worm head """

    par.ntry += 1    
    M = PIMC.M    
    β = PIMC.β
    L = PIMC.L    
    
    # Start from base k slices below head
    # base               head
    # 1      2  .... k    k+1
    # fixed move    move  move
    
    k = rand(1:worm_K)
    idlist = @view idlist_buf[1:k+1]
    new_bead_indices = 2:k+1 
    
    base = PIMC.head    
    @inbounds for _ in 1:k
        base = links.prev[base]
        if base == PIMC.tail            
            # met tail
            return 100*par.nacc/par.ntry
        end
    end

    idlist[1] = base
    Uold = 0.0
    b = base
    @inbounds for i in new_bead_indices
        b = links.next[b]
        Uold += U_stored(PIMC, beads, b)        
        idlist[i] = b 
    end
    # store path 
    storage = store_X!(beads, idlist) 
        
    # move head 
    # -------------------------------------------------
    @inbounds @views begin
        if pbc
            for d in 1:dim
                x = beads.X[d, PIMC.head]   
                x += par.step*(rand() - 0.5)*L # par.step* U[-L/2, L/2]
                x -= L * floor((x + L/2) / L)
                beads.X[d, PIMC.head] = x
            end
        else
            for d in 1:dim
                x = beads.X[d, PIMC.head]   
                x += par.step*randn()
                beads.X[d, PIMC.head] = x
            end
        end
    end
    # -------------------------------------------------        
    # check new head-tail limit
    if enforce_worm_limit(PIMC, beads, PIMC.head, PIMC.tail)
        restore_X!(beads, idlist)
        # or move head back to old position (other beads weren't moved yet)
        return 100*par.nacc/par.ntry
    end    
    
    # generate new path base -> head
    # ----------------------------------------
    generate_path!(PIMC, beads, links, idlist)
    # ----------------------------------------

   
    Unew = 0.0
    @inbounds @views begin
        for i in new_bead_indices # base+1 -> head
            # storage index is idlist index
            Unew += U_update(PIMC, beads, storage.X[:, i], idlist[i], :move)
        end
    end
    
    
    # rho_0 factors of base->new head and base->old head
    Δτ_baseH = wrapβ(beads.times[PIMC.head] - beads.times[base], β)
    @views begin
        d2new = dist2(beads.X, base, PIMC.head, L)
        d2old = dist2(beads.X[:, base], storage.X[:, k+1], L)
    end
    rhoratio = (d2new - d2old)/(4λ*Δτ_baseH)
    
    ΔS = Unew - Uold  + rhoratio
    # Metropolis
    accept = metro(ΔS)

    if accept
        par.nacc += 1
        for id in idlist[new_bead_indices]
            update_stored_slice_data(PIMC, beads, id)
        end
    else
        restore_X!(beads, idlist)
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
        #if pbc
        #    par.step = clamp(par.step, 0.001, 1.0)
        #else
            par.step = clamp(par.step, 0.001, 3.0)
        #end
    end
    #
    return 100*par.nacc/par.ntry
end


    
mutable struct Worm_Advance_Params
    ntry::Int
    nacc::Int
end
const DEFAULT_WORM_ADVANCE_PAR = Worm_Advance_Params(0,0)


#exp_dist = Exponential(0.5)
#expo =  Exponential(4.0)

function truncated_distribution(p, Kmax)
    #k = ceil(Int, rand(Truncated(exp_dist, 1, Kmax)))
    #return k
    #return ceil(Int, rand(expo))
             
    # CDF of Geometric(p) is F(k) = 1 - (1 - p)^k
    # inverse sampling:
    u = rand()
    q = 1 - p
    k = ceil(Int, log(1 - u * (1 - q^Kmax)) / log(q))
    return k
end

const xold_buffer = MVector{dim, Float64}(undef)

# ADVANCE
function worm_advance!(PIMC::t_pimc, beads::t_beads, links::t_links,
                       gauss::MVector{dim, Float64} = gauss_buffer,
                       idlist_buf::Vector{Int} = idlist_buffer,
                       xold::MVector{dim, Float64} = xold_buffer,
                       par::Worm_Advance_Params=DEFAULT_WORM_ADVANCE_PAR)
    """Advances worm head"""
    par.ntry += 1
    
    M = PIMC.M
    μ = PIMC.μ
    β = PIMC.β
    L = PIMC.L
    
    k = 1 # options: rand(1:worm_K) , truncated_distribution(0.5, M)

    
    t_H = beads.ts[PIMC.head]
    t_T = beads.ts[PIMC.tail]
    
    if PIMC.canonical
        # don't let head advance past tail (would lead to N->N+1)
        m = t_H
        @inbounds for i = 1:k
            m = mod1(m+1, M)
            if m==t_T
                # just drop
                return 100*par.nacc/par.ntry 
                # interpret as a worm_close attempt ?
            end
        end
    end
    
    # pick new head k slices above current Head (no links to follow)    
    idlist = @view idlist_buf[1:k+1] # from head to new head
    # 1     2    3    ... k   k+1
    # head  new  new  ... new new 
    new_bead_indices = 2:k+1
    
    Uold = 0.0
    m = t_H
    idlist[1] = PIMC.head
    @inbounds begin
        for i in new_bead_indices         
            m = mod1(m+1, M)
            b = beads.inactive_at_t[m][1]
            idlist[i] = b
            Uold += U_stored(PIMC, beads, b)         
        end
    end
    newH = idlist[end]    
    Δτ = wrapβ(beads.times[newH] - beads.times[PIMC.head], β)

    # sample new head from old head with free-particle Green's function rho_0   
    σ = sqrt(2λ*Δτ)
    # -----------------------------------------------------------------
    if pbc        
        @inbounds begin
            for d in 1:dim
                x = beads.X[d, PIMC.head] + σ*randn()
                x -= L * floor((x + L/2) / L)
                beads.X[d, newH] = x
            end
        end
    else
        @inbounds begin
            for d in 1:dim
                x = beads.X[d, PIMC.head] + σ*randn()
                beads.X[d, newH] = x
            end
        end
    end
    # -----------------------------------------------------------------
    
    if enforce_worm_limit(PIMC, beads, newH, PIMC.tail)
        #newH is still inactive, no need to reset it 
        return 100*par.nacc/par.ntry
    end
    
    # --------------------------------------------------
    generate_path!(PIMC, beads, links, idlist)
    # --------------------------------------------------
    
    Unew = 0.0
    activate_beads!(beads, view(idlist, new_bead_indices)) # beads must be active to get correct U     
    for id in idlist[new_bead_indices]
        Unew += U_update(PIMC, beads, xold, id, :add) # dummy old position
    end
    deactivate_beads!(beads, view(idlist, new_bead_indices)) # undo activation
    ΔU = Unew - Uold


    # Metropolis
    ΔS = ΔU + μ*Δτ
    accept = metro(ΔS)
    
    if accept
        par.nacc += 1        
        PIMC.head = newH        
        links.next[newH] = -1
        # activate new beads 
        activate_beads!(beads, view(idlist, new_bead_indices)) 
        for id in idlist[new_bead_indices]
            update_stored_slice_data(PIMC, beads, id)
        end
        
    else        
        # break link created by generate_path
        links.next[PIMC.head] = -1
    end
    
    return 100*par.nacc/par.ntry
end


mutable struct Worm_Recede_Params
    ntry::Int
    nacc::Int
end
const DEFAULT_WORM_RECEDE_PAR = Worm_Recede_Params(0,0)


# RECEDE
function worm_recede!(PIMC::t_pimc, beads::t_beads, links::t_links,
                      idlist_buf::Vector{Int}=idlist_buffer,
                      par::Worm_Recede_Params=DEFAULT_WORM_RECEDE_PAR)
    """Recedes worm head """
    par.ntry += 1
        
    M = PIMC.M   
    μ = PIMC.μ
    β = PIMC.β
    L = PIMC.L

    t_H = beads.ts[PIMC.head]
    t_T = beads.ts[PIMC.tail]
        
    k = 1 # options : rand(1:worm_K) , truncated_distribution(0.5, M) 
    

    if PIMC.canonical
        # don't let head recede past tail (would lead to N->N-1)
        m = t_H
        @inbounds for i = 1:k
            m = mod1(m-1, M)
            if m==t_T
                # just drop
                return 100*par.nacc/par.ntry 
                # treat as a worm_close attempt?
            end
        end
    end
    idlist = @view idlist_buf[1:k] # head, head-1, ..., head-k+1
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

    # deactivate temporarily
    deactivate_beads!(beads, idlist)
    Unew = 0.0
    for b in idlist
        Unew += U_update(PIMC, beads, view(beads.X, :, b), b, :remove)
    end
    # restore activation
    activate_beads!(beads, idlist)
        
    ΔU = Unew - Uold   
    Δτ = wrapβ(beads.times[PIMC.head] - beads.times[newH], β)
    
    # Metropolis
    ΔS = ΔU + μ*Δτ 
    accept = metro(ΔS)
    
    if accept
        par.nacc += 1       
        PIMC.head = newH
        links.next[PIMC.head] = -1
        # deactivate removed beads
        deactivate_beads!(beads, idlist)
        for b in idlist
            update_stored_slice_data(PIMC, beads, b)
        end
    end
    return 100*par.nacc/par.ntry
    
end

end
