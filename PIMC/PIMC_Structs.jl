__precompile__(false)
module PIMC_Structs

using Printf
using Random
using LinearAlgebra: BLAS, ⋅, norm
BLAS.set_num_threads(1) 
using JLD2 # binary restart files
using BenchmarkTools
using InteractiveUtils
using Distributions: Categorical

using TimerOutputs
const to = TimerOutput()

# local modules:
push!(LOAD_PATH,".")
using QMC_Statistics: t_Stat, get_stats
using PIMC_Systems
using PIMC_Common

export t_pimc, t_links, t_beads, t_move, t_measurement
export init_pimc, run_pimc, check_links
export write_restart, read_restart!



mutable struct t_move
    frequency::Int64
    name::Symbol
    sname::String
    exe::Function
    ntries::Int64
    nacc::Int64
    acc::Float64
    # constructor
    function t_move(; frequency=1, name=name, sname=sname, exe=exe, ntries=0, nacc=0, acc=0.0)
        return new(frequency, name, sname, exe, ntries, nacc, acc)
    end
end

mutable struct t_measurement
    frequency::Int64
    name::Symbol
    sname::String
    exe::Function
    stat::t_Stat
    filename::String
end

struct t_beads
    X::Matrix{Float64} # bead positions (dim, bead id)
    ids::Vector{Int64} # bead ids
    ts::Vector{Int64}   # mapping bead id -> time slice
    at_t::Dict{Int64, Vector{Int64}}  # mapping bead time slice ->  list of id's
    times::Vector{Float64} # mapping bead id -> imaginary time
    active::BitVector
    active_at_t::Vector{Vector{Int}}
    inactive_at_t::Vector{Vector{Int}}
end
      

struct t_links
    next::Vector{Int64}
    prev::Vector{Int64}
end



struct t_report
    frequency::Int64
    name::Symbol
    exe::Function
    function t_report(;frequency=frequency, name=name, exe=exe)
        return new(frequency, name, exe)
    end
end



# parametric type
mutable struct t_pimc{SP<:SystemPotentials}
    canonical::Bool                  # Canonical or not, not means grand canonical
    M::Int64                         # number of time slices    
    β::Float64                       # inverse temperature
    τ::Float64                       # imaginary time step
    L::Float64                       # box size in PBC
    μ::Float64                       # chemical potential, μ=0 in canonical
    ipimc::Int64                     # PIMC step counter
    iworm::Int64                     # worm step counter
    head::Int64                      # worm head bead id
    tail::Int64                      #   ...     tail id
    swapcount::Int64
    rc::Float64                      # pair potential cutoff in neighborlist 
    # Chin action parameters:
    chin_a1::Float64
    chin_t0::Float64
    # potentials
    syspots::SP #SystemPotentials
    #
    restart_file::String
    hdf5_file::String
    filesuffix::String
    # reported values
    acceptance ::Dict #{String, Float64} 
    # tasks to do now and then
    measurements::Vector{t_measurement}
    moves::Vector{t_move}
    reports::Vector{t_report}
end

function init_pimc(; M::Int64,
                   canonical::Bool, β::Float64,  L::Float64, rc::Float64,
                   syspots::SystemPotentials
                   )

    
    # simulations box
    if pbc && isapprox(L, 0.0)
        error("Need L>0 for PBC")
    end
   
    
    # Bead positions
    # 
    # reserve some extra space (needed in canonical obdm measurement and always in grand canonical)    
    Nbeads = N_slice_max*M
    # bead ids
    ids = collect(1:Nbeads) 
    # bead id -> time slice mapping    
    ts = [(id - 1) ÷ N_slice_max + 1 for id in ids]
    # dictionary time slice -> bead ids on that time slice
    at_t = Dict{Int64, Vector{Int64}}()
    for t in 1:M
        id_list = Int64[]
        for id in ids
            if ts[id] == t
                push!(id_list, id)
            end
        end
        at_t[t] = id_list
    end
    
    

    # bead id -> imaginary time mapping (τ units), set in PIMC_main calling init_action! 
    times = Array{Float64}(undef, length(ids))
    
    if PIMC_Common.action == PrimitiveAction
        τ = β/M
    elseif PIMC_Common.action  == ChinAction
        τ = 3β/M
    else
        error("unknown action")
    end
    @show PIMC_Common.action, τ
    

    # allocate beads and their backup
    beads = t_beads(Matrix{Float64}(undef, dim, Nbeads), ids, ts, at_t, times,
                    falses(Nbeads), Vector{Vector{Int}}(undef, M), Vector{Vector{Int}}(undef, M))
    beads_backup = deepcopy(beads) # deepcopy to get independent array memory
    
    
    
    # activate N beads for each slice
    for m in 1:M
        bs = beads.at_t[m][1:N]
        beads.active[bs] .= true # bitvector
        # list active and inactive beads on each slice
        beads.active_at_t[m] = Int[]
        beads.inactive_at_t[m] = Int[]
        beads_backup.active_at_t[m] = Int[]
        beads_backup.inactive_at_t[m] = Int[]
        for b in beads.at_t[m]
            if beads.active[b]
                push!(beads.active_at_t[m], b)
                push!(beads_backup.active_at_t[m], b)
            else
                push!(beads.inactive_at_t[m], b)
                push!(beads_backup.inactive_at_t[m], b)
            end
        end        
    end
        
    
    # bead <-> bead links (whether active or not)   
    next  = zeros(Int64, Nbeads)  # id of next bead 
    prev  = zeros(Int64, Nbeads)  # id of previous bead
    for t in 1:M        
        t_next = mod1(t+1, M)
        bs_t = beads.at_t[t]
        bs_t_next = beads.at_t[t_next]
        for (id, id_next) in zip(bs_t, bs_t_next)
            next[id] = id_next
            prev[id_next] = id
        end
    end
    # make link struct and its backup
    links = t_links(next, prev)
    links_backup = deepcopy(links)
    
    
    
    if pbc
        println("grid position init for $N particles")
        pos = grid_points(N, dim, L, pbc) # coordinates in box [0,         
        pos .+= 1.5 .* (rand(dim,N) .- 0.5) # small random shift
        # move coordinates to box [-L/2, L/2]
        for i in 1:N
            periodic!(view(pos, :, i), L)
        end
        # follow links so that world lines are (almost) straight 
        bs = beads.at_t[1][1:N]  # start links from N beads at slice 1
        i = 1 # particle i world line
        @inbounds begin
            for id1 in bs
                id = id1            
                while true
                    beads.X[:, id] .= pos[:,i]
                    # just to be sure:
                    periodic!(view(beads.X, :, id), L)
                    id = links.next[id]
                    id == id1 && break
                end
                i += 1
            end
        end        
    else
        λ = 0.5 # just for this init
        for id in beads.ids[beads.active]
            beads.X[:, id] .= randn(dim) * sqrt(2*λ*τ) # gaussian distribution
        end       
    end

    
    ipimc = 0
    iworm = 0
    head = -1
    tail = -1
    swapcount = 0
    # 
    μ = 0.0
    chin_a1 = 0.33 
    chin_t0 = 0.1215   
    restart_file = ""
    hdf5_file = ""
    filesuffix = "" 
    acceptance = Dict{t_move, Float64}()
    measurements = t_measurement[]
    moves = t_move[]
    reports = t_report[]

  
    
    # can use only positional arguments here
    PIMC = t_pimc{typeof(syspots)}(
        canonical, M, β, τ, L, μ, ipimc, iworm, head, tail, swapcount,
        rc, 
        chin_a1, chin_t0,
        syspots,
        restart_file,
        hdf5_file,
        filesuffix,
        acceptance,
        measurements,
        moves,
        reports
    )

    check_links(PIMC, beads, links)

    return PIMC, beads, links, beads_backup, links_backup
end

         

function run_pimc(PIMC::t_pimc, beads::t_beads, links::t_links, beads_backup::t_beads, links_backup::t_links, limit::Int=-1)
    """Runs the PIMC simulation (stops after limit steps if set, else eternally)"""
    
    # Moves run in Z-section, where a worm is not open
    moves = t_move[]
    probs = Float64[]
    for move in PIMC.moves
        if move.name !== :worm_move
            if contains(move.sname, "worm")
                continue
            end
        end
        push!(moves, move)
        push!(probs, move.frequency)
    end
    sum_probs = sum(probs)
    probs ./= sum_probs  
    distrib = Categorical(probs)
    
    # Moves run in C-section, where a worm is open
    worm_moves = t_move[]
    probs = Float64[]
    for move in PIMC.moves
        move.name == :worm_move && continue
        move.name == :worm_open && continue
        if !contains(move.sname, "worm")
            continue
        end
        push!(worm_moves, move)
        push!(probs, move.frequency)
    end
    sum_probs = sum(probs)
    probs ./= sum_probs  
    worm_distrib = Categorical(probs)
          
    # copy open-worm measurements to own list    
    worm_measurements = Vector{t_measurement}(undef, 0)     
    for meas in PIMC.measurements    
        if meas.name in [:head_tail_histogram, :obdm]
            push!(worm_measurements, meas)            
        end
    end
    
        
    PIMC.ipimc = -Ntherm # thermalization is negative ipimc
   
    for move in PIMC.moves
        PIMC.acceptance[move] = 0.0
    end
    
    while true
        PIMC.ipimc += 1
        move = moves[rand(distrib)]
        
        # sanity check
        if PIMC.canonical && count(beads.active) != N*PIMC.M
            @show count(beads.active), N*PIMC.M
            error("wrong number of active beads (usually mistake in worm moves)")
        end

        if move.name == :worm_move
            # special                    
            PIMC.acceptance[move] = move.exe(PIMC, beads, links, beads_backup, links_backup,
                                             worm_measurements, worm_moves, worm_distrib, to)
        else
            @timeit to "moves" begin
                @timeit to move.sname begin
                    PIMC.acceptance[move] = move.exe(PIMC, beads, links)
                end
            end
        end

        # timing output
        if PIMC.ipimc%10000==0
            println(); show(to); println()
        end
               
        
        # thermalization frequent progress print
        if PIMC.ipimc<0 && PIMC.ipimc%(Ntherm/10)==0
            println("="^80)
            @printf "Therm ipimc = %s out of %s\n" PIMC.ipimc Ntherm 
        end

        # screen reports 
        for report in PIMC.reports
            if PIMC.ipimc%report.frequency==0
                report.exe(PIMC)
            end
        end
        
        if limit>0 && abs(PIMC.ipimc)>limit
            # profiling or testing
            return
        end
        
        PIMC.ipimc < 0 && continue # in thermalization

        if PIMC.ipimc==0
            println("="^80)
            println("="^80)
            println("=== Thermalization ends ===")
            println("="^80)
            println("="^80)
            # Zero counters when thermalization is done
            # Important for normalization of the one-body density matrix     
            worm_stats.N_open_try = 0
            worm_stats.N_open_acc = 0
            worm_stats.N_close_try = 0
            worm_stats.N_close_acc = 0
            return
        end

        if PIMC.ipimc%20000==0 
            write_restart(PIMC, beads, links)            
            #GC.gc() # manual garbage collection
        end
        #
        # Measuring phase
        #
        for meas in setdiff(PIMC.measurements, worm_measurements)
            if PIMC.ipimc%meas.frequency==0
                @timeit to "measurements" begin
                    @timeit to meas.sname begin                      
                        meas.exe(PIMC, beads, links, meas)
                    end
                end
            end
        end
    end
    
    
end

function check_links(PIMC::t_pimc, beads::t_beads, links::t_links)
    """Checks for broken links indicating a programming mistake"""
    next = links.next
    prev = links.prev
    head = 0
    tail = 0
    for id in beads.ids[beads.active]
        next_id  = next[id]        
        if next_id == -1
            if head != 0
                error("two heads!")
            end
            head=id
            if links.next[head]!=-1
                @show links.next[head]
                error("Head next is not -1")
            end
        elseif prev[next_id] != id
            @show beads.ts[id],  beads.ts[next_id]
            @show prev[next_id] , id, next_id
            error("link $id <- $next_id is broken")
        end
        for b in beads.ids[beads.active]
            if prev[b] == id && b != next_id
                error("2-to-1 intersection, both $b and $prev_id have bead $id as prev")
            end
        end
        
        prev_id = links.prev[id]        
        if prev_id == -1
            if tail != 0
                error("two tails!")
            end
            tail=id
            if links.prev[tail]!=-1
                @show links.prev[tail]
                error("Tail prev is not -1")
            end
        elseif next[prev_id] != id
            @show beads.ts[prev_id],  beads.ts[id]
            error("link $prev_id -> $id is broken")
        end
        for b in beads.ids[beads.active]
            if next[b] == id && b != prev_id
                error("both $b and $prev_id have bead $id as next")
            end
        end
    end
end


function grid_points(N::Int64, dim::Int64, L::Float64, pbc::Bool)
    """points on equally spaced grid in dim dimensions"""
    nbox  = floor(Int64, N^(1/dim)) # number of one-particle boxes
    nbox^dim < N && (nbox += 1)
    ii = ones(dim)
    
    X = Matrix{Float64}(undef, dim, N)
    for i in 1: N
        X[:, i] = (ii .- 0.5) .* L/nbox  
        i==N && break
        ii[1] += 1
        if ii[1] > nbox
            ii[1] = 1
            ii[2] += 1
            if ii[2] > nbox                  
                ii[2] = 1
                ii[3] += 1                   
            end
        end
    end  
    rmin = 1e6
    for i in 1: N
        for j in i+1:N
            rij = dist(view(X, :, i), view(X, :, j), L)
            rmin = min(rij, rmin)
        end
    end
    @show rmin
    return X
end


function write_restart(PIMC::t_pimc, beads::t_beads, links::t_links)
    if PIMC_Common.TEST 
        println("TEST, no restart file written")
        return
    end
    println("writing restart file")
    @save PIMC.restart_file PIMC beads links worm_C=PIMC_Common.worm_C  worm_K=PIMC_Common.worm_K worm_limit=PIMC_Common.worm_limit
end

function read_restart!(PIMC::t_pimc, beads::t_beads, links::t_links)
    
    if !isfile(PIMC.restart_file)
        println("no restart file $(PIMC.restart_file)")
        error("restart failed")
    end
    data = load(PIMC.restart_file)

    if PIMC.canonical != data["PIMC"].canonical
        error("Restart file is not canonical = $(PIMC.canonical)")
    end
    try
        # read worm parameters - important 
        # old restart files may not have'm
        PIMC_Common.worm_C = data["worm_C"]
        PIMC_Common.worm_K = data["worm_K"]
        PIMC_Common.worm_limit = data["worm_limit"]
    catch
        println("no worm_C etc. worm parameters in restart file, consider optimization")
    end
    if restart_with_data
        # continue measurements
        println("Restarting with old measured data")
        @load PIMC.restart_file PIMC # this loads the *whole* PIMC structure
        #
    end

    beads.X .= data["beads"].X
    beads.active .= data["beads"].active    
    links.next .= data["links"].next
    links.prev .= data["links"].prev
    # reconstruct active and inactive bead vectors, don't try to use the one in backup file
    @inbounds for m in 1:PIMC.M
        w      = beads.active_at_t[m]
        v      = beads.inactive_at_t[m]
        empty!(w)
        empty!(v)
        @inbounds for b in beads.at_t[m]
            if beads.active[b]
                push!(w, b)
            else
                push!(v, b)
            end
        end
    end
end

end

