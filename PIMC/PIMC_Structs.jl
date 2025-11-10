__precompile__(false)
module PIMC_Structs

using Printf
using Random
using LinearAlgebra: BLAS, ⋅
BLAS.set_num_threads(1) 
using JLD2 # binary restart files
using BenchmarkTools
using InteractiveUtils

# local modules:
push!(LOAD_PATH,".")
using QMC_Statistics: t_Stat, get_stats
using PIMC_Systems
using PIMC_Common


export t_pimc, t_links, t_beads, t_move, t_measurement
export init_pimc, run_pimc, check_links
export write_restart, read_restart!
export has_potential

@inline norm(x) = sqrt(sum(abs2,x))

struct t_move
    frequency::Int64
    name::Symbol
    exe::Function
    ntries::Int64
    nacc::Int64
    # constructor
    function t_move(; frequency=1, name=name, exe=exe, ntries=0, nacc=0)
        return new(frequency, name, exe, ntries, nacc)
    end
end

mutable struct t_measurement
    frequency::Int64
    name::Symbol
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



# For non-existing confinement or pair potential cases
struct NoPotential end
(::NoPotential)(x...) = 0.0
(::NoPotential)(x::AbstractVector) = zeros(eltype(x), length(x))
(::NoPotential)(x::Real) = 0.0

# traits; multiple dispatch chooses based on argument type, most specialized match wins
has_potential(::NoPotential) = false   
has_potential(::Any) = true    

# parametric type
mutable struct t_pimc{A<:PIMC_Common.action,
                      Fpair, Fconf, Fder, Fder2, Fgrad, Fgrad2}
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
    # Chin action parameters:
    chin_a1::Float64
    chin_t0::Float64
    # potentials 
    pair_potential::Fpair
    confinement_potential::Fconf
    der_pair_potential::Fder
    der2_pair_potential::Fder2
    grad_confinement_potential::Fgrad
    grad2_confinement_potential::Fgrad2
        
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
                   canonical::Bool, β::Float64,  L::Float64,
                   pair_potential::Union{Function, Nothing} = nothing,
                   confinement_potential::Union{Function, Nothing} = nothing,
                   der_pair_potential::Union{Function, Nothing} = nothing,
                   der2_pair_potential::Union{Function, Nothing} = nothing,
                   grad_confinement_potential::Union{Function, Nothing} = nothing,
                   grad2_confinement_potential::Union{Function, Nothing} = nothing
                   ) 
    pair_potential              = pair_potential            === nothing ? NoPotential() : pair_potential
    confinement_potential       = confinement_potential     === nothing ? NoPotential() : confinement_potential
    der_pair_potential          = der_pair_potential        === nothing ? NoPotential() : der_pair_potential
    der2_pair_potential         = der2_pair_potential       === nothing ? NoPotential() : der2_pair_potential
    grad_confinement_potential  = grad_confinement_potential === nothing ? NoPotential() : grad_confinement_potential
    grad2_confinement_potential = grad2_confinement_potential === nothing ? NoPotential() : grad2_confinement_potential

        
    # the actual types of potentials
    Fpair  = typeof(pair_potential)
    Fconf  = typeof(confinement_potential)
    Fder   = typeof(der_pair_potential)
    Fder2  = typeof(der2_pair_potential)
    Fgrad  = typeof(grad_confinement_potential)
    Fgrad2 = typeof(grad2_confinement_potential)

    # simulations box
    if pbc && isapprox(L,0.0)
        error("Need L>0 for PBC")
    end
    @show confinement_potential
    if pbc && confinement_potential != NoPotential()
        error("Can't use both PBC and confinement potential")
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
    
    active = falses(Nbeads)
    # allocate beads and their backup
    beads = t_beads(Matrix{Float64}(undef, dim, Nbeads), ids, ts, at_t, times, active)
    beads_backup = deepcopy(beads) # deepcopy to get independent array memory
    
    # activate N beads for each slice
    for m in 1:M
        bs = beads.at_t[m][1:N]
        active[bs] .= true
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
        for id1 in bs
            id = id1            
            while true
                @inbounds beads.X[:, id] .= pos[:,i] 
                id = links.next[id]
                id == id1 && break
            end
            i += 1
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
    PIMC = t_pimc{PIMC_Common.action, Fpair, Fconf, Fder, Fder2, Fgrad, Fgrad2}(
        canonical, M, β, τ, L, μ, ipimc, iworm, head, tail, swapcount,
        chin_a1, chin_t0,
        pair_potential,
        confinement_potential,
        der_pair_potential,
        der2_pair_potential,
        grad_confinement_potential,
        grad2_confinement_potential,
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

    
function find_measurement(PIMC::t_pimc, target_name)
    """find a measurement by name"""
    for meas in PIMC.measurements
        if meas.name == target_name
            return meas
        end
    end
    return nothing
end
         

function run_pimc(PIMC::t_pimc, beads::t_beads, links::t_links, beads_backup::t_beads, links_backup::t_links, limit::Int=-1)
    """Runs the PIMC simulation"""
    moves = []
    move_name_to_num = Dict()
    for (i, move) in enumerate(PIMC.moves)
        for rep in 1:move.frequency
            push!(moves, move)
        end
        move_name_to_num[move.name] = i
    end
    
    meas_name_to_num = Dict()
    for (i, meas) in enumerate(PIMC.measurements)
        meas_name_to_num[meas.name] = i
    end
    
    
    move_timers = zeros(length(PIMC.moves))
    meas_timers = zeros(length(PIMC.measurements))
    move_counts = zeros(Int64, length(PIMC.moves))
    meas_counts = zeros(Int64, length(PIMC.measurements))

    # move open-worm measurements to own list    
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
    
    start_time = time()
    while true
        PIMC.ipimc += 1
        move = rand(moves)
        movenum = move_name_to_num[move.name]
        if PIMC.canonical && count(beads.active) != N*PIMC.M
            @show count(beads.active), N*PIMC.M
            error("wrong number of active beads (usually mistake in worm moves)")
        end
        t = @elapsed begin
            if move.name == :worm_move
                # special
                PIMC.acceptance[move] = move.exe(PIMC, beads, links, beads_backup, links_backup, worm_measurements)
            else
                PIMC.acceptance[move] = move.exe(PIMC, beads, links)
            end
        end
        
        move_counts[move_name_to_num[move.name]] += 1
        if move_counts[move_name_to_num[move.name]] > 1
            # skip compilation call
            move_timers[move_name_to_num[move.name]] += t
        end

        
        if PIMC.ipimc%1000==0
            println(" ")            
            tsum = sum(move_timers)
            t = 0.0
            for move in PIMC.moves
                i = move_name_to_num[move.name]                
                @printf("timings: %15s %10.3f seconds %10.1f %% \n", move.name,
                        move_timers[i]/move_counts[i], move_timers[i]/tsum*100.0)
                t += move_timers[i]
            end
            println(" ")
            runtime =  time()-start_time
            @printf("total        move time = %-15.1f seconds\n",t)
            @printf("total measurement time = %-15.1f seconds\n",runtime-t)            
            @printf("total         run time = %-15.1f seconds\n",runtime)
        end
       
        
        if PIMC.ipimc<0 && PIMC.ipimc%(Ntherm/10)==0
            println("="^80)
            @printf "Therm ipimc = %s out of %s\n" PIMC.ipimc Ntherm 
        end

        # report on screen
        for report in PIMC.reports
            if PIMC.ipimc%report.frequency==0
                report.exe(PIMC)
            end
        end
        
        if limit>0 && abs(PIMC.ipimc)>limit
            # profiling
            return
        end
        
        PIMC.ipimc < 0 && continue # in thermalization

         
        #
        # Measuring phase
        # 
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
        end
        

               

        if PIMC.ipimc%10000==0
            write_restart(PIMC, beads, links)            
            #GC.gc() # manual garbage collection
        end
       
        for meas in setdiff(PIMC.measurements, worm_measurements)
            if PIMC.ipimc%meas.frequency==0                
                t = @elapsed begin
                    meas.exe(PIMC, beads, links, meas)
                end
                
                    
                meas_counts[meas_name_to_num[meas.name]] += 1
                if meas_counts[meas_name_to_num[meas.name]]>1
                    # skip compilation call
                    meas_timers[meas_name_to_num[meas.name]] += t
                    # if timing, comment out saving restart, or you get a different result every time
                    #if meas.name==:E_vir                    
                    #    @show t
                    #    exit()
                    #end
                end
            end
        end
        if PIMC.ipimc>0 && PIMC.ipimc%1000==0
            println("")
            println("Cumulative Measurement timings (not worm measurements):")            
            tsum = sum(meas_timers)
            for meas in setdiff(PIMC.measurements, worm_measurements)
                i = meas_name_to_num[meas.name]                
                @printf("timings: %30s %15.2f seconds %10.1f %% \n", meas.name,
                        meas_timers[i], meas_timers[i]/tsum*100.0)
            end
            println("")
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
    println("writing restart file ...")
    @save PIMC.restart_file PIMC beads links worm_C=PIMC_Common.worm_C  worm_K=PIMC_Common.worm_K worm_limit=PIMC_Common.worm_limit
    println("done")
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
        for meas in PIMC.measurements
            if meas.name==:obdm
                println("ignoring obdm in restart, starting new data collection")
                meas.stat.nblocks = 0
                meas.stat.sample.data .= 0
                meas.stat.sample.data2 .= 0
                meas.stat.sample.input_σ2 = 0.0
            end
            break
        end
    else
        println("Using default chin_a1 and chin_to, ignoring values in restart file")     
        PIMC.chin_a1 = 0.33
        PIMC.chin_t0 = 0.1215
    end

    beads.X .= data["beads"].X
    beads.active .= data["beads"].active
    links.next .= data["links"].next
    links.prev .= data["links"].prev
end

end

