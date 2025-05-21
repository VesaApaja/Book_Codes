__precompile__(false)
module PIMC_Structs

using Printf
using Random
using LinearAlgebra: BLAS, ⋅
BLAS.set_num_threads(1) 
using JLD2 # binary restart files
    
# local modules:
push!(LOAD_PATH,".")
using QMC_Statistics: t_Stat, get_stats
using PIMC_Systems
using PIMC_Common


export t_pimc, t_links, t_beads, t_move, t_measurement
export init_pimc, run_pimc, check_links
export read_restart!

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
    X::Matrix{Float64} # unwrapped bead positions (dim, bead id)
    ids::Vector{Int64} # bead ids
    ts::Vector{Int64}   # mapping bead id -> time slice
    at_t::Dict{Int64, Vector{Int64}}  # mapping bead time slice ->  list of id's
    times::Vector{Float64} # mapping bead id -> imaginary time
    active::BitVector
end
      

struct t_links
    next::Array{Int64}
    prev::Array{Int64}
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
mutable struct t_pimc{A<:PIMC_Common.action}   
    canonical::Bool                  # Canonical or not, not means grand canonical
    M::Int64                           # number of time slices    
    β::Float64                       # inverse temperature
    τ::Float64                       # imaginary time step
    L::Float64                       # box size in PBC
    μ::Float64                       # chemical potential, μ=0 in canonical
    pair_potential::Union{Function, Nothing}        # pair potential V(r_ij)
    confinement_potential::Union{Function, Nothing} # confinement V_ext(vecr)
    der_pair_potential::Union{Function, Nothing}  # ∇_i V(r_ij)
    grad_confinement_potential::Union{Function, Nothing} # ∇_i V_ext(vecr)
    ipimc::Int64                     # PIMC step counter
    iworm::Int64                     # worm step counter
    head::Int64                      # worm head bead id
    tail::Int64                      #   ...     tail id  
    restart_file::String
    hdf5_file::String
    # reported values
    acceptance ::Dict #{String, Float64} 
    # tasks to do now and then
    measurements::Array{t_measurement,1}
    moves::Array{t_move,1}
    reports::Array{t_report,1}
end

function t_pimc{A}(; canonical=true,
                M=10,
                β=1.0,
                τ=0.1, L=1.0, μ=0.0,
                ipimc=0, iworm=0, head=-1, tail=-1,
                restart_file = "",
                hdf5_file = "",
                pair_potential=nothing,
                confinement_potential=nothing,
                der_pair_potential=nothing,
                grad_confinement_potential=nothing,
                acceptance=Dict(),
                measurements=[],
                moves=[],
                reports=[]) where A<:PIMC_Common.AbstractAction
    return t_pimc{A}(canonical,
                     M, β, τ, L, μ,
                     pair_potential, confinement_potential,
                     der_pair_potential, grad_confinement_potential,
                     ipimc, iworm, head, tail, restart_file, hdf5_file,
                     acceptance,
                     measurements, moves, reports
                     )
end


function init_pimc(; M::Int64,
                   canonical::Bool, β::Float64,  L::Float64,                   
                   pair_potential::Union{Function, Nothing}=nothing,
                   confinement_potential::Union{Function, Nothing}=nothing,
                   der_pair_potential::Union{Function, Nothing}=nothing,
                   grad_confinement_potential::Union{Function, Nothing}=nothing,                   
                   ) 
            
    

    if pbc && isapprox(L,0.0)
        error("Need L>0 for PBC")
    end
    if pbc && confinement_potential != nothing
        error("Can't use both PBC and confinement potential")
    end
    
    # Bead positions
    # TODO: reserve some extra space if not canonical   
    # bead ids
    ids = collect(1:N*M)
    # bead id -> time slice mapping
    ts = [(id - 1) ÷ N + 1 for id in ids]
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

    # mark all beads active 
    active = trues(length(ids))
    # allocate beads and their backup
    beads = t_beads(Matrix{Float64}(undef, dim, N*M), ids, ts, at_t, times, active)
    # active must be copied, else beads.active is beads_backup.active
    beads_backup = t_beads(Matrix{Float64}(undef, dim, N*M), ids, ts, at_t, times, copy(active))

    # bead-bead links
    
    next  = zeros(Int64, N*M)  # id of next bead 
    prev  = zeros(Int64, N*M)  # id of previous bead
    active = findall(beads.active)
    for id in active

        t = beads.ts[id]
        t_next = mod1(t+1, M)
        # find active bead not in next 
        id_nexts = setdiff(active, next)
        # find a bead at t_next 
        ind = findfirst(id -> beads.ts[id] == t_next, id_nexts)
        id_next = id_nexts[ind]
        next[id] = id_next
    end
    
    for id in active
        prev[next[id]] = id
    end
    # links and their backup
    links = t_links(next, prev)
    links_backup = t_links(copy(next), copy(prev))
    
    if pbc
        println("grid position init")
        pos = grid_points(N, dim, L, pbc) # coordinates in pbc box [0, L]
        # follow links so that world lines are (almost) straight 
        bs = beads.at_t[1] # start links from these
        i = 1 # particle i world line

        for id1 in bs
            id = id1
            
            while true
                beads.X[:, id] .= pos[:,i]  # .+ 0.05*(rand(dim).-0.5) # add randomness
                periodic!(view(beads.X, :, id), L)
                id = links.next[id]
                id == id1 && break
            end
            i += 1
        end
    else
        λ = 0.5 # just for this init
        active = findall(beads.active)
        for id in active
            beads.X[:, id] .= randn(dim) * sqrt(2*λ*τ) # gaussian distribution
        end
     
    end

    @show PIMC_Common.action
    
    
    PIMC = t_pimc{PIMC_Common.action}(M=M, β=β, τ=τ, L=L, ipimc=0, iworm=0, head=-1, tail=-1,
                  pair_potential=pair_potential,
                  confinement_potential=confinement_potential,
                  der_pair_potential=der_pair_potential,
                  grad_confinement_potential=grad_confinement_potential                  
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
    
    count_timers = zeros(Int64, length(PIMC.moves))
    timers = zeros(length(PIMC.moves))
       

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
    
    
    while true
        PIMC.ipimc += 1
        move = rand(moves)
        movenum = move_name_to_num[move.name]
        if PIMC.canonical && count(beads.active) != N*PIMC.M
            @show count(beads.active), N*PIMC.M
            error("wrong number of beads, usually mistake in worm moves")
        end
        t = @elapsed begin
            if move.name == :worm_move
                # special
                PIMC.acceptance[move] = move.exe(PIMC, beads, links, beads_backup, links_backup, worm_measurements)
            else
                PIMC.acceptance[move] = move.exe(PIMC, beads, links)
            end
        end
        count_timers[move_name_to_num[move.name]] += 1
        timers[move_name_to_num[move.name]] += t

        
        
        if PIMC.ipimc%10000==0
            println(" ")            
            tsum = sum(timers)
            for move in PIMC.moves
                i = move_name_to_num[move.name]                
                @printf("timings: %15s %10.3f %% \n", move.name, timers[i]/tsum*100.0)
            end
            println(" ")
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
        PIMC.ipimc==0 && println("thermalization ends")
       

        if PIMC.ipimc%10000==0 && PIMC_Common.sys != :HarmonicOscillator
            # don't bother to write restart for HO tests
            print("writing restart file ... ")
            write_restart(PIMC, beads, links)            
            println("done")
            #GC.gc() # manual garbage collection
        end
        
        
        # in measuring phase
        for meas in setdiff(PIMC.measurements, worm_measurements)
            if PIMC.ipimc%meas.frequency==0
                meas.exe(PIMC, beads, links, meas)                
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
    active = findall(beads.active)
    for id in active
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
    end
   
end


function grid_points(N::Int64, dim::Int64, L::Float64, pbc::Bool)
    """points on roughly equally spaced grid in dim dimensions"""
    nbox  = floor(Int64, N^(1/dim)) # number of one-particle boxes
    nbox^dim < N && (nbox += 1)
    ii = ones(dim)
    
    X = Matrix{Float64}(undef, dim, N)
    for i in 1: N
        X[:, i] = (ii .- 0.5) .* L/nbox  .+ 1.5.*(rand(dim) .- 0.5) # small random shift
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
    # scale to middle of the box
    #X .*= 0.9

   
    
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
    @save PIMC.restart_file PIMC beads links
end

function read_restart!(PIMC::t_pimc, beads::t_beads, links::t_links)
    # @load "restart.tmp"  PIMC beads links
    # would replace existing structures, I don't want old measument data read
    data = load(PIMC.restart_file)
    not_load = ["measurements", "moves", "potential", "ipimc", "iworm"]
    for field in fieldnames(typeof(PIMC))
        any(occursin.(not_load, String(field))) && continue
        println("loading $field")
        setproperty(PIMC, field) =  data["PIMC"].field       
        #PIMC.field .= data["PIMC"].field
    end
    beads.X .= data["beads"].X
    beads.active .= data["beads"].active
    links.next .= data["links"].next
    links.prev .= data["links"].prev
end

end

