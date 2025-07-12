__precompile__(false)
module PIMC_Reports
using Printf
using HDF5
using Dates

using PIMC_Common
using PIMC_Structs:t_beads, t_links, t_pimc, t_measurement, t_report
using QMC_Statistics

export t_report, report_acceptance
export pimc_report
export report_energy
export add_report!
export pimc_results_to_hdf5


function add_report!(PIMC::t_pimc, frequency::Int64, name::Symbol,  exe::Function)
    report = t_report(frequency=frequency, name=name, exe=exe)
    push!(PIMC.reports, report)
    println("Added screen report: ",name,",  frequency ",frequency)
end


function report_acceptance(PIMC::t_pimc)
    println("Acceptances:")
    for move in PIMC.moves
        if move.name==:worm_move
            @printf("%15s %8.3f %s\n", move.name, PIMC.acceptance[move], " (worm_open)")
        else
            @printf("%15s %8.3f\n", move.name, PIMC.acceptance[move])
        end
    end
end

function myprintf(x, y, f::IO=stdout)
    @printf(f, "# %s = %s\n", x, y)
end

function pimc_report(PIMC::t_pimc, io::IO=stdout)
    # parameter report
    if PIMC_Common.TEST
        println(io, "# "*"="^80)
        println(io, "# TEST CODE ! (Somewhere there's a PIMC_Common.TEST = true that marks this a test version)")
        println(io, "# "*"="^80)
    end
    println(io, "# "*"="^80)
    println(io, "# PIMC PARAMETERS")
    println(io, "# ---------------")
    
    myprintf("sys", PIMC_Common.sys, io)
    myprintf("bose", bose, io)
    myprintf("pbc", pbc, io)
    myprintf("T", 1/PIMC.β, io)
    myprintf("dim", dim, io)
    myprintf("λ", λ, io)
    myprintf("ρ", N/PIMC.L^dim, io)
        
    not_show = ["potential","measurements","moves","reports","ipimc","iworm","head","tail","acceptance"]
    for field in fieldnames(typeof(PIMC))
        any(occursin.(not_show, String(field))) && continue
        value = getproperty(PIMC, field)                
        myprintf(field, value, io)
    end
    
    myprintf("worm_C", worm_C, io)
    myprintf("worm_K" ,worm_K, io)
    myprintf("worm_limit" ,worm_limit, io)
    println(io, "# "*"="^80)
end

notdone = true
Eexact_dist = 0.0
Eexact_bose = 0.0


function report_energy(PIMC::t_pimc, meas::t_measurement)
    global notdone, Eexact_dist, Eexact_bose 
    β, M = PIMC.β, PIMC.M
   

    ave, std, input_σ2, Nblocks, blocks  = get_stats(meas.stat)
    
    # file output
    open(meas.filename, "a") do f
        if notdone
            # exact energies, if known, else 0.0
            Eexact_dist = PIMC_Common.system.E_exact(β, dim, N) 
            Eexact_bose = PIMC_Common.system.E_exact_bosons(β, dim, N)
            # write PIMC parameters to a header
            pimc_report(PIMC, f)
            notdone = false
        end
        print(f, ave[1]," ", std[1], " ", Eexact_dist, " ", Eexact_bose)
        print(f, " ", ave[2], " ", std[2], " ", ave[3], " ", std[3])
        # raw data, average over current block:
        for b in blocks
            print(f, " ", b)
        end        
        println(f, " ", M, " ", 1/β)
    end
    # set HDF4 data Dict
    results[string(meas.name)] = Dict(
        "E" => ave[1], 
        "std_E" => std[1],
        "T" => ave[2],
        "std_T" => std[2],
        "V" => ave[3],
        "std_V" => std[3]
    )
    

    # screen output
    @printf("%d %-8s <E> = %12.8f ± %-12.8f  <T> = %12.8f ± %-12.8f <V> =  %12.8f ± %-12.8f\n",
            PIMC.ipimc, meas.name,  ave[1], std[1], ave[2], std[2], ave[3], std[3])
   
    if meas.name == :E_vir # output only once per report      
        if !isapprox(Eexact_dist, 0.0) || !isapprox(Eexact_bose, 0.0)
            if PIMC_Common.sys == :Noninteracting
                @printf("   | GC  dist. <E> = %12.8f Bose   <E> = %12.8f\n", Eexact_dist, Eexact_bose)
            else
                @printf("   | Exact dist. <E> = %12.8f Bose   <E> = %12.8f\n", Eexact_dist, Eexact_bose)
            end
        end
    end
    
end




function pimc_results_to_hdf5(PIMC::t_pimc)
    """ PIMC results to an HDF5 file. """
    if PIMC.ipimc<0
        # nothing to save to hdf5
        return
    end
    println("** Writing HDF5 file **")
    fname = PIMC.hdf5_file 
    # Dict that holds PIMC results
    data = PIMC_Common.results
    
    # Metadata
    h5open(fname, "w") do f
            
        # git commit
        # metadata entry is a dictionary
        commit_hash = ""         
        try
            # not always part of git 
            commit_hash = readchomp(`git rev-parse HEAD`)
            #git_status = readchomp(`git status --porcelain`)
            #dirty = !isempty(git_status) # uncommitted edits?
        catch
            nothing
        end
        actionstr = PIMC_Common.action==ChinAction ? "chin" : "prim"
        data["metadata"] = Dict(
            "git_commit" => commit_hash,
            "date" => string(Dates.now()),
            "sys" => string(PIMC_Common.sys),
            "N" => N,
            "dim" => dim,
            "bose" => bose,
            "lambda" => λ, 
            "canonical" => PIMC.canonical,
            "pbc" => pbc,
            "T" => 1/PIMC.β,
            "beta" => PIMC.β,
            "tau" => PIMC.τ,          
            "M" => PIMC.M,
            "L" => PIMC.L,           
            "mu" => PIMC.μ,                                
            "worm_C" => worm_C,
            "worm_K" => worm_K,
            "worm_limit" => worm_limit,
            "action" => actionstr,
            "chin_a1" => PIMC.chin_a1,
            "chin_t0" => PIMC.chin_t0
        )

                
        # HDF5 can't store a heterogeneous Dict at once
        g = create_group(f, "metadata")
        for (k, v) in data["metadata"]
            g[k] = v  
        end

        # results
       
        for meas in PIMC.measurements
            namestr = string(meas.name)
            if haskey(data, namestr)        
                g = create_group(f, namestr)
                for (k, v) in data[namestr]
                    g[k] = v  
                end
            end
        end
        
       
    end
    println("done")
end



end
