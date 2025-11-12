__precompile__(false)
#
# PIMC main
#
using Random, LinearAlgebra, Statistics
using Printf
using DelimitedFiles
using Profile
using Distributions: Categorical

# local modules:
push!(LOAD_PATH,".")
using PIMC_Utilities
using PIMC_Common
using PIMC_Structs
using QMC_Statistics
using PIMC_Moves
using PIMC_Measurements
using PIMC_Reports

using PIMC_Action_Interface

#PIMC_Common.TEST=true
#Random.seed!(123456) #TESTING

# ==================================

function init_Mβ!(β::Float64)
    # find best M and β close to input β so that τ is fixed
    
    if PIMC_Common.action <: PrimitiveAction
        Ms = collect(1:1000)
        # β = M*τ        
        βs = Ms*PIMC_Common.τ        
    elseif PIMC_Common.action <: ChinAction
        Ms = collect(3:3:1000)
        # β = M*τ/3
        βs = Ms*PIMC_Common.τ/3
    end
    d, index = findmin(abs.(βs .- β))
    β = βs[index]  
    M = Ms[index]
    if d>1e-6
        println("Note: using nearest temperature T=$(1/β) with M=$M that gives τ = $(PIMC_Common.τ)")
    end
    if M>M_max
        println("M = $M, but code assumes M_max=$M_max")
        error("M > M_max") 
    end
    
    return M, β
end



function init_harmonic_oscillator(; β::Float64=1.0, M::Int64)
    
    PIMC, beads, links, beads_backup, links_backup = init_pimc( M=M,
                                                                β=β,                                   
                                                                L=1e6,  
                                                                canonical=true,
                                                                confinement_potential=PIMC_Common.system.potential,
                                                                grad_confinement_potential=PIMC_Common.system.grad_potential,
                                                                grad2_confinement_potential=PIMC_Common.system.grad2_potential
                                                                )
    

   
    
    return PIMC, beads, links, beads_backup, links_backup
end

function init_noninteracting(;β::Float64=1.0, M::Int64)
    
    # Although the free-particle Green's functions are sampled exactly, a large τ
    # and large thermal wavelength is a bad combination!
   
    

    if dim==3
        #  
        #  BEC temperature in GC ensemble (3D)
        # T_c = 2πħ^2/(mkB) ζ(3/2)^(­2/3) (N/V)^(2/3) 
        #   λ = ħ^2/(2m):= 1, ħ:=1, kB:=1 (energy in units of T)
        # T_c = 4π λ  ζ(3/2)^(­2/3) (N/V)^(2/3)  := 1 temperature scale T_c
        # <=> N/V = (1/(4πλ))^(3/2) ζ(3/2)
        # V = L^3 , so
        # L = (4πλ)^(1/2) (N/ ζ(3/2) )^(1/3)         
        zeta32 = 2.6123753486854883        
        L = (4π*λ)^(1/2)*(N/zeta32)^(1/3)
    else
        # L should be sufficiently large, or you see artefacts (negative kinetic energy etc.)
        # Make sure the same L is in Systems!        
        L = 10.0 * N^(1/dim)        
    end
    E_dist = PIMC_Common.system.E_exact(β, dim, N)
    E_bose = PIMC_Common.system.E_exact_bosons(β, dim, N)
    @show E_dist
    @show E_bose
    
    @show N
    @show L
    @show N/L^dim
    

    PIMC, beads, links, beads_backup, links_backup = init_pimc(M=M,
                                   β=β,                                   
                                   L=L, 
                                   canonical=true,
                                   )
    return PIMC, beads, links, beads_backup, links_backup
end
       
function init_He_liquid(;β::Float64=1.0, M::Int64)      

    
    # read measured density from file
    dat = readdlm("He_liquid/He_liquid_measured_density.dat", comments=true, comment_char='#')
    Ts = dat[:, 1]  
    ρs = dat[:, 2]
    # linear interpolation to find ρ(T)
    T = 1/β
    ind = findfirst(x -> x >= T, Ts)
    if ind==nothing
        # probably past end
        error("Density is outside known T regime")        
    else
        #  ρ in units 1/Å^3
        ρ  = ((Ts[ind]-T)*ρs[ind-1] + (T-Ts[ind-1])*ρs[ind])/(Ts[ind]-Ts[ind-1])
    end

    # MODIFY ρ
    #PIMC_Common.TEST = true
    #ρ += 0.002
    #ρ = 0.025
    
    # solve L from ρ = N/L^dim

    L = (N/ρ)^(1/dim)   # in Å
    @show N, L, ρ

    
    # potential energy tail correction
    Vtail = PIMC_Common.system.tail_correction(ρ, L, dim)
    @show Vtail
    PIMC_Common.Vtail = Vtail

    
    

    #= 
    # apply cutoff - just for experimenting, beware cutoff may ruin the potential anergy
    # PIMC_Common.system.init_V_cutoff(L/2)
    # and call init_pimc with
    # pair_potential=PIMC_Common.system.potential_with_cutoff,
    # der_pair_potential=PIMC_Common.system.der_potential_with_cutoff
    #  
    # Another way to apply potential cutoffs is 
    # using closures (anonymous functions are not playing well with hfd5)
    # pair_potential = PIMC_Common.system.make_potential(L/2)
    # der_pair_potential = PIMC_Common.system.make_der_potential(L/2)
    #
    =#
    
    PIMC, beads, links, beads_backup, links_backup= init_pimc(β=β, M=M, L=L, 
                                                              canonical=true,
                                                              pair_potential=PIMC_Common.system.potential,
                                                              der_pair_potential=PIMC_Common.system.der_potential,
                                                              der2_pair_potential=PIMC_Common.system.der2_potential
                                                              )

    
    return PIMC, beads, links, beads_backup, links_backup
end

function main()


    #
    # Read command line arguments 
    possible_args=["T"]
    arg_dict = argparse(possible_args)
    if arg_dict==nothing
        println("  usage: julia PIMC_main.jl T=temperature")
        println("example:  julia PIMC_main.jl T=3.0")
        error("missing command line argument")
    end
    # use command line values 
    T = get(arg_dict, "T", 0.0)
    if isapprox(T, 0.0)
        error("T=0.0 does not work. Did you set command line parameter T=... ?")
    end
    #
    
    # set M, β (nearest to input 1/T) and worm_K
    M, β  = init_Mβ!(1/T)
    T = 1/β
    
    
    
    
    @show PIMC_Common.sys
    
    PIMC, beads, links = nothing, nothing, nothing
    if PIMC_Common.sys == :HarmonicOscillator
        # Harmonic Oscillator:
        PIMC, beads, links, beads_backup, links_backup = init_harmonic_oscillator(;β=β, M=M)        
    elseif PIMC_Common.sys == :Noninteracting
        # Noninteracting particles in a PBC box
        PIMC, beads, links, beads_backup, links_backup = init_noninteracting(;β=β, M=M)
    elseif PIMC_Common.sys == :HeLiquid
        # Liquid He:
        PIMC, beads, links, beads_backup, links_backup = init_He_liquid(;β=β, M=M)
    end
    PIMC == nothing && error("Unknown Common.sys, set it in Common.jl")
    
    actionstr =  PIMC_Common.action == ChinAction ? "chin" : "prim"
    
    # suffix to data files
    bosestr = bose ? "bose" : "dist"
    filesuffix = "."*PIMC_Common.system.name*"_"*actionstr*"_T"*string(round(1/PIMC.β, digits=4))*
        "_tau"*string(round(PIMC_Common.τ, digits=4))*
        "_M"*string(PIMC.M)*"_N"*string(N)*
        "_"*bosestr*
        ".dat"

    dir = "./"*PIMC_Common.system.name*"/"    
    @show dir, filesuffix
    mkpath(dir)
    PIMC.filesuffix = filesuffix 
    PIMC.restart_file = dir*"restart"*filesuffix*".jld2"

    # initialize action
    init_action!(PIMC, beads)
    
    
    if restart
        println("Restarting from file $(PIMC.restart_file)")
        try
            read_restart!(PIMC, beads, links)
        catch
            println("Restart failed, malformed restart file or error in read_restart!()")              # 
            println("*** FRESH START ***")
            PIMC_Common.restart = false
            PIMC_Common.restart_with_data = false
        end
    end
   
    #
    

    
    # Moves:
    # ======
    add_move!(PIMC, 10, :bead_move, bead_move!)
    add_move!(PIMC, 10, :rigid_move, rigid_move!) 
    add_move!(PIMC, 20, :bisection_move, bisection_move!)
    add_move!(PIMC, 60, :worm_move, worm_move!) # NB: only swap update has particle exchange

    # Reports
    # =======
    add_report!(PIMC, 5000, :PIMC_report, pimc_report)
    add_report!(PIMC, 1000, :acceptance, report_acceptance)   
    PIMC.hdf5_file =  dir*"results"*filesuffix*".h5"
    println("HDF5 output to file  $(PIMC.hdf5_file)")
    add_report!(PIMC, 1000, :Results_to_HDF5, pimc_results_to_hdf5)

    
    # report parameters
    pimc_report(PIMC)
    
    

        
    pimc_results_to_hdf5(PIMC)

    # initialize Action stored values (U is updated from these initial values)
    init_stored(PIMC, beads)
    
    # Warm-up with fast moves; skip if restart
    if !restart
        moves = [:rigid; repeat([:bead], 10); repeat([:bisection], 30)] # move frequencies for warm_up
        println("warm-up beads with bead, rigid, and bisection moves, no worm")
        for i in 1:30000 # just a few            
            move = rand(moves)           
            if move == :rigid
                rigid_move!(PIMC, beads, links)
            elseif move == :bead
                bead_move!(PIMC, beads, links)
            else
                bisection_move!(PIMC, beads, links) 
            end
            if i%1000==0                
                Epot = E_pot_all(PIMC, beads)
                @show Epot/N
            end
        end    
        println("warm-up done, begin thermalization\n")
        write_restart(PIMC, beads, links)            
        println("="^80)
    end

    #
    # Thermalization
    # ==============
    run_pimc(PIMC, beads, links, beads_backup, links_backup, Ntherm)
    PIMC_Common.Ntherm = 0 # signals "thermalization done" to subsequent calls to run_pimc
    #
    # Estimate τ_int from a short run
    # ===============================
    # add_measurement!() : 
    # 2nd arguments: measurement every # of steps
    # 5th arguments: blocksize, used in estimating statistical error
    
    println("Estimating energy τ_int in a short simulation")
    println("adding dummy E_vir measurement:")
    add_measurement!(PIMC, 100000000, :E_vir, meas_E_vir, 3, 10000000, dir*"E_vir"*filesuffix)
    meas = filter(m -> m.name === :E_vir, PIMC.measurements)[1] # dummy
    Es = Vector{Float64}()
    Ps = Vector{Float64}()
    nsteps = 200
    println("PIMC steps (up to $nsteps): ")
  
    for i in 1:nsteps
        print("$i ")
        run_pimc(PIMC, beads, links, beads_backup, links_backup, 1)
        E = meas_E_vir(PIMC, beads, links, meas, opt=true) # meas is dummy
        P = meas_virial_pressure(PIMC, beads, links, meas, opt=true) # meas is dummy
        push!(Es, E)
        push!(Ps, P)
    end
    P = sum(Ps)/length(Ps)
    println()
    @printf("%10.5f %20.8f %20.8f T density pressure\n", 1/PIMC.β, N/(PIMC.L^dim), P)
    println()
    τ_int = integrated_autocorr(Es)
    println("PIMC_Chin_Action E_vir: τ_int =  $τ_int")
    # remove dummy E_vir measurement
    PIMC.measurements = filter(m -> m.name != :E_vir, PIMC.measurements)
            
    println("="^80)
    #exit()

    #
    # Measurements:
    # =============
    
    bf = max(1, round(Int, τ_int))   # also 2*τ_int is fine
    println("Base frequency of measurements $bf")
    # add_measurement! : 
    # 2nd arguments: measurement every # of steps
    # 5th arguments: blocksize, used in estimating statistical error
    add_measurement!(PIMC, bf, :E_th, meas_E_th, 3, 5*bf, dir*"E_th"*filesuffix)
    add_measurement!(PIMC, bf, :E_vir, meas_E_vir, 3, 5*bf, dir*"E_vir"*filesuffix)
    # worm details
    add_measurement!(PIMC, 10*bf, :head_tail_histogram, meas_head_tail_histogram, PIMC.M, 50*bf, dir*"head_tail_histogram"*filesuffix)
    
    if bose && N>1
        add_measurement!(PIMC, bf, :cycle_count, meas_cycle_count, N, 5*bf, dir*"cycle_count"*filesuffix)
        add_measurement!(PIMC, bf, :superfluid_fraction, meas_superfluid_fraction, 1, 5*bf, dir*"rhos"*filesuffix)
        add_measurement!(PIMC, bf, :obdm, meas_obdm, 31, 5*bf, dir*"obdm"*filesuffix)
    end
    if PIMC_Common.sys == :HarmonicOscillator        
        add_measurement!(PIMC, 10*bf, :density_profile, meas_density_profile, 301, 50*bf, dir*"density_profile"*filesuffix)
    end
    if PIMC_Common.sys in (:HeLiquid, :Noninteracting)
        add_measurement!(PIMC, bf, :radial_distribution, meas_radial_distribution, 101, 5*bf, dir*"g"*filesuffix)
        # NB: virial pressure and static structure factor (in present form) are computationally expensive 
        #add_measurement!(PIMC, bf, :virial_pressure, meas_virial_pressure, 1, 5*bf, dir*"Pressure"*filesuffix)
        #add_measurement!(PIMC, bf, :static_structure_factor, meas_static_structure_factor, 101, 2*bf, dir*"S"*filesuffix)
    end
    
    # Delete old raw data files (the hdf5 file will always be overwritten)
    # comment out if you want to keep collecting data after a restart (and get separate error estimates)
    for meas in PIMC.measurements          
        rm(meas.filename, force = true)
    end
    if PIMC_Common.opt_chin
        file = "Chin_opt"*PIMC.filesuffix
        rm(file, force=true)
    end
    
    
    # PIMC START
    @time run_pimc(PIMC, beads, links, beads_backup, links_backup)
    
    
    # or profile for a few steps
    #println("PROFILING RUN, see profile.output")
    #@profile run_pimc(PIMC, beads, links, beads_backup, links_backup, 1000)

    #open("profile.output", "w") do f
    #    Profile.print(f, format=:flat, sortedby=:overhead)
    #end




    
end

main()




