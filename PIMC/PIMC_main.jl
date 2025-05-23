__precompile__(false)
#
# PIMC main
#
using Random, LinearAlgebra, Statistics
using Printf
using DelimitedFiles
using Profile

# local modules:
push!(LOAD_PATH,".")
using PIMC_Utilities
using PIMC_Common
using PIMC_Structs
using QMC_Statistics
using PIMC_Moves
using PIMC_Measurements
using PIMC_Reports

using PIMC_Primitive_Action: meas_E_th as meas_E_th_prim, meas_E_vir as meas_E_vir_prim
using PIMC_Primitive_Action: init_action! as init_primitive_action!



using PIMC_Chin_Action: meas_E_th as meas_E_th_chin, meas_E_vir as meas_E_vir_chin
using PIMC_Chin_Action: init_action! as init_chin_action!



##Random.seed!(123456) #TESTING

# compile-time dispatch:
# ======================
function meas_E_th(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    return meas_E_th_prim(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
end
function meas_E_th(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    return meas_E_th_chin(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
end
function meas_E_th(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    A = PIMC_Common.action  
    return meas_E_th(A(), PIMC, beads, links, meas)
end
function meas_E_vir(::PrimitiveAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    return meas_E_vir_prim(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
end
function meas_E_vir(::ChinAction, PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    return meas_E_vir_chin(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
end
function meas_E_vir(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    A = PIMC_Common.action  
    return meas_E_vir(A(), PIMC, beads, links, meas)
end
# ==================================

function init_M(β::Float64)
    if PIMC_Common.action <: PrimitiveAction
        M = round(β/τ_target)|>Int64
        M = M>10 ? M : 10        
    elseif PIMC_Common.action <: ChinAction
        M = round(3*β/PIMC_Common.τ_target)|>Int64
        M = M>9 ? M : 9
        while M%3 != 0
            M += 1
        end        
    end
    M
end



function init_harmonic_oscillator(; β::Float64=1.0, M::Int64)
    
    PIMC, beads, links, beads_backup, links_backup = init_pimc( M=M,
                                   β=β,                                   
                                   L=1e6,  
                                   canonical=true,
                                   confinement_potential=PIMC_Common.system.potential,
                                   grad_confinement_potential=PIMC_Common.system.grad_potential)
    

   
    
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
    dat = readdlm("He_liquid_measured_density.dat", comments=true, comment_char='#')
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
                                                              der_pair_potential=PIMC_Common.system.der_potential
                                                              )

    
    return PIMC, beads, links, beads_backup, links_backup
end

function main()


    
    # Read command line arguments 
    possible_args=["T"]
    arg_dict = argparse(possible_args)
    # use command line values 
    T = get(arg_dict, "T", 0.0)
    if isapprox(T, 0.0)
        error("T=0.0 does not work. Did you set command line parameter T=... ?")
    end
    
    # set M and worm_K
    β =  1/T
    M  = init_M(β)
    @show M
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
        "_tau"*string(round(PIMC_Common.τ_target, digits=4))*
        "_M"*string(PIMC.M)*"_N"*string(N)*
        "_"*bosestr*
        ".dat"

    dir = "./"*PIMC_Common.system.name*"/"    
    @show dir, filesuffix
    mkpath(dir)
    
    PIMC.restart_file = dir*"restart"*filesuffix*".jld2"
    if restart
        println("Restarting from file $(PIMC.restart_file)")
        try
            read_restart!(PIMC, beads, links)
        catch
            #error("Restart failed, malformed restart file or error in read_restart!()")
            println("Restart failed, malformed restart file or error in read_restart!()")
            println("FRESH START")
            PIMC_Common.restart = false
        end
    end

   

   
    
    # sanity checks
    if worm_K >= PIMC.M
        @show  worm_K, PIMC.M
        error("Can't have worm_K >= PIMC.M, swap_bead may become worm.head")
    end

    
    if PIMC_Common.action == PrimitiveAction
        init_primitive_action!(PIMC, beads)
    elseif PIMC_Common.action == ChinAction
        init_chin_action!(PIMC, beads)
    end
    

    # Measurements:
    # =============
    add_measurement!(PIMC, 10, :E_th, meas_E_th, 3, 100, dir*"E_th"*filesuffix)
    add_measurement!(PIMC, 10, :E_vir, meas_E_vir, 3, 100, dir*"E_vir"*filesuffix)

    # worm details
    add_measurement!(PIMC, 10, :head_tail_histogram, meas_head_tail_histogram, PIMC.M, 100, dir*"head_tail_histogram"*filesuffix)
    
    if bose && N>1
        add_measurement!(PIMC, 10, :cycle_count, meas_cycle_count, N, 100, dir*"cycle_count"*filesuffix)
        add_measurement!(PIMC, 10, :superfluid_fraction, meas_superfluid_fraction, 1, 100, dir*"rhos"*filesuffix)
    end
    if PIMC_Common.sys == :HarmonicOscillator        
        add_measurement!(PIMC, 10, :density_profile, meas_density_profile, 301, 1000, dir*"density_profile"*filesuffix)
    end
    if PIMC_Common.sys in (:HeLiquid, :Noninteracting)
        add_measurement!(PIMC, 10, :radial_distribution, meas_radial_distribution, 101, 100, dir*"g"*filesuffix)       
    end

   
    # Moves:
    # ======
    add_move!(PIMC, 10, :bead_move, bead_move!)
    add_move!(PIMC, 10, :rigid_move, rigid_move!) 
    add_move!(PIMC, 60, :bisection_move, bisection_move!)
    add_move!(PIMC, 30, :worm_move, worm_move!) # must have for bosons, only swap update has particle exchange

    # Reports
    # =======
    add_report!(PIMC, 1000, :PIMC_report, pimc_report)
    add_report!(PIMC, 1000, :acceptance, report_acceptance)   
    PIMC.hdf5_file =  dir*"results"*filesuffix*".h5"
    println("HDF5 output to file  $(PIMC.hdf5_file)")
    add_report!(PIMC, 1000, :Results_to_HDF5, pimc_results_to_hdf5)

    
    # report parameters
    pimc_report(PIMC)
    
    # delete old data files (hdf5 file will be overwritten anyhow)   
    for meas in PIMC.measurements          
        rm(meas.filename, force = true)
    end

        
    pimc_results_to_hdf5(PIMC::t_pimc)

    # initialize Action stored values (U is updated from these initial values)
    init_stored(PIMC, beads)
    
    # Warm-up with fast moves; skip if restart
    if !restart
        moves = [:rigid; repeat([:bead], 10); repeat([:bisection], 30)] # move frequencies for warm_up
        println("warm-up beads with bead, rigid, and bisection moves, no worm")
        for i in 1:10000 # just a few            
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
        println("="^80)
    end
    
    # PIMC START
    run_pimc(PIMC, beads, links, beads_backup, links_backup)

    # or profile for 100 steps
    #println("PROFILING RUN, see profile.output")
    #profile run_pimc(PIMC, beads, links, beads_backup, links_backup, 500)

    #open("profile.output", "w") do f
    #    Profile.print(f, format=:flat, sortedby=:overhead)
    #end




    
end

main()




