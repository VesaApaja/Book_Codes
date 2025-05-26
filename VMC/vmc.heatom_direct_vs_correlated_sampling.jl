#=
JULIA

Correlated sampling energy optimization

VMC on a helium atom with trial wave function φ_T
defined in module Model_Heatom.jl

The Hamiltonian is
H = -1/2* (nabla_1^2 + nabla_2^2) - 2/r1 - 2/r2 + 1/r12  
=#


#using Distributions
using Printf
using StaticArrays
using Statistics
using LaTeXStrings

using Plots

using Optimization, OptimizationNLopt, NLopt
using Random

# local modules:
push!(LOAD_PATH,".")
using Common: VMC_Params
using VMCstep
using Model_Heatom
using Utilities
#
# Parameters
#
const Ntherm = 100 # thermalization steps
const Nw = 100000 # number of walkers used in correlated sampling

mutable struct Walker
    R     ::MMatrix{dim,N,Float64}
    ψ     ::Float64
    E     ::Float64
end

#Random.seed!(123414) # For testing only 

# global :
vmc_params = VMC_Params(0,0,0.0)
walker = Vector{Walker}(undef, 1) # dummy init

# initialization 
function init(vmc_params::VMC_Params) 
    global walker
    println("init ",Nw," walkers")    
    # 
    # Initialize walkers
    #
    R = @MMatrix rand(dim,N)    # coordinates
    walker = [Walker(R, 0.0, 0.0) for i in 1:Nw] # R, ψ=0, E=0    
    
    vmc_params.Ntry=0
    vmc_params.Naccept=0
    vmc_params.step = 3.1
end



function generate_walkers(wf_params::Vector{Float64})       
    println("generating walkers with wf_params ",wf_params)
    # thermalization
    Ψ_fixed_params(x) = Ψ(x, wf_params)
    for i in 1:Ntherm
        vmc_step!(walker[1].R, vmc_params, Ψ_fixed_params)      
    end    
    # generate Nw walkers
    Eave = 0.0
    for iw = 1:Nw
        for i = 1:10 # run walker 1 for a while
            vmc_step!(walker[1].R, vmc_params, Ψ_fixed_params)            
        end
        ψ = vmc_step!(walker[1].R, vmc_params, Ψ_fixed_params)            
        walker[iw].R = copy(walker[1].R)
        walker[iw].ψ = ψ
        walker[iw].E = EL(walker[1].R, wf_params)
        Eave += walker[iw].E
    end
    @printf("initial <E> = %15.5f\n", Eave/Nw) 
end
function get_Eσ(wf_params::Vector{Float64}, correlated::Bool)
    Esamples = Array{Float64, 1}(undef, Nw)
    if correlated        
        ratios = Array{Float64, 1}(undef, Nw)
        for iw = 1:Nw
            wratio = Ψ(walker[iw].R, wf_params)^2/walker[iw].ψ^2         
            Esamples[iw] = EL(walker[iw].R, wf_params)* wratio
            ratios[iw] = wratio
        end
        r_ave = mean(ratios)
        E_ave = mean(Esamples)/r_ave
    else
        generate_walkers(wf_params)  #  always generate new walkers 
        for iw = 1:Nw
            Esamples[iw] = EL(walker[iw].R, wf_params)
        end
        E_ave = mean(Esamples)
    end    
    σ = std(Esamples)/sqrt(Nw)
    for par in wf_params
        @printf("wf parameter %20.15f\n",par)
    end
    if correlated
        @printf("CORRELATED SAMPLING <E> = %20.15f ± %20.15f  <weight ratio> = %20.15e\n", E_ave, σ, r_ave)
    else
         @printf("ORDINARY SAMPLING <E> = %20.15f ± %20.15f\n", E_ave, σ)
    end
    E_ave, σ
end


function main()
    #
    # Main program 
    #
    init(vmc_params)
   
    println("CORRELATED SAMPLING")
    par = get_wave_function_params(:initial_parameters_for_optimization)
    wf_params = [par.α, par.α12, par.β]
    # seach limits
    β_min = 0.0
    β_max = 0.5
    Nβs = 50
    βs = LinRange(β_min, β_max, Nβs)

    # Define optimizer
    n_parameters = 1
    opt =  Opt(:LN_BOBYQA, n_parameters) 
    lower_bounds!(opt, β_min)
    upper_bounds!(opt, β_max)
    xtol_rel!(opt, 1e-5)    # Relative tolerance on optimization parameters
    maxeval!(opt, 10000)

    # help arrays
    Es = Array{Float64, 1}(undef, Nβs)
    σs = Array{Float64, 1}(undef, Nβs)
    
    # Correlated sampling
    # -------------------
    println("CORRELATED SAMPLING")
    
    generate_walkers([par.α, par.α12, par.β])  # generate walkers only *once*
    for (i,β) in enumerate(βs)
        Es[i], σs[i] = get_Eσ([par.α, par.α12, β], true) # true for correlated sampling        
    end
    p = plot(βs, Es, yerror = σs, xlabel = L"β",ylabel = L"E(β)",ylimits=(-2.90,-2.84), framestyle=:box,
             label="Correlated sampling")
    plot!([par.β], seriestype="vline",linecolor="red", label="Walkers sampled with this parameter")
    display(p)

    println("\n\nOptimization of E(β) using algorithm bobyqa\n")

    # E_opt(β) closure
    # x is an array, x[1] is β, the second [1] picks E from tuple output E, σ
    E_opt = x -> get_Eσ([par.α, par.α12, x[1]], true)[1] 
    
    println("start optimization from β = ",par.β, " E = ", E_opt(par.β))
    
    # Define objective 
    min_objective!(opt, (x, _) -> E_opt(x))

    (E_corr_min, wf_params_corr_opt, ret) = optimize(opt, [par.β])

    # Output results
    println("minimum at β: ", wf_params_corr_opt)
    println("Minimum value: ", E_corr_min)
    println("Return code: ", ret) # FORCED_STOP emans failed, probably error in function evalution
    
    plot!([wf_params_corr_opt[1]], seriestype="vline", linecolor="green", label="Minimum found by algorithm")
    display(p)
    println("press enter")
    readline() # just to make the plots appears long enough to see them
    file = "correlated_sampling_He-atom.pdf"
    println("output to file $file")
    savefig(p, file)

    # Ordinary sampling
    # -----------------
    println("ORDINARY SAMPLING")
    
    correlated = false
    for (i,β) in enumerate(βs)
        wf_params .= [par.α, par.α12, β]
        Es[i], σs[i] = get_Eσ(wf_params, correlated)        
    end
    p = plot(βs, Es, yerror = σs, xlabel = L"β",ylabel = L"E(β)", ylimits=(-2.90,-2.84), framestyle=:box,
             label="Ordinary sampling")
       
    println("\n\nOptimization of E(β) using algorithm bobyqa\n")
    println("start optimization with β = ",par.β)
    E_opt = x -> get_Eσ([par.α, par.α12, x[1]], false)[1]  # false for non-correlated samples 
    min_objective!(opt, (x, _) -> E_opt(x))
    (E_min, wf_params_opt, ret) = optimize(opt, [par.β])
    println("minimum at β: ", wf_params_opt)
    println("Minimum value: ", E_min)
    println("Return code: ", ret)
    if ret=="FORCED_STOP"
        error("No minimum found, probably error in function.")
    end
    
    plot!([wf_params_corr_opt[1]], seriestype="vline", linecolor="green", label="Minimum found by algorithm")
    display(p)
    println("press enter")
    readline() # just to make the plots appears long enough to see them
    file = "ordinary_sampling_He-atom.pdf"
    println("output to file $file")
    savefig(p, file)

    
end


@time main()
