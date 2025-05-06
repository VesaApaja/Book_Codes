#=
JULIA

Correlated sampling energy optimization

VMC on a helium atom with trial wave function φ_T
defined in module Model_Heatom.jl

The Hamiltonian is
H = -1/2* (nabla_1^2 + nabla_2^2) - 2/r1 - 2/r2 + 1/r12  
=#


using Distributions
using Printf
using StaticArrays
using Statistics
using LaTeXStrings

using Plots

using Optimization
using Random
using PRIMA

# local modules:
push!(LOAD_PATH,".")
using VMCstep
using Model_Heatom
using Utilities
#
# Parameters
#
const Ntherm = 100 # thermalization steps
const Nw = 10000000 # number of walkers used in correlated sampling

mutable struct Walker
    R     ::MMatrix{D,N,Float64}
    ψ     ::Float64
    E     ::Float64
end

#Random.seed!(123414) # optional: use same seed to get rep
 

# initialization 
function init(vmc_params ::VMC_Params) ::Array{Walker,1}
    println("init ",Nw," walkers")    
    # 
    # Initialize walkers
    #
    walker = [Walker(zeros(Float64,D,N),0.0,0.0) for i in 1:Nw] # R, ψ=0, E=0
    for iw in 1:Nw
        ww = walker[iw]
        R = @MMatrix rand(D,N)        # coordinates
        ww.R = copy(R)
        ww.ψ = ln_psi2(R)  # wave function (ln(psi^2))
        ww.E = Elocal(R)        
    end
    
    vmc_params.Ntry=0
    vmc_params.Naccept=0
    vmc_params.step = 3.1
    walker
end
# global :
vmc_params = VMC_Params(0,0,0.0)
walker = walker = [Walker(rand(Float64,D,N),0.0,0.0) for i in 1:Nw] # R, ψ=0, E=0  

# Correlated sampling functions
# -----------------------------
function generate_walkers()   
    # thermalization
    for i in 1:Ntherm
        vmc_step!(walker[1].R, vmc_params)      
    end
    println("generating walkers")
    # generate Nw walkers
    for iw = 1:Nw
        for i = 1:10 # run walker 1 for a while
            vmc_step!(walker[1].R, vmc_params)            
        end
        ψ,E = vmc_step!(walker[1].R, vmc_params)            
        walker[iw].R = copy(walker[1].R)
        walker[iw].ψ = ψ
        walker[iw].E = E
    end
    println("done")
end
    
function get_correlated_Eσ(wf_params)
    Esamples = Array{Float64, 1}(undef, Nw)
    ratios = Array{Float64, 1}(undef, Nw)
    for iw = 1:Nw
        wratio = exp(ln_psi2(walker[iw].R, wf_params)-ln_psi2(walker[iw].R))
        Esamples[iw] = Elocal(walker[iw].R, wf_params)* wratio
        ratios[iw] = wratio
    end
    r_ave = mean(ratios)
    E_ave = mean(Esamples)/r_ave
    
    σ = std(Esamples)/sqrt(Nw)
    for par in wf_params
        @printf("wf parameter %20.15f\n",par)
    end
    @printf("E = %20.15f ± %20.15f  <φ_T ratio> = %20.15e\n", E_ave, σ, r_ave)
    E_ave, σ
end

# driver for bobyqa
function E_opt_correlated(wf_params)
    E_ave, σ = get_correlated_Eσ(wf_params)
    E_ave
end

function get_correlated_σ2(wf_params)
    Eguess = 0.0
    for iw = 1:Nw
        Eguess += walker[iw].E
    end
    Eguess /= Nw
    
    σ2samples = Array{Float64, 1}(undef, Nw)
    ratios = Array{Float64, 1}(undef, Nw)    
    for iw = 1:Nw
        wratio = exp(ln_psi2(walker[iw].R, wf_params)-ln_psi2(walker[iw].R))
        σ2samples[iw] = (Elocal(walker[iw].R, wf_params)-Eguess)^2 * wratio
        ratios[iw] = wratio
    end
    r_ave = mean(ratios)
    σ2_ave = mean(σ2samples)/r_ave
    σ = std(σ2samples)/sqrt(Nw)
    for par in wf_params
        @printf("wf parameter %20.15f\n",par)
    end
    @printf("σ^2 = %20.15f ± %20.15f  <φ_T ratio> = %20.15e\n", σ2_ave, σ , r_ave)
    σ2_ave, σ
end

# driver for bobyqa
function σ2_opt_correlated(wf_params)    
    σ2_ave, σ = get_correlated_σ2(wf_params)
    σ2_ave
end


function main()
    #
    # Main program 
    #
   
    init(vmc_params)
   
    println("CORRELATED SAMPLING")
    if trial!="3par_opt"
        println("Use 3par_opt Model in VMC optimization runs")
        exit()
    end

    
    println("\nOptimization of E using algorithm\n")


    # keep wf_params as Vector, bobyqa won't handle MVectors
    u0 = [α, α12, β]  # start with values from Model
    wf_params = copy(u0)  

    println("start with paramaters = ", wf_params)
    generate_walkers()  # generate walkers only *once*

    # limits
    # optimize all parameters
    #xl = [0.0, 0.0, 0.0]
    #xu = [20.0, 20.0, 20.0]
    
    # optimize only α and β:
    #xl = [0.0, α12, 0.0]
    #xu = [3.0, α12+1e-5, 1.0]
    # optimize only β:
    xl = [α, α12, 0.0]
    xu = [α+1e-5, α12 + 1e-5, 1.0]  # algo needs finite range 
    sol_Eopt , info = bobyqa(E_opt_correlated, u0; xl = xl, xu = xu)

    println("Energy minimum at wf parameter")
    for s in sol_Eopt
        @printf("%.6f\n",s)
    end
        
    # Variance optimization
    # ---------------------
    # use same walkers as in energy optimization
       

    println("\n\nOptimization of σ^2 using algorithm\n")
    
    u0 = [α, α12, β] # start with values from Model
    wf_params = copy(u0)
    println("start with paramaters = ", wf_params)
    
    sol_σ2 , info = bobyqa(σ2_opt_correlated, u0; xl = xl, xu = xu)

    println("\n\nRESULTS:")
    println("Variance minimum at wf parameter")
    for s in sol_σ2
        @printf("%.6f\n",s)
    end
    wf_params = sol_σ2
    get_correlated_Eσ(wf_params)

    
    println("\n\nEnergy minimum at wf parameter")
    for s in sol_Eopt
        @printf("%.6f\n",s)
    end
    wf_params = sol_Eopt
    get_correlated_Eσ(wf_params)
    
end


@time main()
