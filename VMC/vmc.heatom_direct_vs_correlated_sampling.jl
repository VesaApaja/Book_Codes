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
const Nw = 100000 # number of walkers used in correlated sampling

mutable struct Walker
    R     ::MMatrix{D,N,Float64}
    ψ     ::Float64
    E     ::Float64
end

#Random.seed!(123414) # optional: use same seed to get rep
 

# initialization 
function init(params ::VMC_Params) ::Array{Walker,1}
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
    
    params.Ntry=0
    params.Naccept=0
    params.step = 3.1
    
    println("done")
    walker
end
# global :
params = VMC_Params(0,0,0.0)
walker = walker = [Walker(rand(Float64,D,N),0.0,0.0) for i in 1:Nw] # R, ψ=0, E=0  

# Correlated sampling functions
# -----------------------------
function generate_walkers()   
    # thermalization
    for i in 1:Ntherm
        vmc_step!(walker[1].R, params)      
    end
    println("generating walkers")
    # generate Nw walkers
    for iw = 1:Nw
        for i = 1:100 # run walker 1 for a while
            walker[1].E, walker[1].ψ = vmc_step!(walker[1].R, params)            
        end
        walker[iw].R = copy(walker[1].R)
        walker[iw].ψ = copy(walker[1].ψ)
        walker[iw].E = copy(walker[1].E)
    end
    println("done")
end
    
function get_correlated_Eσ(β)    
    Es = Array{Float64, 1}(undef, Nw)
    rs = Array{Float64, 1}(undef, Nw)
    for iw = 1:Nw
        wratio = exp(ln_psi2(walker[iw].R, α, β)-ln_psi2(walker[iw].R))
        Es[iw] = Elocal(walker[iw].R, α, β)* wratio
        rs[iw] = wratio
    end
    r_ave = mean(rs)
    E_ave = mean(Es)/r_ave
    
    σ = std(Es)/sqrt(Nw)
    @printf("β = %20.15f E = %20.15f ± %20.15f  <φ_T ratio> = %20.15e\n", β, E_ave, σ, r_ave)
    E_ave, σ
end

# driver for bobyqa
function E_opt_correlated(u)
    β = u[1]
    E_ave, σ = get_correlated_Eσ(β)
    E_ave
end


# Ordinary sampling functions
# ----------------------------
function generate_walkers(α ::Float64, β ::Float64)   
    # thermalization
    for i in 1:Ntherm
        vmc_step!(walker[1].R, α, β, params)      
    end
    @printf("generating walkers with β = %.5f\n", β)
    # generate Nw walkers
    for iw = 1:Nw
        for i = 1:100 # run walker 1 for a while
            walker[1].E, walker[1].ψ = vmc_step!(walker[1].R, α, β, params)            
        end
        walker[iw].R = copy(walker[1].R)
        walker[iw].ψ = copy(walker[1].ψ)
        walker[iw].E = copy(walker[1].E)
    end
end
function get_Eσ(β)
    generate_walkers(α, β)   # always generate new walkers 
    Esamples = Array{Float64, 1}(undef, Nw)
    ratios = Array{Float64, 1}(undef, Nw)
    for iw = 1:Nw
        Esamples[iw] = Elocal(walker[iw].R, α, β)
    end
    E_ave = mean(Esamples)
    σ = std(Esamples)/sqrt(Nw)
    @printf("β = %20.15f <E> = %20.15f ± %20.15f  <φ_T ratio> = %20.15e\n", β, E_ave, σ, 1.0)
    E_ave, σ
end

# driver for bobyqa
function E_opt(u)
    β = u[1]
    E_ave, σ = get_Eσ(β)
    E_ave
end


function main()
    #
    # Main program 
    #
   
    init(params)
    # seach limits
    β_min = 0.0
    β_max = 0.5
    Nβs = 50
    βs = LinRange(β_min, β_max, Nβs)

    # Correlated sampling
    # -------------------
    println("CORRELATED SAMPLING")
    β = 0.3
    generate_walkers()  # generate walkers only *once*
    Es = Array{Float64, 1}(undef, Nβs)
    σs = Array{Float64, 1}(undef, Nβs)
    for (i,β) in enumerate(βs)
        Es[i], σs[i] = get_correlated_Eσ(β)        
    end
    p = plot(βs, Es, yerror = σs, xlabel = L"β",ylabel = L"E(β)",ylimits=(-2.90,-2.84), framestyle=:box,
             label="Correlated sampling")
    plot!([β], seriestype="vline",linecolor="red", label="Walkers sampled with this parameter")
    display(p)
    

    println("\n\nOptimization of E(β) using algorithm bobyqa\n")

    β = 0.2
    u0 = [β] # start from sampling parameters
    println("start with β = ",β)
    
    sol , info = bobyqa(E_opt_correlated, u0; xl=[β_min], xu = [β_max])
    @printf("minimum at β = %.6f\n",sol[1])

    
    plot!(sol, seriestype="vline", linecolor="green", label="Minimum found by algorithm")
    
    display(p)
    println("press enter")

    readline() # just to make the plots appears long enough to see them
    savefig(p,"correlated_sampling_He-atom.pdf")

    # Ordinary sampling
    # -----------------
    println("ORDINARY SAMPLING")
    
    Es = Array{Float64, 1}(undef, Nβs)
    σs = Array{Float64, 1}(undef, Nβs)
    for (i,β) in enumerate(βs)
        Es[i], σs[i] = get_Eσ(β)        
    end
    p = plot(βs, Es, yerror = σs, xlabel = L"β",ylabel = L"E(β)", ylimits=(-2.90,-2.84), framestyle=:box,
             label="Ordinary sampling")


       
    println("\n\nOptimization of E(β) using algorithm bobyqa\n")
    β = 0.2
    u0 = [β] # start from sampling parameters
    println("start with β = ",β)
    
    sol , info = bobyqa(E_opt, u0; xl=[β_min], xu = [β_max])
    @printf("minimum at β = %.6f\n",sol[1])
    
    plot!(sol, seriestype="vline", linecolor="green", label="Minimum found by algorithm")
    display(p)
    println("press enter")
    readline() # just to make the plots appears long enough to see them
    savefig(p,"ordinary_sampling_He-atom.pdf")

    
end


@time main()
