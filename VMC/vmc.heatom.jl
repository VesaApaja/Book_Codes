#=
JULIA

VMC on a helium atom with trial wave function φ_T
define in module Model_Heatom

The Hamiltonian is
H = -1/2* (nabla_1^2 + nabla_2^2) - 2/r1 - 2/r2 + 1/r12  
=#


import LinearAlgebra: norm

using Distributions
using LoopVectorization
using Printf
using StaticArrays
using Statistics


# local modules:
push!(LOAD_PATH,".")
using VMCstep
using Model_Heatom
using Utilities
using QMC_Statistics
#
# Parameters
#
println("He atom VMC")
if trial!="3par"
    @show trial
    println("Use 3par Model in VMC production runs")
    exit()
end

const blocksize = 1000000   # data block size
const Ntherm = 100         # thermalization steps
const accuracy_goal = 2e-4 

mutable struct Walker
    R     ::MMatrix{D,N,Float64}
    ψ     ::Float64
    E     ::Float64
end


# initialization 
function init(params ::VMC_Params) ::Walker
    println("init")
    # 
    # Initialize a walker
    #
    walker = Walker(zeros(Float64,D,N),0.0,0.0)  # R , ψ=0, E=0
    R = @MMatrix rand(D,N) # random coordinates
    walker.R = R
    walker.ψ = ln_psi2(R)  # wave function (ln(psi^2))
    walker.E = Elocal(R)        

    params.Ntry=0
    params.Naccept=0
    params.step = 3.1
    
    println("init done")
    walker
end

       
function main()
    #
    # Main program 
    # 
    # initialize

    params = VMC_Params(0,0,0.0)
    walker = init(params) 
    # thermalization
    println("thermalizing")    
    for i in 1:Ntherm
        ψ,E = vmc_step!(walker.R, params)
        walker.ψ = ψ
        walker.E = E
    end

    println("thermalization done")

    println("Checking numerically EL and drift against φ_T")
    for i in 1:5
        ψ,E = vmc_step!(walker.R, params)
        walker.ψ = ψ
        walker.E = E
        R = walker.R
        num_check_EL(R, E, ln_psi2, V)
    end
    num_check_∇S(walker.R, ln_psi2, drift)
    println("checks passed")
    
    filename = string("E_heatom_VMC")
    println("output: ",filename)
    # init file output
    open(filename,"w") do f
        println(f," ")
    end
    #
    # VMC
    #
    # init E measurement
    Estat = init_stat(1, blocksize)
    #    
    ivmc = 0

    while true # until accuracy_goal is reached
        ψ, E = vmc_step!(walker.R, params)        
        walker.ψ = ψ
        walker.E = E

        # add new energy data
        add_sample!(Estat, E)
        ivmc +=1

        
        if ivmc%10 == 0                      
            adjust_step!(params)
        end
        # saving to file is slow
        #open(filename,"a") do f
        #    println(f,ivmc," ",E)
        #end
        #
        # output when a block is full
        if Estat.finished            
            E_ave, E_std, E_inputvar2, Nb = get_stats(Estat)
            @printf("VMC step %15d E = %.10f <E> = %.10f +/- %.10f\n",ivmc, E,  E_ave, E_std)

            if Nb>10 && E_std < accuracy_goal
                println("reached accuracy goal")
                println("used trial wf ",trial)
                println("Trial wf parameters:")
                @show(α)
                @show(α12)
                @show(β)
                @printf("input σ^2 = %.6f\n", E_inputvar2)
                println("result:")
                output_MCresult(E_ave, E_std)
                
                break
            end
                        
        end
    end
end



@time main()
