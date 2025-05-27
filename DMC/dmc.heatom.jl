#=
JULIA

 DMC on a helium atom with trial wave function 
 φ_T = e^{-α*r1}e^{-α*r2} ; r1,r2 distance of els from proton
 one el in spin-up and the other in spin-down state
  
Measures only total energy
Hamiltonian in a.u. (Z=2) :

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
using Utilities
using QMC_Statistics
using Common
#
# Parameters
#
const order = 1   # 0, 1 or 2
println("He atom DMC using algorithm of order ",order)

# He atom parameters
const D = 0.5   # hbar^2/(2m) in a.u.
const N = 2              # number of electrons
const Z = 2
const dim = 3              # dimension           
const Eexact = -2.903724377  # accurate ground state energy


# VMC and DMC parameters

# imaginary time step, either default or command line argument 
t = 0.001         #  default
# command line arg
if length(ARGS)!=0
    t = parse(Float64, ARGS[1]) # convert string to float
end
const τ = t ::Float64

const blocksize = floor(Int64,100/τ) # DMC data block size

const NVMC = 1000        # number of VMC steps
const NDMC = 10000000    # number of DMC steps
const Nwx = 20000        # max number of walkers
const Nw_target = 1000   # on average this many walkers
const κ = 0.1            # how zealously DMC tries to keep Nw_target walkers  

# ==================================
# Trial wf
# ========
# parameters
const α = 2.0 # 1.847529
const β = 0.159321
const γ = 1.0
const α12 = 0.5 # 0.359070

@inline function get_unitvecs(R::MMatrix{dim,N,Float64})
    vecr1 = R[:,1]
    vecr2 = R[:,2]
    r1 = norm(vecr1)  
    r2 = norm(vecr2)
    vecr12 =  vecr1 - vecr2
    r12 = norm(vecr12)
    vecr1/r1,vecr2/r2,vecr12/r12
end

function V(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    - Z/r1 - Z/r2 + 1/r12
end

@inline function φ_T(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1 = norm(R[:,1])
    r2 = norm(R[:,2])
    b = 1+β*r12
    exp(-α*(r1+r2) + α12*r12/b)
end


# F=2∇S
@inline function F(R ::MMatrix)
    hatr1, hatr2, hatr12 =  get_unitvecs(R)
    r12 = norm(R[:,1]-R[:,2])
    b = 1+β*r12    
    ∇S = hcat(-α*hatr1 + α12/b^2*hatr12, -α*hatr2 - α12/b^2*hatr12)
    2∇S
end


@inline function EL(R ::MMatrix)
    r12 = norm(R[:,1]-R[:,2])
    r1  = norm(R[:,1])
    r2  = norm(R[:,2])
    hatr1, hatr2, hatr12 = get_unitvecs(R)
    b = 1+β*r12
    ∇S = 0.5*F(R)
    ∇2S = -α*2/r1 -α*2/r2  + 4*α12/b^3*1/r12 
    EL = -0.5*(sum(∇S.^2) + ∇2S) - Z/r1 - Z/r2 + 1/r12
end

# ==================================



mutable struct Walker
    R     ::MMatrix{dim,N,Float64}
    alive ::Int64
    E     ::Float64
end

# initialization 
function init(params ::VMC_Params) ::Array{Walker,1}
    println("init")    
    # 
    # Initialize walkers
    #
    walker = [Walker(zeros(Float64,dim,N),0,0.0) for i in 1:Nwx] # R, alive=0, E=0
    println("generating $Nw_target walkers")
    # set Nw_target walkers
    for iw in 1:Nw_target
        ww = walker[iw]
        ww.alive = 1
        R = @MMatrix rand(dim,N)        # coordinates
        ww.R = copy(R)
        ww.E = EL(R)        
    end
    params.Ntry=0
    params.Naccept=0
    params.step = 3.1
    
    println("init done")
    walker
end

# one diffusion+drift step in DMCstep = params.step
    
@inline function diffusion_drift_step!(x ::MMatrix{dim,N,Float64},
                                       y ::MMatrix{dim,N,Float64},
                                       z ::MMatrix{dim,N,Float64})
    # help arrays y and z
    if order==1
        # diffusion(τ)+drift(τ)
        η = reshape(rand(Normal(0,1),dim*N),(dim,N))
        Fx = F(x) 
        @. x += sqrt(2*D*τ)*η + D*τ*Fx 
           
    elseif order==2
        #
        # diffusion(τ/2)+drift(τ)+diffusion(τ/2)
        #
        # step 1)
        η = reshape(rand(Normal(0,1),dim*N),(dim,N))        
        @. y = x + sqrt(D*τ)*η    
        Fy = F(y)
        # step 2)
        @. z = y + D*τ/2*Fy        
        Fz = F(z)
        # step 3)
        η = reshape(rand(Normal(0,1),dim*N),(dim,N))
        @. x = y + D*τ*Fz + sqrt(D*τ)*η
    elseif order==3
        dt = 2*D*τ
        eta = reshape(rand(Normal(0,1),dim*N),(dim,N))
        @. x = x + sqrt(dt/2)*eta
        F1 = F(x)
        @. y = x + dt/2*F1/2 
        F2 = F(y)
        eta = reshape(rand(Normal(0,1),dim*N),(dim,N)) 
        @. x += dt*F2/2 + sqrt(dt/2)*eta
    end

    
end



function lnG(xp ::MMatrix{dim,N,Float64}, x ::MMatrix{dim,N,Float64})         
    # only parts not symmetric in x'<->x
    # G(x'<-x,τ) = exp(-(x'-x-dτF(x))^2/(4Dτ)
    # 
    if order != 1
        error("lnG only for 1st order code")
    end
    lnG = -norm(xp - x - D*τ*F(x))^2 /(4*D*τ) 
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
    Nw = Nw_target # for VMC
    @inbounds for i in 1:10
        @inbounds for iw in 1:Nw
            vmc_step!(walker[iw].R, params, φ_T)
        end
    end

    println("thermalization done")

    println("Checking numerically EL and drift against φ_T")
    for i in 1:5
        vmc_step!(walker[1].R, params, φ_T)
        R = walker[1].R
        E = EL(R)
        num_check_EL(R, E, φ_T, V)
    end
    
    num_check_∇S(walker[1].R, φ_T, F)
    println("checks passed")

    filename = string("E_heatom_",order,"_tau=",τ)
    #filename = string("E_heatom_noacceptreject",order,"_tau=",τ)
    println("output: ",filename)
    # init file output
    open(filename,"w") do f
        println(f," ")
    end
    #
    # VMC
    #
    E_ave = 0
    nE  = 0
    @inbounds for ivmc in 1:NVMC
        E = 0.0 
        @inbounds for iw in 1:Nw         
            vmc_step!(walker[iw].R, params, φ_T)
            walker[iw].E = EL(walker[iw].R)
            E += walker[iw].E 
        end
        E_ave += E/Nw        
        nE +=1
        
        if ivmc%10 == 0            
            @printf("VMC  E = %.10f <E> = %.10f\n",E/Nw,E_ave/nE)            
        end
        adjust_step!(params)
    end
    
    #
    # DMC
    #
    ET = E_ave/nE  # trial energy, to be updated
    println("ET = $ET")
    
    Ntherm = floor(Int64,1.0/τ) # start DMC measurement after excitations have died out; system specific !
    idmc = -Ntherm

    E_ave = 0
    nE = 0
    


    
    Rold = @MMatrix zeros(Float64,dim,N)

    copies = @MVector zeros(Int64,Nwx)
    Ntry = 0 ::Int64
    Nacc = 0 ::Int64

    R1 = @MMatrix zeros(Float64,dim,N)
    R2 = @MMatrix zeros(Float64,dim,N)
   
    # init E measurement
    Estat = init_stat(1, blocksize)

    while true
        #
        # take a DMC step in each walker
        #
        
        E = 0.0  ::Float64
        n = 0
        @inbounds for iw in 1:Nwx
            copies[iw] = 0
            if walker[iw].alive==0; continue; end  # not alive, skip
            ELold = walker[iw].E  ::Float64            
            Rold  = copy(walker[iw].R) # copy
            Rnew = walker[iw].R # alias
            #
            diffusion_drift_step!(Rnew, R1, R2)
            # 
            
            ELnew = EL(walker[iw].R)    ::Float64
            
            accept = true
            
            if order==1
                # metropolis-Hastings                
                Wnew = lnG(Rold, Rnew) + 2*log(φ_T(Rnew)) ::Float64 # lnG(new->old) + lnpsi2(new)
                Wold = lnG(Rnew, Rold) + 2*log(φ_T(Rold))  ::Float64 # lnG(old->new) + lnpsi2(old)
                accept = metro(Wold,Wnew)
            end
            Ntry +=1
            if accept
                Nacc +=1
                walker[iw].E = ELnew
            else
                ELnew = ELold
                walker[iw].E = ELold
                Rnew .= Rold                
            end

            weight  = exp(-τ*(0.5*(ELold+ELnew)-ET)) ::Float64 # symmetric under R<->R'
            copies[iw]  = floor(Int64,weight + rand()) ::Int64
        end
        # Branching
        for iw in 1:Nwx
            if walker[iw].alive != 1; continue; end  # not alive or a child, skip
            if copies[iw]==1; continue; end         # one copy is already there
            if copies[iw]==0
                # walker dies
                walker[iw].alive = 0
                continue
            end
            # copies[iw]>1
            # copy the walker to empty slots 
            for inew in 1:copies[iw]-1   # with copies=3 this is 1,2
                for iw2 in 1:Nwx
                    if walker[iw2].alive>0; continue; end 
                    # free slot
                    walker[iw2].R = copy(walker[iw].R)
                    walker[iw2].E =  walker[iw].E 
                    walker[iw2].alive = 1 
                    copies[iw2] = 1  # child will not be copied 
                    break
                end
            end
        end
        # collect energy from walkers
        # note: do this *after* drift, diffusion and branching ! 
        E   = 0.0  ::Float64
        n   = 0    ::Int64
        for iw in 1:Nwx
            if walker[iw].alive == 1
                EL = walker[iw].E
                E   +=  EL
                n   += 1
            end
        end

        
        Nw = sum([walker[i].alive for i in 1:Nwx])
        E_ave += E/n 
        nE += 1

        idmc +=1
        
        if Nw==Nwx
            println("Error: hit max number of walkers")
            exit(1)
        end

        # update trial energy; on average Nw_target walkers
        # A too large factor κ will cause a bad feedback,
        # a too small will let number of walkers get too large
        ET = E_ave/nE + κ*log(Nw_target/Nw)

        # add new energy data
        add_sample!(Estat, E/n)

        if idmc<=0
            @printf("DMC %10d  E = %.10f <E> = %.10f ET = %.10f <E>_exact = %.10f  %6d Walkers \n",
                    idmc, E/Nw, E_ave/nE, ET, Eexact, Nw)
            if order==1;@printf("DMC acceptance = %.5f %%\n",Nacc*100.0/Ntry);end
        end

        # block data
        # ==========
        # screen and file output
        if Estat.finished
            Eb_ave, Eb_std, E_inputvar2, Nb = get_stats(Estat)
            @printf("DMC %10d E %.10f <E> = %.10f +/- %.10f ET = %.10f <E>_exact = %.10f  %6d Walkers \n",
                    idmc, E/n ,Eb_ave, Eb_std, ET, Eexact, Nw)
            open(filename,"a") do f
                println(f,τ," ",Eb_ave," ",Eb_std," ",Eexact)
            end
        end
        
        # 
        if idmc==0
            println("THERMALIZATION ENDS")
            Estat = init_stat(1, blocksize)
            E_ave = 0
            nE = 0
        end

        if idmc == NDMC
            println("Trial wf parameters:")
            @show(α)
            @show(α12)
            @show(β)
            println("result: τ  <E>  error")
            @printf("%g ",τ)
            output_MCresult(Eb_ave, Eb_std)
            break
        end
    end
        
end
    
@time main()

