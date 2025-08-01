module VMCstep

using StaticArrays
using LinearAlgebra: norm

push!(LOAD_PATH,".")
using Common: VMC_Params

export vmc_step!, adjust_step!




# one VMC step
#=
function vmc_step!(R ::MMatrix, params ::VMC_Params; ln_psi2=missing)
    Ψ = ln_psi2(R)
    dim = size(R,1)
    N = size(R,2)
    @inbounds for i in 1:N
        rr = @SVector rand(Float64,dim)
        d   = params.step*( rr .- 0.5 ) ::SVector{dim,Float64}
        R[:,i] += d
        Ψ_new = ln_psi2(R)
        diff = Ψ_new-Ψ        
        
        accept = false
        params.Ntry +=1
        if diff>=0.0
            accept = true            
        else
            if exp(diff) > rand() 
                accept = true
            end
        end
        if accept
            params.Naccept +=1
            Ψ = Ψ_new            
        else
            # revert to old value
            R[:,i] -= d
        end        
    end
    Ψ
end
=#

# alternatives
# ------------
function vmc_step!(R ::MMatrix, params ::VMC_Params, wf_params ::Vector{Float64})    
    Ψ = ln_psi2(R, wf_params)
    N = size(R,2)
    @inbounds for i in 1:N
        rr = @SVector rand(dim)
        
        d   = params.step*( rr .- 0.5 ) ::SVector{dim,Float64}
        R[:,i] += d
        Ψ_new = ln_psi2(R, wf_params)
        diff = Ψ_new-Ψ        
        
        accept = false
        params.Ntry +=1
        if diff>=0.0
            accept = true            
        else
            if exp(diff) > rand() 
                accept = true
            end
        end
        if accept
            params.Naccept +=1
            Ψ = Ψ_new
        else
            # revert to old value
            R[:,i] -= d
        end        
    end
    Ψ
end

function vmc_step!(R ::MMatrix, params ::VMC_Params, Ψ ::Function)
    # sanity:
    if params.step<1e-15
        error("VMC step is zero")
    end
    ΨR = Ψ(R)
    if ΨR < 0.0
        error("vmc_step! : Ψ(R)<0, this shouldn't happen.")
    end
    Ψ2 = ΨR^2
    dim = size(R,1)
    N = size(R,2)
    for i in 1:N
        rr = rand(dim)
        d  = params.step*( rr .- 0.5 )
        R[:,i] += d
        Ψ_new = Ψ(R)
        Ψ2_new = Ψ_new^2
        ratio = Ψ2_new/Ψ2
        # Metropolis:
        accept = false        
        params.Ntry +=1
        if Ψ_new<0.0
            # fermi nodal cell change, and shoudn't happen for boson ground state either
            accept = false
        else        
            if ratio >= 1.0
                accept = true            
            else
                if ratio > rand() 
                    accept = true
                end
            end
        end
        if accept
            params.Naccept +=1
            Ψ2 = Ψ2_new            
        else
            # revert to old value
            R[:,i] -= d
        end        
    end
    sqrt(Ψ2)
end
#
# Electrons and protons moving, first half electrons, second half protons
# Used in H2 code
# Not to be used with a wave function that has nodes
#
function vmc_step!(R ::MMatrix{dim,N}, params ::Vector{VMC_Params}, Ψ ::Function) where {dim,N}
    ΨR = Ψ(R)
    Ψ2 = ΨR^2
    Nhalf = Int(N/2)
    # electrons and protons move with different steps
    for i in 1:N
        p = (i <= Nhalf) ? params[1] : params[2]
        rr = rand(dim)
        d  = p.step*( rr .- 0.5 )
        R[:,i] += d
        Ψ_new = Ψ(R)
        Ψ2_new = Ψ_new^2
        ratio = Ψ2_new/Ψ2
        # Metropolis:
        accept = false        
        p.Ntry +=1        
        if ratio >= 1.0
            accept = true            
        elseif ratio > rand() 
            accept = true
        end
        if accept
            p.Naccept +=1
            Ψ2 = Ψ2_new            
        else
            # revert to old value
            R[:,i] -= d
        end        
    end
    sqrt(Ψ2)
end
#
# Only electrons moving
# Used in H2 code
# Not to be used with a wave function that has nodes
#

export vmc_step_H2!
    
function vmc_step_H2!(R ::MMatrix{dim,N}, p ::VMC_Params, Ψ ::Function) where {dim,N}
    ΨR = Ψ(R)
    Ψ2 = ΨR^2
    Nhalf = Int(N/2)    
    for i in 1:Nhalf        
        rr = rand(dim)
        d  = p.step*( rr .- 0.5 )
        R[:,i] += d
        Ψ_new = Ψ(R)
        Ψ2_new = Ψ_new^2
        ratio = Ψ2_new/Ψ2
        # Metropolis:
        accept = false        
        p.Ntry +=1        
        if ratio >= 1.0
            accept = true            
        elseif ratio > rand() 
            accept = true
        end
        if accept
            p.Naccept +=1
            Ψ2 = Ψ2_new            
        else
            # revert to old value
            R[:,i] -= d
        end        
    end
    sqrt(Ψ2)
end



function vmc_step!(R ::MMatrix{dim,N}, params ::VMC_Params, wf_params ::Vector{Float64}, Ψ ::Function) where {dim,N}
    ΨR(x) = Ψ(x, wf_params)
    Ψ2 = ΨR(R)^2
    @inbounds for i in 1:N
        rr = @SVector rand(dim)
        d   = params.step*( rr .- 0.5 )
        R[:,i] += d
        Ψ2_new = ΨR(R)^2
        ratio = Ψ2_new/Ψ2
        # Metropolis:
        accept = false        
        params.Ntry +=1
        if ratio >= 1.0
            accept = true            
        else
            if ratio > rand() 
                accept = true
            end
        end
        if accept
            params.Naccept +=1
            Ψ2 = Ψ2_new            
        else
            # revert to old value
            R[:,i] -= d
        end        
    end
    sqrt(Ψ2)
end




const minstep = 1e-5
const maxstep = 20.0

# adjust step to keep acceptance 50-60 %
function adjust_step!(params ::VMC_Params)
    acceptance = params.Naccept*100.0/params.Ntry
    if acceptance<50.0
        params.step *= 0.9
    end
    if acceptance>60.0
        params.step *= 1.1
    end
    params.step = max(minstep,params.step)
    params.step = min(maxstep,params.step)
    if params.step == maxstep
        println("BAD Error: step is maxstep ",maxstep)
        @show(acceptance)
        exit()
    end
end


end
