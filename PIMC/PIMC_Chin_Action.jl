__precompile__(false)
module PIMC_Chin_Action
using Printf
# AD:
using ForwardDiff

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1) 
using StaticArrays
using BenchmarkTools
using InteractiveUtils


push!(LOAD_PATH,".")
using PIMC_Structs
using QMC_Statistics
using PIMC_Systems
using PIMC_Common
using PIMC_Reports: report_energy
using PIMC_Utilities: boson_virial_exchange_energy, centroid_ref!

export U, K, U_stored, U_update, update_stored_slice_data
export meas_E_th, meas_E_vir, init_action!, meas_virial_pressure
export opt_chin_a1_chin_t0

# M = 6
# Remember:  W_{1-2a1} between two t1*τ propagations
#
# ( 1 -------------------- W_a1 )
#      t_6 = 2t0*τ
#  6 -------------------- W_a1
#      t_5 = t1*τ                   
#  5 -------------------- W_{1-2a1}       (m+1)%3 = 0   (g(r) and obdm are measured on this slice)
#      t_4 = t1*τ               
#  4 -------------------- W_a1 
#      t_3 = 2t0*τ                    
#  3 -------------------- W_a1 
#      t_2 = t1*τ                    
#  2 -------------------- W_{1-2a1}         (m+1)%3 = 0  (g(r) and obdm are measured on this slice)
#      t_1 = t1*τ                    
#  1 -------------------- W_a1     
# 
#  m                      V_m
# 

const Nthreads = Threads.nthreads()

# Chin Action parameter limits (Chin and Chen, J. Chem. Phys. 117, 1409 (2002))
# used in opt_chin (experimental)
const chin_a1_lims = [0,1/2]  
const chin_t0_lims = [0,1/2*(1-1/sqrt(3))] # about [0, 0.21]



if PIMC_Common.opt_chin
    # experimental
    # prepare to changes in parameter due to optimization 
    # global parameters, values reset in set_chin_params()
    chin_a1::Float64 = 0.0  # (almost) free parameter
    chin_t0::Float64 = 0.0  # (almost) free parameter
    chin_t1::Float64 = 0.0
    chin_u0::Float64 = 0.0
    chin_v1::Float64 = 0.0
    chin_v2::Float64 = 0.0
    function set_chin_params(PIMC::t_pimc)
        global chin_a1, chin_t0, chin_t1, chin_u0, chin_v1, chin_v2
        chin_a1 = PIMC.chin_a1        
        chin_t0 = PIMC.chin_t0
        chin_t1 = 1/2 - chin_t0
        chin_u0 = 1/12*( 1 - 1/(1-2*chin_t0) + 1/(6*(1-2*chin_t0)^3) )
        chin_v1 = 1/(6*(1-2*chin_t0)^2)
        chin_v2 = 1-2*chin_v1
        return nothing
    end
else
    # Generic good Chin action parameters (Boronat '09)
    println("Chin Action: fixing chin_a1 and chin_t0 to const values (ignores values in PIMC.***) ")
    const chin_a1 = 0.33        
    const chin_t0 = 0.1215
    const chin_t1 = 1/2 - chin_t0
    const chin_u0 = 1/12*( 1 - 1/(1-2*chin_t0) + 1/(6*(1-2*chin_t0)^3) )
    const chin_v1 = 1/(6*(1-2*chin_t0)^2)
    const chin_v2 = 1-2*chin_v1
end



# light-weight buffers, not thread-safe!
const vec_buffer =  MVector{dim, Float64}(undef) 
const vec_buffer2 = MVector{dim, Float64}(undef)
const vec_buffer3 = MVector{dim, Float64}(undef) 


const gradVTVx_buffer = Matrix{Float64}(undef, dim, N_slice_max)


# Storage for V(x) and [V,[T,V]](x) and force F on each slice
mutable struct t_stored
    set::Bool
    V::Vector{Float64}
    VTV::Vector{Float64}
    F::Matrix{Float64}
    ΔV::Vector{Float64}
    ΔF::Matrix{Float64}
    V_tmp::Vector{Float64}
    VTV_tmp::Vector{Float64}
    F_tmp::Matrix{Float64}
    function t_stored(M::Int, Nb::Int)
        new(false, zeros(M), zeros(M), zeros(dim, Nb), zeros(M), zeros(dim, Nb), zeros(M), zeros(M), zeros(dim, Nb))  
    end
end
const stored_ref = Ref{Union{Nothing, t_stored}}(nothing)

function get_stored()    
    return stored_ref[]
end

# Vm energy storage (to avoid both E_th and E_vir evaluating the same EVm)
mutable struct t_estore
    ipimc::Int
    iworm::Int
    Evalue::Float64
end
const EVm_store = t_estore(0, 0, 0.0)


function init_action!(PIMC::t_pimc, beads::t_beads) ::Nothing
    """Init Chin Action"""
    #global     
    β = PIMC.β
    M = PIMC.M
    τ = 3*β/M    
    PIMC.τ = τ
    if PIMC_Common.opt_chin
        set_chin_params(PIMC)
    else
        PIMC.chin_a1 = chin_a1
        PIMC.chin_t0 = chin_t0
    end
    t0 = chin_t0
    t1 = chin_t1    
    # t1 + t1 + 2t0 = 1
    # imaginary time progresses t1, t1, 2t0, t1, t1, 2t0 ...
    chin_times = Float64[]
    t = 0.0 
    for m in 1:M
        push!(chin_times, t)
        if m%3 == 0
            Δt = 2*t0*τ
        else
            Δt = t1*τ
        end
        t += Δt
    end
    # set bead times
    for m in 1:M
        bs = beads.at_t[m]
        for id in bs
            beads.times[id] = chin_times[m]
        end
    end
    # we may call this init several times in optimization of chin_a1 and chin_t0
    if stored_ref[] === nothing 
        stored_ref[] = t_stored(PIMC.M, length(beads.ids)) 
    end    
    return nothing
end


# Thermodynamic energy estimator
# ==============================


function meas_E_th(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false) ::Float64
    M = PIMC.M    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
    # kinetic energy terms
    NM = count(beads.active)  # instantaneous N*M, in case of GC
    PIMC.canonical && NM != N*M && error("NM is not N*M, bead activation problem?")
    
    Ekin1 = dim*N*M/(2*β)
    Ekin2::Float64 = 0.0
    @inbounds begin
        for id in beads.ids[beads.active]
            id_next = links.next[id]
            Δr2 = dist2(beads.X, id, id_next, L)        
            Δτ  = wrapβ(beads.times[id_next] - beads.times[id], β)
            Ekin2 += Δr2/Δτ
        end
    end
       
    Ekin  = Ekin1 - Ekin2/(4*λ*β) # large cancellation
    #
    # potential energy terms
    # ∂β = 3/M ∂τ
    # EVm = <∂β sum_m (τ*V_m(x,τ))>
    # 
    EVm = get_EVm(PIMC, beads)
    
    # total energy per particle
    Ekin /= N
    EVm /= N
    E = Ekin + EVm

    if opt
        return E
    end
    
    # Collect statistics
    add_sample!(meas.stat, SVector{3, Float64}(E, Ekin, EVm))
    if meas.stat.finished
        # block full, report
        report_energy(PIMC, meas)
    end
    return E
end


function meas_E_vir(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false) ::Float64
    """Virial energy estimator"""

    M = PIMC.M    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
    
    Ekin = dim*N/(2*β)  # exact distinguishable N-particle kinetic energy    
    Eexc::Float64 = 0.0
    if bose
        Eexc = boson_virial_exchange_energy(PIMC, beads, links)
        Ekin += Eexc
    end
   
    # Virial terms are
    #   sum_m [∂β (τ Vtilde_m(x_m;τ))] + 1/(2β) sum_m (x_m - x_ref_m) ⋅ ∂x_m [τ  Vtilde_m(x_m;τ)]
    # = M/3 sum_m [∂τ (τ Vtilde_m(x_m;τ))] + 3/M 1/2 sum_m (x_m - x_ref_m) ⋅ ∂x_m Vtilde_m(x_m;τ)
    #  get_EVm() := [∂τ (τ Vtilde_m(x_m;τ))] is computed analytically 

    
    # term τ*sum_m ∂β'[Vtilde_m(x_m^s; τ')]|β'=β
    # =   3/M *β * sum_m ∂β'[Vtilde_m(x_m^s; τ')]|β'=β]

    # for debugging and timing:
    if false
        PIMC_Common.TEST = true
        println("direct:")
        @btime Evir_chin_direct($PIMC, $beads, $links)
        println("scaled:")
        @btime Evir_chin_scaled($PIMC, $beads, $links)
        
        t_scaled = @elapsed begin
            Evir_scaled = Evir_chin_scaled(PIMC, beads, links)
        end
        
        t_direct = @elapsed begin
            Evir_direct = Evir_chin_direct(PIMC, beads, links)
        end        
        println()
        @printf("TEST: E_vir scaled  %0.10f  E_vir direct  %0.10f  ΔE %0.10f \n",
                Evir_scaled, Evir_direct, Evir_scaled - Evir_direct)
        @printf("TEST  timings\n")
        @printf("scaled %15.5f\n",t_scaled)
        @printf("direct %15.5f\n",t_direct)
        
        if !isapprox(Evir_scaled, Evir_direct, atol=1e-7)
            @show Evir_scaled, Evir_direct, Evir_scaled - Evir_direct
            error("mismatch")
        end
        exit()
    end    

    # pick scaled AD method or direct derivative method:
    if true
        Evir = Evir_chin_scaled(PIMC, beads, links)        
    else
        Evir = Evir_chin_direct(PIMC, beads, links)
    end
    Ekin += Evir
    #
    # Evm := 3/M * sum_m ∂β[β * Vtilde_m(x_m; τ)]    
    #
    EVm = get_EVm(PIMC, beads)
    
    
    Ekin /= N
    EVm /= N
    E = Ekin + EVm

    if opt
        return E
    end
       
    # Collect statistics
    add_sample!(meas.stat, SVector{3, Float64}(E, Ekin, EVm))
    if meas.stat.finished
        # block full, report
        report_energy(PIMC, meas)
    end
    return E
   
end


mutable struct t_Pcoll
    P::Float64 
    n::Int
end
Pcoll = t_Pcoll(0.0, 0)

function meas_virial_pressure(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement,
                              gradVm::MVector{dim, Float64} = vec_buffer,
                              gradVTVx::Matrix{Float64} = gradVTVx_buffer; opt::Bool=false,
                              Pcoll::t_Pcoll = Pcoll,
                              dr::MVector{dim, Float64} = vec_buffer2,
                              stored::t_stored = get_stored()
                              ) ::Float64

    if N<64
        error("Virial pressure is very inaccurate for N<64")
    end
    M = PIMC.M
    τ = PIMC.τ
    L = PIMC.L
    β = PIMC.β
        

    
    faca_1 = chin_v2
    faca_2 = chin_v1    
    facb_1 = τ^2*chin_u0*(1-2*chin_a1) 
    facb_2 = τ^2*chin_u0*chin_a1
    vir_pot::Float64 = 0.0
    
    
    # Co-moving centroid reference rref for all active beads
    rref = centroid_ref!(PIMC, beads, links)
    # 
    # potential term
    #
     
    @inbounds begin
        for m in 1:M
            bs = beads.active_at_t[m]
            nbs = length(bs)                         
            compute_gradVTV!(PIMC, beads, bs, gradVTVx)
            if (m+1)%3 == 0
                faca = faca_1
                facb = facb_1
            else
                faca = faca_2
                facb = facb_2
            end        
            for i in 1:nbs
                b = bs[i]
                @views gradVm .= faca*stored.F[:, b] + facb*gradVTVx[:, i]
                #vir_pot += dot(view(rref, :, b), gradVm)
                distance!(view(beads.X, :, b), view(rref, :, b), L, dr) # x_m - x_m^*
                #dr .= view(beads.X, :, b)- view(rref, :, b)
                vir_pot += dot(dr, gradVm)  # (x_m - x_m^*)⋅∂Vm/∂x_m 
            end
        end
        vir_pot *= -τ/(2β) # unit: energy; τ/β=3/M
    end # inbounds
    #
    # exchange term, uses boson_virial_exchange_energy from PIMC_Utilities
    #
    vir_exc::Float64 = 0.0
    if bose       
        vir_exc = boson_virial_exchange_energy(PIMC, beads, links) # unit: energy 
    end 

    
    V = L^dim # volume, cubic box
    
    # virial pressure
    # P = N*k*T/Ω  | ideal gas pressure
    #    + 2/(dim*Ω) < 1/M ∑_m=1^M  ∑_m=1^M (x_{M+m}-x_m)⋅(x_{M+m-1}-x_{M+m})/(4π t_{M+m-1}τ) | exchange 
    #                  - τ/2 ∑_m=1^M x_m^* ⋅  ∂\tilde V_m/∂x_m >
    #  :=  N*k*T/Ω  + 2/(dim*Ω) * (vir_exc + vir_pot)
    #
    # Pressure unit for liquid He is K/Å^dim. Convert to bar by multiplying with 138.0649
    #
    P = N/(β*V) + 2/(dim*V)  * (vir_exc + vir_pot) # unit: energy /volume
    
    ΔP::Float64 = 0.0
    #
    if PIMC_Common.sys == :HeLiquid && dim==3
        # finite-size correction
        rcut = PIMC.rc
        ρ = N/V 
        ΔP = -2π/3 * ρ^2 * rcut^3 * PIMC.syspots.pair_potential(rcut)  - ρ*PIMC_Common.Vtail
        P += ΔP
    end
    #
    
    
    if opt
        return P
    end

    # barostat to find equilibrium P=0
    #
    if PIMC.canonical && PIMC_Common.sys == :HeLiquid
        PIMC_Common.TEST=true
        Pcoll.P += P
        Pcoll.n += 1
        #if Pcoll.n % 10 == 0
            @printf("virial pressure %10.5f  terms: classical=%-10.5f exchange=%-10.5f potential=%-10.5f  ΔP=%-10.5f\n",
            P,  N/(β*V), 2/(dim*V)*vir_exc, 2/(dim*V)*vir_pot, ΔP)            
            Vold = V
            V = Vold*(1 + 1e-3 * Pcoll.P/Pcoll.n) # Berentsen barostat, change volume Vold -> V
            L = V^(1/dim) # new box
            PIMC.L = L
            ρ = N/V
            # scale bead coordinates 
            scale = (V/Vold)^(1/dim)
            bs =  beads.ids[beads.active]
            for b in bs
                @views beads.X[:, b] .*= scale
            end
        # update cutoff distance, assuming it's L/2
           PIMC.rc = L/2
            # update tail correction           
            PIMC_Common.Vtail = PIMC_Common.system.tail_correction(ρ, PIMC.rc, L, dim, pbc)
            
            @printf(" Barostat: P %-10.5f new box side %-10.5f new density %-10.5f\n", Pcoll.P/Pcoll.n, L, ρ)
            Pcoll.P = 0.0
            Pcoll.n = 0
        #end        
    end
    #
    
    
    # Collect statistics
    add_sample!(meas.stat, P)
    if meas.stat.finished
        # block full
        ave, std, input_σ2, Nblocks  = get_stats(meas.stat)
        open(meas.filename, "a") do f
            println(f, "$ave $std")
        end
        @printf("Virial Pressure %12.8f ± %12.8f\n", ave, std)
        # set HDF4 data Dict
        results[string(meas.name)] = Dict(
            "Pressure" => ave, 
            "std" => std
        )
        
    end
    return P
end


#
# chin_a1 and chin_t0 optimization 
# ================================
# based on minimizing virial and thermal energy difference
#
function get_ΔE(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement) ::Tuple{Float64, Float64, Float64}
    EVm_store.ipimc = 0 # force recalculation of EVm
    Eth  = meas_E_th(PIMC, beads, links, meas, opt=true)
    Evir = meas_E_vir(PIMC, beads, links, meas, opt=true)
    ΔE = abs(Eth-Evir)
    return ΔE, Eth, Evir
end


mutable struct t_a1_t0
    a1sum::Float64
    t0sum::Float64   
    n::Int
end
a1_t0_col = t_a1_t0(0.0, 0.0, 0)

mutable struct t_E
    Eth::Float64
    Evir::Float64
    ok::Bool
end

E_store = t_E(0.0, 0.0, false)


function opt_chin_a1_chin_t0(PIMC::t_pimc, beads::t_beads, links::t_links, a1_t0_col::t_a1_t0 = a1_t0_col, E_store::t_E=E_store) ::Nothing
    """Optimize Chin action parameters chin_a1 and chin_t0 based on |Eth - Evir|."""
    # dummy stat and measurement
    stat = t_Stat()
    meas = t_measurement(1, :dummy, "dummy", x->(), stat, "none") # (frequency, name, sname, exe, stat, filename)
    best_ΔE, Eth, Evir = get_ΔE(PIMC, beads, links, meas)
    
    # Don't try to optimize before thermalization is almost done!
    #  - check that Eth and Evir fluctuate around the same value, Eth-Evir changes sign
    #  - or Eth < Evir, which usually means we're cool
    if E_store.ok==false
        if Eth < Evir || (Eth-Evir) * (E_store.Eth-E_store.Evir) < 0
            E_store.ok = true
        else
            E_store.Eth  = Eth
            E_store.Evir = Evir
        end
    end
    if E_store.ok==false
        @printf("(Possibly) not thermalized energies, won't optimize chin_a1 and chin_t0;  Eth = %-.5f Evir = %-.5f\n", Eth, Evir)
        return 
    end
                   
    a1 = PIMC.chin_a1
    t0 = PIMC.chin_t0
    orig_a1 = a1
    orig_t0 = t0 

    function find_opt()
        step_a1 = 0.01/sqrt(a1_t0_col.n)
        step_t0 = 0.01/sqrt(a1_t0_col.n)
        max_iter = 10
        tol = 1e-5
        for iter in 1:max_iter
            improved = false
            for (da1, dt0) in ((step_a1, 0.0), (-step_a1, 0.0), (0.0, step_t0), (0.0, -step_t0))
                new_a1 = a1 + da1
                new_t0 = t0 + dt0
                
                # force limits
                if chin_a1_lims[1]  <= new_a1 <= chin_a1_lims[2] && chin_t0_lims[1] <= new_t0 <= chin_t0_lims[2]
                    # update all Chin Action parameters
                    chin_a1, chin_t0 = new_a1, new_t0
                    PIMC.chin_a1 = chin_a1
                    PIMC.chin_t0 = chin_t0
                    set_chin_params(PIMC)
                    # compute new time slices
                    init_action!(PIMC, beads)
                    # 
                    new_ΔE, new_Eth, new_Evir = get_ΔE(PIMC, beads, links, meas)
                    if new_ΔE + tol < best_ΔE
                        a1, t0 = new_a1, new_t0
                        Eth, Evir = new_Eth, new_Evir
                        best_ΔE = new_ΔE
                        @printf("opt chin_a1 = %-15.5f chin_t0 = %-15.5f ΔE = %-15.5f Eth = %-15.5f Evir = %-15.5f\n",
                                a1, t0, best_ΔE, Eth, Evir)
                        improved = true
                    end                    
                end                
            end
            step_a1 *= 0.5
            step_t0 *= 0.5
        end
        return a1, t0, best_ΔE, improved, Eth, Evir
    end
    
    a1, t0, best_ΔE, improved, Eth, Evir = find_opt()
   
    if improved
        a1_t0_col.a1sum += a1
        a1_t0_col.t0sum += t0
    else
        if a1_t0_col.n > 0
            a1_t0_col.a1sum += a1_t0_col.a1sum/a1_t0_col.n
            a1_t0_col.t0sum += a1_t0_col.t0sum/a1_t0_col.n
        else
            a1_t0_col.a1sum += orig_a1
            a1_t0_col.t0sum += orig_t0
        end
    end
    a1_t0_col.n += 1
    # update all Chin action parameters to average so far    
    chin_a1 = a1_t0_col.a1sum/a1_t0_col.n
    chin_t0 = a1_t0_col.t0sum/a1_t0_col.n    
    PIMC.chin_a1 = chin_a1
    PIMC.chin_t0 = chin_t0
    set_chin_params(PIMC)
    # compute new time slices
    init_action!(PIMC, beads)
    open("Chin_opt"*PIMC.filesuffix, "a")  do f
        @printf(f, "%15.5f %15.5f %15.5f %15.5f\n", chin_a1, chin_t0, Eth, Evir)
    end
    @printf("Optimized so far chin_a1 = %-8.5f chin_t0 = %-8.5f\n", chin_a1, chin_t0)
    return nothing
end



function get_EVm(PIMC::t_pimc, beads::t_beads,
                 stored::t_stored = get_stored() 
                 ) ::Float64
    """ ∂β [∑_m=1^M τ V_m(x_m;τ)]"""
    #
    # β = Mτ/3 for Chin Action
    #   ∂β [∑_m=1^M τ V_m(x_m;τ)]
    # = 3/M  ∂τ [∑_m=1^M τ V_m(x_m;τ)]
    # (m+1)%3!=0:
    # =  3/M  ∂τ ∑_m=1^M [τ v_1 V(x_m)  + τ^3 u0a1 VTV(x_m)]
    # =  3/M   ∑_m=1^M [ v_1 V(x_m)  + 3τ^2 u0a1 VTV(x_m)]
    # (m+1)%3==0:
    # =  3/M  ∂τ ∑_m=1^M [τ v_1 V(x_m)  + τ^3 u0(1-2a1) VTV(x_m)]
    # =  3/M ∑_m=1^M [v_1 V(x_m)  + 3τ^2 u0(1-2a1) VTV(x_m)]
    #
    if !has_potential(PIMC.syspots.confinement_potential)  && !has_potential(PIMC.syspots.pair_potential)
        return 0.0
    end

    
    # use stored value if already computed, both ipimc and iworm must be the same   
    if EVm_store.ipimc == PIMC.ipimc && EVm_store.iworm == PIMC.iworm
        return EVm_store.Evalue
    end
    M = PIMC.M
    τ = PIMC.τ

    faca_1 = chin_v2
    faca_2 = chin_v1    
    facb_1 = 3*τ^2*chin_u0*(1-2*chin_a1) 
    facb_2 = 3*τ^2*chin_u0*chin_a1

    EVm::Float64 = 0.0
    @inbounds for m in 1:M
        Vx = stored.V[m]
        VTVx = stored.VTV[m]
        if (m+1)%3==0
            EVm += faca_1 * Vx + facb_1 * VTVx # ∂τ(τ*v1*W_(1-2a1))
        else
            EVm += faca_2 * Vx + facb_2 * VTVx # ∂τ(τ*v1*W_a1)
        end
    end
    EVm *= 3/M    
    # store
    EVm_store.ipimc = PIMC.ipimc
    EVm_store.iworm = PIMC.iworm
    EVm_store.Evalue = EVm
    #
    EVm
end

function compute_ΔV(PIMC::t_pimc, beads::t_beads, rsold::SVector{dim, Float64},
                    bs::AbstractVector{Int}, id::Int, act::Symbol,
                    stored::t_stored = get_stored()
                    ) ::Nothing
    """Computes change in potential energy ΔV to stored.ΔV[m]"""
    if !has_potential(PIMC.syspots.confinement_potential)  && !has_potential(PIMC.syspots.pair_potential)
        m = beads.ts[id]    
        stored.ΔV[m] = 0.0
        return nothing
    end

    L::Float64 = PIMC.L
    rc::Float64 = PIMC.rc
    V_ext = PIMC.syspots.confinement_potential
    Vpair = PIMC.syspots.pair_potential

    
    ΔV::Float64 = 0.0
    rsi::Float64 = 0.0
    
    if act == :move
        rs = @view beads.X[:, id]
        # move from rsold -> rs := beads.X[:, id]        
        if has_potential(V_ext)            
            ΔV += V_ext(rs) - V_ext(rsold) 
        end
        if has_potential(Vpair)
            @inbounds begin
                for b in bs
                    b == id && continue
                    ri = @view beads.X[:, b]
                    rsi = dist(rs, ri, L)
                    if rsi < rc
                        ΔV += Vpair(rsi)
                    end
                    rsi = dist(rsold, ri, L)
                    if rsi < rc
                        ΔV -=  Vpair(rsi)
                    end
                end
            end # inbounds
        end
    elseif act == :remove
        # bead id is removed, there's no new rs
        # change from rsold -> nothing
        if has_potential(V_ext)
            ΔV -= V_ext(rsold) 
        end
        if has_potential(Vpair)
            # potential from all to rsold is subtracted
            @inbounds for b in bs
                b == id && continue # id may already be absent from bs
                rsi = dist(rsold, view(beads.X, :, b), L)
                if rsi < rc
                    ΔV -=  Vpair(rsi)
                end
            end
            ΔV -= PIMC_Common.Vtail # don't forget me!
        end
        
        
    elseif act == :add
        rs = @view beads.X[:, id]
        # bead id is added, rsold is dummy
        # move from nothing -> rs := beads.X[:, id]       
        if has_potential(V_ext)
            ΔV += V_ext(rs) 
        end
        if has_potential(Vpair)
            # potential from rs to rest is added            
            @inbounds for b in bs
                b == id && continue # id may already be absent from bs
                ri = @view beads.X[:, b]                
                rsi = dist(rs, ri, L)
                if rsi < rc
                    ΔV +=  Vpair(rsi)
                end
            end            
            ΔV += PIMC_Common.Vtail # don't forget me!
        end        
    end
    m = beads.ts[id]    
    stored.ΔV[m] = ΔV
    return nothing
end

function compute_ΔF(PIMC::t_pimc, beads::t_beads, rsold::SVector{dim, Float64}, bs::Vector{Int},
                    id::Int, act::Symbol,
                    vec_rsi::MVector{dim, Float64} = vec_buffer,
                    vec_rks::MVector{dim, Float64} = vec_buffer2,
                    stored::t_stored = get_stored()
                    ) ::Nothing
    """ ΔF = ∇ ΔV for each k after change in one bead (move, remove or add) to stored.ΔF[:, bs] """
    L = PIMC.L
    rc = PIMC.rc
    ∇V_ext = PIMC.syspots.grad_confinement_potential   
    V´ =  PIMC.syspots.der_pair_potential
    @inbounds begin
        for b in bs
            for d in 1:dim
                stored.ΔF[d, b] = 0.0
            end
        end
    end
    s = id # shorter name
    
    rks::Float64 = 0.0
    rsi::Float64 = 0.0
    sdVsi::Float64 = 0.0
    sdVks::Float64 = 0.0
    
    if act == :move
        # rsold -> rs := beads.X[:, id]               
        # updates to forces linked to bead id
        #if has_potential(∇V_ext)
        #    rs = SVector{dim}(beads.X[:, s])
        #    @inbounds @views stored.ΔF[:, s] .+= ∇V_ext(rs) - ∇V_ext(rsold)
        #end
        if has_potential(V´)
            @inbounds begin
                for k in bs                    
                    if k == s
                        # change on force acting on s
                        # from s, all other coordinates moved
                        for i in bs
                            i==s && continue                       
                            for d in 1:dim
                                x = beads.X[d, s] - beads.X[d, i] 
                                if pbc
                                    x -= L * round(x/L)
                                end
                                vec_rsi[d] = x
                            end
                            rsi = norm(vec_rsi)
                            if rsi < rc
                                sdVsi = V´(rsi)/rsi
                                for d in 1:dim
                                    stored.ΔF[d, s] += sdVsi*vec_rsi[d]
                                end
                            end                            
                            for d in 1:dim
                                x = rsold[d] - beads.X[d, i] 
                                if pbc
                                    x -= L * round(x/L)
                                end
                                vec_rsi[d] = x
                            end
                            rsi = norm(vec_rsi)
                            if rsi < rc
                                sdVsi = -V´(rsi)/rsi
                                for d in 1:dim
                                    stored.ΔF[d, s] += sdVsi*vec_rsi[d]
                                end
                            end
                        end
                    else
                        # change on force acting on k (≠s)
                        # from k (≠s), only s moved
                        for d in 1:dim
                            x = beads.X[d, k] - beads.X[d, s] 
                            if pbc
                                x -= L * round(x/L)
                            end
                            vec_rks[d] = x
                        end
                        rks = norm(vec_rks)                        
                        if rks < rc
                            sdVks = V´(rks)/rks
                            for d in 1:dim
                                stored.ΔF[d, k] += sdVks*vec_rks[d]
                            end
                        end
                        for d in 1:dim
                            x = beads.X[d, k] - rsold[d]
                            if pbc
                                x -= L * round(x/L)
                            end
                            vec_rks[d] = x
                        end
                        rks = norm(vec_rks)
                        if rks < rc
                            sdVks = -V´(rks)/rks
                            for d in 1:dim
                                stored.ΔF[d, k] += sdVks*vec_rks[d]
                            end
                        end
                    end
                end
            end # inbounds
        end
    elseif act == :remove
        # minus forces linked to bead that was at rsold (s is not in bs)        
        if has_potential(∇V_ext)
            @inbounds @views stored.ΔF[:, s] .-= ∇V_ext(rsold)
        end
        if has_potential(V´)
            @inbounds begin
                for k in bs
                    # change on force acting on k by removing s
                    for d in 1:dim
                        x = beads.X[d, k] - rsold[d]
                        if pbc
                            x -= L * round(x/L)
                        end
                        vec_rks[d] = x
                    end
                    rks = norm(vec_rks)
                    if rks < rc
                        dVks = V´(rks)/rks
                        for d in 1:dim
                            stored.ΔF[d, k] -= dVks*vec_rks[d]
                        end
                    end
                end                            
            end
        end
    elseif act == :add
        # plus forces linked to bead id (s in list bs), no rsold
        # take out old outdated force, if any; an inactive bead should have zero force
        # looks strange, but now stored.F[:, s] + stored.ΔF[:, s] = 0
        @inbounds for d in 1:dim
            stored.ΔF[d, s] = -stored.F[d, s]
        end
        if has_potential(∇V_ext)
            @inbounds @views stored.ΔF[:, s] .+= ∇V_ext(beads.X[:, s])
        end
        if has_potential(V´)
            @inbounds begin
                for k in bs
                    k == s && continue
                    for d in 1:dim
                        x = beads.X[d, k] - beads.X[d, s] 
                        if pbc
                            x -= L * round(x/L)
                        end
                        vec_rks[d] = x
                    end
                    rks = norm(vec_rks)
                    if rks < rc
                        dVks = V´(rks)/rks
                        # force on bead k due to bead s
                        for d in 1:dim
                            dd = dVks*vec_rks[d]
                            stored.ΔF[d, k] = dd # just once for each k
                            # force on bead s due to bead k  
                            stored.ΔF[d, s] -= dd # ∑ k(≠s)
                        end
                    end
                end
            end
        end 
    end
    return nothing
end



function get_ΔVTV(PIMC::t_pimc, beads::t_beads, rsold::SVector{dim, Float64},
                  bs::AbstractVector{Int}, id::Int, act::Symbol,
                  stored::t_stored = get_stored()
                  ) ::Float64
    """Change in [V,[T,V]] = 2λ∑_i |F_i|^2 after one bead move; F->F_old + ΔF"""
    
    # compute stored.ΔF[:, bs]; call here to make sure ΔVTV uses new values   
    compute_ΔF(PIMC, beads, rsold, bs, id, act)
    ΔVTV = 0.0   
    @inbounds begin
        for k in bs        
            # d-loop does ΔVTV += 4λ*stored.F[:, k] ⋅ stored.ΔF[:, k] + 2λ*sum(stored.ΔF[:, k].^2),
            # avoid temporaries of ⋅ and sum
            for d in 1:dim
                ΔF = stored.ΔF[d, k]
                ΔVTV += 4λ * stored.F[d, k] * ΔF + 2λ * ΔF*ΔF
            end
        end
        if act == :remove
            # id was not in list bs
            #ΔVTV -= 2λ*sum(stored.F[:, id].^2)
            @inbounds for d in 1:dim
                F = stored.F[d, id]
                ΔVTV -= 2λ*F*F
            end
        end
        if act == :add        
            #ΔVTV += 2λ*sum(stored.F[:, id].^2)
            @inbounds for d in 1:dim
                F = stored.F[d, id]
                ΔVTV += 2λ*F*F
            end
        end
    end
    return ΔVTV
end

function compute_V!(PIMC::t_pimc,  beads::t_beads, bs::AbstractVector{Int}, m::Int,
                    vec_rki::MVector{dim, Float64} = vec_buffer,
                    stored::t_stored = get_stored()
                    ) ::Nothing
    """ Potential energy on slice, update stored.V_tmp[m]"""
    L = PIMC.L
    rc = PIMC.rc
    V_ext = PIMC.syspots.confinement_potential
    V_pair = PIMC.syspots.pair_potential
    V::Float64 = 0.0
    nbs = length(bs)
    # k, i are running indices in bs

    if has_potential(V_ext)
        @inbounds for b in bs
            V += V_ext(view(beads.X,:,b)) 
        end
    end
        
    if has_potential(V_pair)
        Xs = SVector{dim}.(eachcol(view(beads.X, :, bs)))
        @inbounds begin
            for k in 1:nbs-1
                #rk = @view beads.X[:, bs[k]]
                #rk = SVector{dim}(beads.X[:, bs[k]])
                for i in k+1:nbs
                    #ri = @view beads.X[:, bs[i]]
                    #ri = SVector{dim}(beads.X[:, bs[i]])
                    #rki = dist(rk, ri, L)
                    rki = dist(Xs[k], Xs[i], L)
                    if rki < rc 
                        V += V_pair(rki)
                    end
                end
            end
        end
        V += PIMC_Common.Vtail*nbs 
    end
    stored.V_tmp[m] = V
    return nothing
end

function compute_F!(PIMC::t_pimc, beads::t_beads, bs::AbstractVector{Int},
                    vec_rki::MVector{dim, Float64} = vec_buffer,
                    stored::t_stored=get_stored()
                    ) ::Nothing
    """ Forces F=∇V on beads bs (active beads on the same slice), update stored.F_tmp[:, bs]"""
    L = PIMC.L
    rc = PIMC.rc
    ∇V_ext = PIMC.syspots.grad_confinement_potential   
    V´ = PIMC.syspots.der_pair_potential
    @inbounds begin
        for b in bs
            for d in 1:dim        
                stored.F_tmp[d, b] = 0.0
            end
        end
    end
    # k, i are bead indices
    if has_potential(∇V_ext)        
        @inbounds begin        
            for k in bs
                rk = @view beads.X[:, k]
                @views stored.F_tmp[:, k] .+= ∇V_ext(rk)
            end
        end
    end
    if has_potential(V´)
        @inbounds begin
            for k in bs        
                for i in bs
                    i==k && continue
                    for d in 1:dim
                        x = beads.X[d, k] - beads.X[d, i] 
                        if pbc
                            x -= L * round(x/L)
                        end
                        vec_rki[d] = x
                    end
                    rki::Float64 = norm(vec_rki)
                    if rki < rc
                        dVki::Float64 = V´(rki)/rki
                        for d in 1:dim
                            stored.F_tmp[d, k] += dVki*vec_rki[d]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

#
# VTV(x) = [V,[T,V]](x) = 2 ħ^2/(2m) sum_{i=1}^N |F_i|^2
#                       = 2 λ sum_{i=1}^N |F_i|^2
#  F_i = ∇_i confinement_potential(r_i)+ sum_{j(\ne i)} ∇_i pair_potential(r_{ij}) 
#

const vec_rki_buffer = MVector{dim, Float64}(undef)

# also possible to use X::StridedMatrix{Float64}, didn't see difference 
function get_Vx_F_unused!(PIMC::t_pimc, X::AbstractMatrix{Float64}, k::Int,
                   Fk::MVector{dim, Float64},
                   vec_rki::MVector{dim, Float64}=vec_rki_buffer) ::Float64 
    """Potential Vx and force F (used for [V,[T,V]]) for kth bead on a slice"""
    # Force F returned in argument Fk.
    # This is the *slow* version, used by coordinate scaling derivatives
    L = PIMC.L
    M = PIMC.M
    rc = PIMC.rc
    
    V_ext = PIMC.syspots.confinement_potential
    ∇V_ext = PIMC.syspots.grad_confinement_potential   
    V  = PIMC.syspots.pair_potential
    V´ = PIMC.syspots.der_pair_potential
        
    nbs::Int = size(X,2) # mostly N   
    @views Fk .= 0.0
    Vxk::Float64 = 0.0
    @inbounds begin
        if has_potential(V_ext)
            rk = @view X[:, k]
            Vxk += V_ext(rk) 
            @views Fk .+= ∇V_ext(rk)
        end
        if has_potential(V)
            rk = @view X[:, k]
            for i in 1:nbs
                i==k && continue
                ri = @view X[:, i] # X is indexed with particle index, not bead id
                distance!(rk, ri, L, vec_rki)                
                rki = norm(vec_rki)
                if rki < rc
                    Vp = V´(rki)
                    for d = 1:dim
                        Fk[d] +=  Vp*vec_rki[d]/rki
                    end
                    if i<k
                        Vxk += V(rki) #  ∑_{i(<k)} V(r_ki)
                    end
                end
            end
        end
    end #inbounds
    Vxk
end

# AD version
function get_Vx_F(PIMC::t_pimc, X::AbstractMatrix{T}, k::Int,
                  Fk::AbstractVector{T})  where T<:Real
    """Potential Vxk and force Fk (used for [V,[T,V]]) for kth bead on a slice"""
    #
    # This version is used by coordinate scaling derivatives
    
    L = PIMC.L
    M = PIMC.M
    rc = PIMC.rc
    
    V_ext = PIMC.syspots.confinement_potential
    ∇V_ext = PIMC.syspots.grad_confinement_potential   
    V  = PIMC.syspots.pair_potential
    V´ = PIMC.syspots.der_pair_potential
        
    nbs = size(X,2) # mostly N
    @inbounds for i in eachindex(Fk)
        Fk[i] = zero(T)  
    end
    Vxk = zero(T)
    @inbounds begin
        if has_potential(V_ext)
            rk = @view X[:, k]
            Vxk += V_ext(rk) 
            @views Fk .+= ∇V_ext(rk)
        end
        if has_potential(V)
            #vec_rki = zeros(T, dim)   # allocates
            vec_rki = MVector{dim, T}(undef) # a lot faster, but allocates   
            #rk = @view X[:, k]
            for i in 1:nbs
                i==k && continue
                #ri = @view X[:, i] # X is indexed with particle index, not bead id
                #distance!(rk, ri, L, vec_rki)
                for d in 1:dim
                    x = X[d, k] - X[d, i] 
                    if pbc
                        x -= L * round(x/L)
                    end
                    vec_rki[d] = x
                end                
                rki = norm(vec_rki)
                if rki < rc
                    Vp = V´(rki)
                    for d in 1:dim
                        Fk[d] +=  Vp*vec_rki[d]/rki
                    end
                    if i<k
                        Vxk += V(rki) #  ∑_{i(<k)} V(r_ki)
                    end
                end
            end
        end
    end #inbounds
    Vxk
end
const F_buffer = MVector{dim, Float64}(undef)

function Vx_and_VTVx_unused(PIMC::t_pimc, X::AbstractMatrix{Float64},
                     F::MVector{dim, Float64} = F_buffer)
    """Potential V(x) and commutator [V[T,V]](x) = 2λ|F(x)|^2 = 2λ|∂_x V(x)|^2 for one slice"""
    if !has_potential(PIMC.syspots.confinement_potential)  && !has_potential(PIMC.syspots.pair_potential)
        return 0.0, 0.0
    end
    
    nbs::Int = size(X,2) # mostly N    
    Vx::Float64 = 0.0
    VTVx::Float64 = 0.0    
    @inbounds for k in 1:nbs
        V = get_Vx_F(PIMC, X, k, F) # F is force on kth bead
        s = 0.0
        @inbounds for i in eachindex(F)
            s += F[i]*F[i] 
        end
        VTVx += 2λ*s # s is here sum(F.^2)
        Vx += V 
    end
    # add tail correction (default Vtail is zero)
    Vx += PIMC_Common.Vtail * nbs
    Vx, VTVx
end

# AD version
function Vx_and_VTVx(PIMC::t_pimc, X::AbstractMatrix{T}) where T<:Real
    """Potential V(x) and commutator [V[T,V]](x) = 2λ|F(x)|^2 = 2λ|∂_x V(x)|^2 for one slice"""
    if !has_potential(PIMC.syspots.confinement_potential)  && !has_potential(PIMC.syspots.pair_potential)
        return zero(T), zero(T)
    end
    
    nbs = size(X,2) # mostly N
    z = zero(T)
    Vx = z
    VTVx = z
    #F = zeros(T, dim) # allocates; tags change, so preallocation is tricky (see make_Vtilde_buffered)
    F = MVector{dim, T}(undef) # faster
    @inbounds  for k in 1:nbs
        V = get_Vx_F(PIMC, X, k, F) # F is force on kth bead
        s = z
        @inbounds for i in eachindex(F)
            s += F[i]*F[i] 
        end
        VTVx += 2λ*s 
        Vx += V 
    end
    # add tail correction (default Vtail is zero)
    Vx += PIMC_Common.Vtail * nbs
    Vx, VTVx
end


const Vtilde_cache = Dict{DataType, Any}()
const dr_Vtilde_buffer = MVector{dim, Float64}(undef)

if Nthreads>1
    const dr_Vtilde_buffer_th = [MVector{dim, Float64}(undef) for _ in 1:Nthreads]
else
    const dr_Vtilde_buffer_th = [MVector{dim, Float64}(undef)]
end

function make_Vtilde_buffered(PIMC::t_pimc, beads::t_beads, links::t_links, rref::Matrix{Float64},
                              dr::MVector{dim, Float64} = dr_Vtilde_buffer,
                              dr_th::Vector{MVector{dim, Float64}} = dr_Vtilde_buffer_th,
                              cache = Vtilde_cache)
    """ Closure to get Vtilde with pre-allocated Xs used in AD """
    
    function Vtilde(β´)
        """ Vtilde_m(x_m^s; τ) """
        M = PIMC.M
        β = PIMC.β
        L = PIMC.L
        τ = PIMC.τ
        # β is external constant,only scaling is derived wrt. β´
        # τ-derivative terms are computed analytically in get_EVm()
        
        T = typeof(β´)
        s = sqrt(β´/β) - one(T) # to be derived wrt. β´
        res = zero(T)  #  res=0.0 leads to type instability
        # use cache _Xs; trailing Matrix{T} makes cache type stable
        Xs = get!(cache, T) do
            #println("PIMC_Chin_Action: Allocating new Xs for scaled coordinates") # just to see it's done only once
            Matrix{T}(undef, dim, N_slice_max)
        end ::Matrix{T}
        
        # without pre-allocated cache: 
        #_Xs_buffer = Matrix{T}(undef, dim, N_slice_max)
        #
        # Co-moving centroid reference for all active beads
        # no real need to call here, pre-filled the rref cache before calling make_Vtilde_buffered
        # centroid_ref!(PIMC, beads, links)
        # 
        # Fill the first nbs entries of the oversized Xs, and send to Vx_and_VTVx the view to 1:nbs
        @inbounds begin
            for m in 1:M
                bs = beads.active_at_t[m]
                nbs = length(bs)
                if Nthreads>1
                    # not really faster
                    Threads.@threads for i in 1:nbs
                        tid = Threads.threadid()
                        b = bs[i]                                
                        distance!(beads.X[:, b], rref[:, b], L, dr_th[tid])
                        Xs[:, i] .= beads.X[:, b] + s*dr_th[tid] # r_m + s*dr
                    end
                else
                    for i in 1:nbs
                        b = bs[i]
                        distance!(view(beads.X, :, b), rref[:, b], L, dr)
                        for d in 1:dim
                            Xs[d, i] = beads.X[d, b] + s*dr[d]                            
                        end
                    end
                end
                # V and VTV at scaled coordinates (slow, can't use update and stored values)
                Vxs, VTVxs = Vx_and_VTVx(PIMC, view(Xs,:,1:nbs))
                # NB: argument τ, because it's not derived by AD
                res += V_m(m, Vxs, VTVxs, τ) 
            end
        end # inbounds
        res  # unit is energy
    end
    return Vtilde
end

# centered difference
@inline function cent_diff(f, x, h = 1e-6)
    (f(x + h) - f(x - h))/(2h)
end

function Evir_chin_scaled(PIMC::t_pimc, beads::t_beads, links::t_links) ::Float64
    """Scaled potential derivative ∑_m ∂β'[Vtilde_m(x_m^s; τ')]|β'=β, used in virial energy estimator """
    
    if !has_potential(PIMC.syspots.confinement_potential) && !has_potential(PIMC.syspots.pair_potential)
        return 0.0
    end    
    M = PIMC.M
    β = PIMC.β
    rref = centroid_ref!(PIMC, beads, links) 
    Vtilde = make_Vtilde_buffered(PIMC, beads, links, rref)
    Evir = 3/M * β * ForwardDiff.derivative(Vtilde, β) # AD
    # Evir = 3/M * β * cent_diff(Vtilde, β) # centered difference (just for checking)
    return Evir    
end



function Evir_chin_direct(PIMC::t_pimc, beads::t_beads, links::t_links,
                          gradVm::MVector{dim, Float64} = vec_buffer,
                          dx::MVector{dim, Float64} = vec_buffer2,
                          # oversized buffers, just ignore last unset entries
                          gradVTVx::Matrix{Float64} = gradVTVx_buffer,
                          stored::t_stored = get_stored()
                          ) ::Float64
    """Virial energy estimator term 1/(2β)*sum_{m=1}^M (x_m-x_m*)⋅ ∂/∂x_m(τ tildeV_m(x_m;τ)) """
    # = τ/(2β)*sum_{m=1}^M (x_m-x_m*)⋅ ∂/∂x_m(tildeV_m(x_m;τ))
    #
    if !has_potential(PIMC.syspots.confinement_potential) && !has_potential(PIMC.syspots.pair_potential)
        return 0.0
    end
    
    M = PIMC.M    
    L = PIMC.L
    τ = PIMC.τ
    β = PIMC.β 
      
    Vx::Float64 = 0.0
    VTVx::Float64 = 0.0
    Evir_chin::Float64 = 0.0

    rref = centroid_ref!(PIMC, beads, links) # fills rref 

    faca_1 = chin_v2
    faca_2 = chin_v1    
    facb_1 = τ^2*chin_u0*(1-2*chin_a1) 
    facb_2 = τ^2*chin_u0*chin_a1
    #
    @inbounds begin
        for m in 1:M
            bs = beads.active_at_t[m]
            nbs = length(bs)
            compute_gradVTV!(PIMC, beads, bs, gradVTVx)
            if (m+1)%3 == 0
                faca = faca_1
                facb = facb_1
            else
                faca = faca_2
                facb = facb_2
            end
            for i in 1:nbs
                b = bs[i]
                @views gradVm .= faca*stored.F[:, b] + facb*gradVTVx[:, i]                   
                distance!(view(beads.X, :, b), view(rref, :, b), L, dx)  # don't view an MMatrix 
                Evir_chin += dx ⋅ gradVm
            end
        end
    end # inbounds    
    Evir_chin *= τ/(2β)
    return Evir_chin
end


const vec_buffer_a =  MVector{dim, Float64}(undef)
const ∇k∇nV_buffer =  MMatrix{dim, dim, Float64}(undef)
const dimdim_buffer =  MMatrix{dim, dim, Float64}(undef)
const Idim = SMatrix{dim, dim}(I)

# used only in Evir_chin_direct and in virial pressure
function compute_gradVTV!(PIMC::t_pimc, beads::t_beads, bs::AbstractVector{Int},
                          gradVTVx::Matrix{Float64},
                          vecr::MVector{dim, Float64} = vec_buffer_a,
                          ∇k∇nV::MMatrix{dim, dim, Float64} = ∇k∇nV_buffer,
                          dimdim::MMatrix{dim, dim, Float64} = dimdim_buffer,
                          stored::t_stored = get_stored(),
                          Idim::SMatrix{dim, dim} = Idim                           
                          ) ::Nothing

    
    if !has_potential(PIMC.syspots.confinement_potential) && !has_potential(PIMC.syspots.pair_potential)
        return nothing
    end
    
    V_ext = PIMC.syspots.confinement_potential
    ∇V_ext = PIMC.syspots.grad_confinement_potential
    ∇2V_ext = PIMC.syspots.grad2_confinement_potential
    V  = PIMC.syspots.pair_potential
    V´ = PIMC.syspots.der_pair_potential
    V´´ = PIMC.syspots.der2_pair_potential
   
    L = PIMC.L
    M = PIMC.M
    rc = PIMC.rc
    
    
    nbs = length(bs)
    

    if has_potential(V_ext)
        Xb = @view beads.X[:, bs]
        Fb = @view stored.F[:, bs]
        @inbounds @views begin
            for i in 1:nbs                
                gradVTVx[:, i] .= 4λ*∇2V_ext(Xb[:,i])*Fb[:, i]
            end
        end
    end
    if has_potential(V)        
        #
        @inbounds @views gradVTVx .= 0.0    
        @inbounds begin
            for k in 1:nbs
                bk = bs[k]
                # diagonals, k=n in ∇k∇nV
                ∇k∇nV .= 0.0       
                for i in 1:nbs                        
                    if i==k
                        continue
                    end                    
                    distance!(beads.X, bs[i], bk, L, vecr)
                    r = norm(vecr)
                    if r < rc
                        invr = 1/r
                        vecr .*= invr  # unit vector
                        f1 = V´(r)*invr
                        f2 = ( V´´(r) - f1 )
                        ∇k∇nV .+= f2 * (vecr * vecr') + f1 * Idim
                    end
                end
                gradVTVx[:, k] .+=  4λ * ∇k∇nV * stored.F[:, bk] 
                # off-diagonals, k!=n in ∇k∇nV, symmetric         
                for n in 1:k-1
                    bn = bs[n]
                    distance!(beads.X, bk, bn, L, vecr)
                    r = norm(vecr)
                    if r < rc
                        invr = 1/r
                        vecr .*= invr   # unit vector               
                        f1 = -V´(r)*invr
                        f2 = -(V´´(r) +  f1)
                        
                        ∇k∇nV .= f2 * (vecr * vecr') + f1 * Idim
                        @views gradVTVx[:, k] .+=  4λ * ∇k∇nV * stored.F[:, bn]
                        @views gradVTVx[:, n] .+=  4λ * ∇k∇nV * stored.F[:, bk]  #Fk                       
                    end                    
                end
            end
        end # inbounds
                        
        
        #=
        # 
        # same with ∇∇V calculation done separately, later contracted with stored force
        # 
        # Preallocate full tensor; makes this slower for larger nbs
        Xb = @view beads.X[:, bs]
        Fb = @view stored.F[:, bs]
        ∇∇V = [@MMatrix zeros(dim, dim) for _ in 1:nbs, _ in 1:nbs] # Hessian
        for k in 1:nbs, n in 1:k-1
            rn = @view Xb[:, n]
            rk = @view Xb[:, k]
            distance!(rk, rn, L, vecr)
            r = norm(vecr)
            if r < rc
                hatr .= vecr / r
                f1 = -V´(r)/r
                f2 = -(V´´(r) + f1)
                ∇∇V[k,n] .= f2*(hatr*hatr') + f1*I
                ∇∇V[n,k] .= ∇∇V[k,n]                # symmetry
            end
        end
        # diagonals
        for k in 1:nbs
            ∇∇V[k,k] .= -sum(∇∇V[k,n] for n in 1:nbs if n≠k)
        end
        # contraction
        gradVTVx .= 0
        for k in 1:nbs
            for n in 1:nbs
                gradVTVx[:,k] .+= 4λ * ∇∇V[k,n] * @view Fb[:,n]
            end
        end
        for k in 1:nbs            
            @show k, gradVTVx[:, k]
        end
        exit()
        =#
        
    end
    

    # finite difference checking:
    #=
    h = 1e-6
    _, VTV_ = Vx_and_VTVx(PIMC, view(beads.X, :, bs))
    gr = zeros(dim)
    for k in 1:nbs
        
        for d in 1:dim
            beads.X[d, bs[k]] += h
            _, VTV_plus = Vx_and_VTVx(PIMC, view(beads.X, :, bs))
            beads.X[d, bs[k]] -= h
            gr[d] = (VTV_plus - VTV_)/h
        end
        println("$k $(gradVTVx[:,k])  $gr")
    end
    =#
    
    #
        
    # HO exact gradVTVx is known:
    # for i in 1:nbs
    #   @show gradVTVx[:, i] - 4λ*beads.X[:, bs[i]]
    # end
    return nothing
end


# Modified potential
# ==================
function V_m(m::Int, Vx::Float64, VTVx::Float64, τ::Float64) ::Float64
    """Chin Action modified potential with V and [V,[T,V]] on slice m"""
    if (m+1)%3 == 0        
        V_m = chin_v2*Vx + τ^2*chin_u0*(1-2*chin_a1) * VTVx # v2*W_(1-2a1)
    else
        V_m = chin_v1*Vx + τ^2*chin_u0*chin_a1 * VTVx   # v1*W_a1
    end
    V_m
end

# AD version
function V_m(m::Int, Vx::T, VTVx::T, τ::Float64) ::T where T<:Real  
    """Chin Action modified potential with V and [V,[T,V]] on slice m"""
    # τ is not derived with AD, hence Float64
    if (m+1)%3 == 0        
        V_m = chin_v2*Vx + τ^2*chin_u0*(1-2*chin_a1) * VTVx # v2*W_(1-2a1)
    else
        V_m = chin_v1*Vx + τ^2*chin_u0*chin_a1 * VTVx   # v1*W_a1
    end
    V_m
end

# Kinetic action
# ==============
function K(PIMC::t_pimc, beads::t_beads, links::t_links) ::Float64
    """Kinetic action of the whole PIMC config"""
    β = PIMC.β
    L = PIMC.L
    K::Float64 = 0.0 
   
    @inbounds for id in beads.ids[beads.active]
        id_next = links.next[id]
        Δr2 = dist2(beads.X, id, id_next, L)        
        Δτ = wrapβ(beads.times[id_next] - beads.times[id], β)         
        K += dim/2*log(4π*λ*Δτ) + Δr2/(4λ*Δτ) 
    end
    K
end

function K(PIMC::t_pimc, beads::t_beads, links::t_links, id::Int) ::Float64
    """Kinetic action of bead id to next and to previous"""    
    β = PIMC.β
    L = PIMC.L
    
    # link to next bead
    id_next = links.next[id]      
    Δr2  = dist2(beads.X, id, id_next, L)
    Δτ = wrapβ(beads.times[id_next] - beads.times[id], β)
    
    # exp(-K) is (4π*λ*Δτ)^{-dim/2}*exp(-Δr^2/(4*λ*Δτ))
    K = dim/2*log(4π*λ*Δτ) +  Δr2/(4λ*Δτ)

    # link to previous bead
    id_prev = links.prev[id]
    Δr2  = dist2(beads.X, id, id_prev, L)
    Δτ = wrapβ(beads.times[id] - beads.times[id_prev], β)    
    K  += dim/2*log(4π*λ*Δτ) + Δr2/(4λ*Δτ)
end



# Inter-action
# ============
mutable struct t_counter
    n::Int
end
const counter = t_counter(0)


function update_stored_slice_data(PIMC::t_pimc, beads::t_beads, id::Int,
                                  stored::t_stored = get_stored(),
                                  counter::t_counter = counter,
                                  ) ::Nothing
    """Move accepted: Update stored values of V, VTV and F from their tmp values"""
    counter.n += 1
    if counter.n>100000
        # trigger full update in next call to U_stored to avoid error accumulation in updates
        init_stored(PIMC, beads)
        counter.n = 0
    end
    m = beads.ts[id]
    bs = beads.active_at_t[m]
    
    stored.V[m] = stored.V_tmp[m]   
    stored.VTV[m] = stored.VTV_tmp[m]
    @inbounds begin
        for b in bs
            for d in 1:dim                
                stored.F[d, b] = stored.F_tmp[d, b]
            end
        end
    end

    if false
        PIMC_Common.TEST = true
        # check SLOW
        #println("updating slice $m, beads $bs")
        compute_V!(PIMC, beads, bs, m) # new stored.V_tmp
        compute_F!(PIMC, beads, bs)
        a = maximum(abs.(stored.V_tmp[m].-stored.V[m])) |>round
        if !isapprox(a, 0.0)
            @show a
            error("bad V update")
        end
        a = maximum(abs.(stored.F_tmp[:, bs].-stored.F[:, bs]))|>round
        if !isapprox(a, 0.0)
            @show a
            error("bad F update")
        end
        a = maximum(abs.(stored.VTV_tmp[m].-stored.VTV[m]))|>round
        if !isapprox(a, 0.0)
            @show a
            error("bad VTV update")
        end
    end
    return
end


function init_stored(PIMC::t_pimc, beads::t_beads,
                     stored::t_stored=get_stored()
                     ) ::Nothing
    """Computing and storing V, VTV and F for each slice from scratch"""
    #println("Chin Action init_stored:: (re)initializing action-specific stored values")
    @inbounds begin
        for m in 1:PIMC.M
            bs = beads.active_at_t[m]
            compute_V!(PIMC, beads, bs, m) # sets stored.V_tmp[m]
            compute_F!(PIMC, beads, bs)    # sets stored.F_tmp[:, bs]
            stored.V[m] = stored.V_tmp[m]
            s = 0.0
            for b in bs
                for d in 1:dim
                    stored.F[d, bs] = stored.F_tmp[d, bs]
                    F = stored.F[d, b]
                    s += F*F
                end
            end
            stored.VTV[m] = 2λ*s
        end
    end 
    stored.set = true
    return nothing
end


function U_stored(PIMC::t_pimc, beads::t_beads, b::Int,
                  stored::t_stored=get_stored()
                  ) ::Float64
    """Inter-action before changes using stored V and VTV"""    
    @assert stored.set "Can't use U_stored before stored values are initialized in init_stored"
    m = beads.ts[b]
    τ = PIMC.τ
    # U could be stored, but recomputation from stored V and VTV is fast
    U = τ * V_m(m, stored.V[m], stored.VTV[m], τ)
    U
end

function U_update(PIMC::t_pimc, beads::t_beads, xold::AbstractVector{Float64}, id::Int, act::Symbol,                  
                  fake::Bool=false,
                  stored::t_stored=get_stored()
                  ) ::Float64
    """Inter-action after *suggested* change xold -> beads.X[:, id]"""   
    #@assert stored.set "Can't use U_update before stored values are initialized in init_stored"

    m = beads.ts[id]
    τ = PIMC.τ    
    bs = beads.active_at_t[m]
    xold_s = SVector{dim}(xold) # convert to SVector for speed
    
    compute_ΔV(PIMC, beads, xold_s, bs, id, act)
    ΔV = stored.ΔV[m]
    ΔVTV = get_ΔVTV(PIMC, beads, xold_s, bs, id, act) # computes also stored.ΔF[:, bs]
    ΔF = stored.ΔF

    Vx = stored.V[m] + ΔV
    VTVx = stored.VTV[m] +  ΔVTV
    Unew = τ * V_m(m, Vx, VTVx, τ)

    # If this is fake close used in obdm measurement, don't mess with the tmp storage values!
    if fake        
        return Unew
    end
   
    # update tmp storage
    stored.V_tmp[m] = Vx
    stored.VTV_tmp[m]  = VTVx
    @inbounds begin
        for d in 1:dim            
            stored.F_tmp[d, bs] = stored.F[d, bs] + ΔF[d, bs]
        end
    end

    if false
        PIMC_Common.TEST = true
        # check SLOW
        
        Vx = stored.V_tmp[m]
        VTVx = stored.VTV_tmp[m]
        Vx_, VTVx_ = Vx_and_VTVx(PIMC, view(beads.X, :, bs))
        if !isapprox(Vx_, Vx, atol=1e-8)
            @show Vx_, Vx
            @show act
            error("wrong V")
        end
        if !isapprox( VTVx_, VTVx, rtol=1e-8)
            @show VTVx_, VTVx
            @show act
            error("wrong VTV")
        end
        UU =  U(PIMC, beads, id)
        if !isapprox(UU, Unew, atol=1e-8)        
            @show Unew, UU, Unew - UU
            @show act
            error("U is wrong")
        end
    end
    
    Unew
end

function U(PIMC::t_pimc, beads::t_beads, id::Int) ::Float64
    """Inter-action τ*U of bead id"""
    # SLOW
    PIMC_Common.TEST = true
    m = beads.ts[id]
    τ = PIMC.τ
    m = beads.ts[id]
    bs = beads.active_at_t[m]
    Vx, VTVx = Vx_and_VTVx(PIMC, view(beads.X,:,bs))
    U = τ * V_m(m, Vx, VTVx, τ)
    U
end

    

function U(PIMC::t_pimc, beads::t_beads, m::Int, τ) ::Float64
    """Inter-action τ*U of beads at slice m, variable τ to be used in AD"""
    PIMC_Common.TEST = true
    # SLOW 
    bs = beads.active_at_t[m]
    Vx, VTVx = Vx_and_VTVx(PIMC, view(beads.X,:,bs))
    U = τ * V_m(m, Vx, VTVx, τ)
    U
end
        
function U(PIMC::t_pimc, beads::t_beads) ::Float64
    """Inter-action τ*U of the whole PIMC"""
    PIMC_Common.TEST = true
    # SLOW 
    τ = PIMC.τ    
    U::Float64 = 0.0 
    @inbounds for m in 1:PIMC.M
        bs = beads.active_at_t[m]
        Vx, VTVx = Vx_and_VTVx(PIMC, view(beads.X,:,bs))   
        U += V_m(m, Vx, VTVx, τ)        
    end
    τ*U
end



end
