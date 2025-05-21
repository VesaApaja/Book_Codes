__precompile__(false)
module PIMC_Chin_Action
using Printf
using ForwardDiff
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1) 
using StaticArrays

push!(LOAD_PATH,".")
using PIMC_Structs
using QMC_Statistics
using PIMC_Systems
using PIMC_Common
using PIMC_Measurements: boson_virial_exchange_energy, E_pot_all
using PIMC_Reports: report_energy
using PIMC_Utilities: active_beads_on_slice

# slightly faster norm 
@inline norm(x) = sqrt(sum(abs2,x))

export U, K, U_stored, U_update, update_stored_slice_data
export meas_E_th, meas_E_vir, init_action!


# M = 6, times set in PIMC_Structs
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


# Chin Action parameters (Chin and Chen, J. Chem. Phys. 117, 1409 (2002))
const chin_a1 = 0.33    # free parameter, ∈[0,1/2]   0.33 for harmonic oscillator (Boronat '09)
const chin_t0 = 0.1215  # free parameter, ∈[0,1/2*(1-1/sqrt(3))]=0.21 , 0.1215 for harmonic oscillator (Boronat '09)
const chin_t1 = 1/2 - chin_t0
const chin_u0 = 1/12*( 1 - 1/(1-2*chin_t0) + 1/(6*(1-2*chin_t0)^3) )
const chin_v1 = 1/(6*(1-2*chin_t0)^2)
const chin_v2 = 1-2*chin_v1

# light-weight vector buffers, not thread-safe!
const vec_buffer = Vector{Float64}(undef, dim)
const vec_buffer2 = Vector{Float64}(undef, dim)
const mat_buffer = Array{Float64, 2}(undef, dim, N*1000) # for r_ref

# Storage for V(x) and [V,[T,V]](x) and forced F on each slice
mutable struct t_stored
    set::Bool
    V::Vector{Float64}
    VTV::Vector{Float64}
    F::Array{Float64}
    ΔV::Vector{Float64}
    ΔF::Array{Float64}
    V_tmp::Vector{Float64}
    VTV_tmp::Vector{Float64}
    F_tmp::Array{Float64}
    function t_stored(M::Int64, Nb::Int64)
        new(false, zeros(M), zeros(M), zeros(dim, Nb), zeros(M), zeros(dim, Nb), zeros(M), zeros(M), zeros(dim, Nb))  
    end
end
stored = t_stored(1, 1) # dummy init

# Vm energy storage (to avoid both E_th and E_vir evaluating the same EVm)
mutable struct t_estore
    ipimc::Int64
    iworm::Int64
    Evalue::Float64
end
EVm_store = t_estore(0, 0, 0.0)


function init_action!(PIMC::t_pimc, beads::t_beads)
    """Init Chin Action"""
    global stored
    println("Init Chin Action")
    β = PIMC.β
    M = PIMC.M
    τ = 3*β/M    
    PIMC.τ = τ
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
    for m = 1:M
        bs = beads.at_t[m]
        for id ∈ bs
            beads.times[id] = chin_times[m]
        end
    end
    stored = t_stored(M, length(beads.ids))
end


# Thermodynamic energy estimator
# ==============================


function meas_E_th(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    M = PIMC.M    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
    # kinetic energy terms
    NM = count(beads.active)  # instantaneous N*M, in case of GC
    NN = NM/M                  # instantaneous N

    PIMC.canonical && NM != N*M && error("NM is not N*M, bead activation problem?")
    
    Ekin1 = dim*NM/(2*β)
    Ekin2::Float64 = 0.0
    active = findall(beads.active)
    @inbounds for id in active
        id_next = links.next[id]
        Δr2 = dist2(view(beads.X, :, id), view(beads.X, :, id_next), L)        
        Δτ = mod(beads.times[id_next] - beads.times[id], β)
        Ekin2 += Δr2/Δτ       
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

    # Collect statistics
    add_sample!(meas.stat, [E, Ekin, EVm])
    if meas.stat.finished
        # block full, report
        report_energy(PIMC, meas)        
    end
end


function meas_E_vir(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    """Virial energy estimator"""

    M = PIMC.M    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
    #NN = count(beads.active)/M  # instantaneous N for GC

    Ekin = dim*N/(2*β)  # exact distinguishable NN particle kinetic energy
    Eexc::Float64 = 0.0 
    if bose
        # exchange energy term        
        Eexc = boson_virial_exchange_energy(PIMC, beads, links)
        Ekin += Eexc
    end

    # Virial terms are
    #   sum_m [∂β (τ Vtilde_m(x_m;τ))] + 1/(2β) sum_m (x_m - x_ref_m) ⋅ ∂x_m [τ  Vtilde_m(x_m;τ)]
    # = M/3 sum_m [∂τ (τ Vtilde_m(x_m;τ))] + 3/M 1/2 sum_m (x_m - x_ref_m) ⋅ ∂x_m Vtilde_m(x_m;τ)
    #  get_EVm() := [∂τ (τ Vtilde_m(x_m;τ))] is computed analytically 
    #  Evir_chin() := 3/M 1/2 sum_m (x_m - x_ref_m) ⋅ ∂x_m Vtilde_m(x_m;τ) 

    
    # term τ*sum_m ∂β'[Vtilde_m(x_m^s; τ')]|β'=β
    # =   3/M *β * sum_m ∂β'[Vtilde_m(x_m^s; τ')]|β'=β]
    Evir = Evir_chin(PIMC, beads, links)
    
    Ekin += Evir 
    #
    # term 3/M * sum_m ∂β[β * Vtilde_m(x_m; τ)]
    EVm = get_EVm(PIMC, beads)
    Ekin /= N
    EVm /= N
    E = Ekin + EVm 

    # Collect statistics
    add_sample!(meas.stat, [E, Ekin, EVm])
    if meas.stat.finished
        # block full, report
        report_energy(PIMC, meas)        
    end
   
end

function get_EVm(PIMC::t_pimc, beads::t_beads)
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
    # use stored value if already computed, both ipimc and iworm must be the same    
    if EVm_store.ipimc == PIMC.ipimc && EVm_store.iworm == PIMC.iworm
        return EVm_store.Evalue
    end
    M = PIMC.M
    τ = PIMC.τ
    EVm = 0.0
    @inbounds for m in 1:M
        Vx = stored.V[m]
        VTVx = stored.VTV[m]
        if (m+1)%3 == 0
            EVm += chin_v2*Vx + 3*τ^2*chin_u0*(1-2*chin_a1) * VTVx # ∂τ(τ*v1*W_(1-a1))
        else
            EVm += chin_v1*Vx + 3*τ^2*chin_u0*chin_a1 * VTVx # ∂τ(τ*v1*W_a1)
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


function compute_ΔV(PIMC::t_pimc, beads::t_beads, rsold::SubArray{Float64}, bs::Vector{Int64}, id::Int64, act::Symbol)
    """Computes change in potential energy ΔV to stored.ΔV[m]"""
     
    L = PIMC.L
    V_ext = PIMC.confinement_potential
    Vpair = PIMC.pair_potential
    
    ΔV = 0.0
    if act == :move
        # move from rsold -> rs := beads.X[:, id] 
        rs = @view beads.X[:, id]
        if V_ext !== nothing
            ΔV += V_ext(rs) - V_ext(rsold) 
        end
        if Vpair !== nothing
            vec_rsi = zeros(dim)           
            s = findfirst(x -> x == id, bs) # bs[s] = id
            X = @view beads.X[:, bs]
            nbs = length(bs)
            # i is runnign index in bs
            @inbounds for i in 1:nbs                
                i==s && continue
                ri = @view X[:, i]                
                @inbounds for (sign, rs_) in zip([+1, -1], [rs, rsold])
                    distance!(rs_, ri, L, vec_rsi)                
                    rsi = norm(vec_rsi)
                    pbc && (rsi > L/2) && continue # pbc cutoff 
                    ΔV += sign*Vpair(rsi)
                end
            end
        end
    elseif act == :remove
        # bead id is removed, there's no new rs
        # move from rsold -> nothing
        if V_ext !== nothing
            ΔV +=  -V_ext(rsold) 
        end
        if Vpair !== nothing            
            # potential from all to rsold is subtracted
            vec_rsi = zeros(dim)
            X = @view beads.X[:, bs]
            nbs = length(bs) 
            @inbounds for i in 1:nbs
                ri = @view X[:, i]
                distance!(rsold, ri, L, vec_rsi)                
                rsi = norm(vec_rsi)
                pbc && (rsi > L/2) && continue # pbc cutoff 
                ΔV += -Vpair(rsi)
            end
            ΔV -= PIMC_Common.Vtail # don't forget me!
        end
        
    elseif act == :add
        # bead id is added, rsold is dummy
        # move from nothing -> rs := beads.X[:, id]
        rs = @view beads.X[:, id]
        if V_ext !== nothing
            ΔV += V_ext(rs) 
        end
        if Vpair !== nothing            
            # potential from rs to rest is added
            vec_rsi = zeros(dim)
            @inbounds for i in bs
                i == id && continue # id may already be absent from bs
                ri = @view beads.X[:, i] 
                distance!(rs, ri, L, vec_rsi)                
                rsi = norm(vec_rsi)
                pbc && (rsi > L/2) && continue # pbc cutoff 
                ΔV += Vpair(rsi)
            end            
            ΔV += PIMC_Common.Vtail # don't forget me!
        end        
    end
    m = beads.ts[id]    
    stored.ΔV[m] = ΔV
    return nothing
end
function compute_ΔF(PIMC::t_pimc, beads::t_beads, rsold::SubArray{Float64}, bs::Vector{Int64}, id::Int64, act::Symbol)
    """ ΔF = ∇ ΔV for each k after change in one bead (move, remove or add) to stored.ΔF[:, bs] """
    
    L = PIMC.L 
    ∇V_ext = PIMC.grad_confinement_potential   
    derV =  PIMC.der_pair_potential
    @inbounds for b in bs
        @inbounds for d in 1:dim
            stored.ΔF[d, b] = 0.0
        end
    end
    s = id # shorter name
    rs = @view beads.X[:, s]
    
    
    if act == :move
        # rsold -> rs := beads.X[:, id]               
        # updates to forces linked to bead id 
        if ∇V_ext !== nothing            
            @inbounds @views stored.ΔF[:, s] .+= ∇V_ext(rs) - ∇V_ext(rsold)
        end
        if derV !== nothing 
            vec_rsi = vec_buffer
            vec_rks = vec_buffer2
            @inbounds for k in bs
                if k == s
                    # change on force acting on s
                    # from s, all other coordinates moved
                    @inbounds for i in bs
                        i==s && continue                       
                        ri = @view beads.X[:, i]
                        @inbounds for (sign, rs_) in zip([+1, -1], [rs, rsold])
                            distance!(rs_, ri, L, vec_rsi)
                            rsi = norm(vec_rsi)
                            pbc && (rsi > L/2) && continue #
                            @inbounds for d in 1:dim
                                stored.ΔF[d, s] += sign*derV(rsi)*vec_rsi[d]/rsi
                            end
                        end
                    end
                else
                    # change on force acting on k (≠s)
                    # from k (≠s), only s moved
                    rk = @view beads.X[:, k]
                    @inbounds for (sign, rs_) in zip([+1, -1], [rs, rsold])
                        distance!(rk, rs_, L, vec_rks)
                        rks = norm(vec_rks)
                        pbc && (rks > L/2) && continue
                        @inbounds for d in 1:dim
                            stored.ΔF[d, k] += sign*derV(rks)*vec_rks[d]/rks
                        end
                    end
                end
            end                
        end
    elseif act == :remove
        # minus forces linked to bead that was at rsold (s is not in bs)
        if ∇V_ext !== nothing            
            @inbounds @views stored.ΔF[:, s] -= ∇V_ext(rsold)
        end
        if derV !== nothing
            vec_rks = vec_buffer
            @inbounds for k in bs
                # change on force acting on k by removing s
                rk = @view beads.X[:, k]                    
                distance!(rk, rsold, L, vec_rks)
                rks = norm(vec_rks)
                pbc && (rks > L/2) && continue
                @inbounds for d in 1:dim
                    stored.ΔF[d, k] += -derV(rks)*vec_rks[d]/rks
                end
            end                            
        end
    elseif act == :add
        # plus forces linked to bead id (s in list bs), no rsold
        # take out old outdated force, if any; an inactive bead should have zero force
        @inbounds for d in 1:dim
            stored.ΔF[d, s] = -stored.F[d, s]
        end
        if ∇V_ext !== nothing            
            stored.ΔF[:, s] += ∇V_ext(rs)
        end
        if derV !== nothing
            vec_rks = vec_buffer
            @inbounds for k in bs
                k == s && continue
                rk = @view beads.X[:, k]
                distance!(rk, rs, L, vec_rks)
                rks = norm(vec_rks)
                pbc && (rks > L/2) && continue
                # force on bead k due to bead s
                @inbounds for d in 1:dim
                    stored.ΔF[d, k] = derV(rks)*vec_rks[d]/rks # just once for each k
                    # force on bead s due to bead k  
                    stored.ΔF[d, s] += -derV(rks)*vec_rks[d]/rks # ∑ k(≠s)
                end
                    
            end
        end
    end
    return nothing
end



function get_ΔVTV(PIMC::t_pimc, beads::t_beads, rsold::SubArray{Float64}, bs::Vector{Int64}, id::Int64, act::Symbol)
    """Change in [V,[T,V]] = 2λ∑_i |F_i|^2 after one bead move; F->F_old + ΔF"""
    
    # compute stored.ΔF[:, bs]; call here to make sure ΔVTV uses new values
    compute_ΔF(PIMC, beads, rsold, bs, id, act)
    ΔVTV = 0.0   
    @inbounds for k in bs        
        # d-loop does ΔVTV += 4λ*stored.F[:, k] ⋅ stored.ΔF[:, k] + 2λ*sum(stored.ΔF[:, k].^2),
        # avoid temporaries of ⋅ and sum
        @inbounds for d in 1:dim
            ΔVTV += 4λ * stored.F[d, k] * stored.ΔF[d, k] + 2λ * stored.ΔF[d, k]^2
        end
    end
    if act == :remove
        # id was not in list bs
        #ΔVTV -= 2λ*sum(stored.F[:, id].^2)
        @inbounds for d in 1:dim
            ΔVTV -= 2λ*stored.F[d, id]^2 
        end
    end
    if act == :add        
        #ΔVTV += 2λ*sum(stored.F[:, id].^2)
        @inbounds for d in 1:dim
            ΔVTV += 2λ*stored.F[d, id]^2 
        end
    end    
    return ΔVTV
end

function compute_V!(PIMC::t_pimc,  beads::t_beads, bs::Vector{Int64}, m::Int64)
    """ Potential energy on slice, update stored.V_tmp[m]"""
    L = PIMC.L
    V_ext = PIMC.confinement_potential
    V_pair = PIMC.pair_potential
    V = 0.0
    nbs = length(bs)
    # k, i are running indices in bs
    @inbounds for k in 1:nbs
        rk = @view beads.X[:, bs[k]]
        if V_ext !== nothing
            V += V_ext(rk) 
        end
        if V_pair !== nothing
            vec_rki = vec_buffer
            @inbounds for i in 1:k-1
                ri = @view beads.X[:, bs[i]]
                distance!(rk, ri, L, vec_rki)                
                rki = norm(vec_rki)
                pbc && (rki > L/2) && continue # pbc cutoff 
                V += V_pair(rki)
            end
        end
    end
    V += PIMC_Common.Vtail*nbs # default is zero tail correction
    stored.V_tmp[m] = V
    return nothing
end

function compute_F!(PIMC::t_pimc, beads::t_beads, bs::Vector{Int64})
    """ Forces F=∇V on beads bs (active beads on the same slice), update stored.F_tmp[:, bs]"""
    L = PIMC.L
    ∇V_ext = PIMC.grad_confinement_potential   
    derV =  PIMC.der_pair_potential
    @inbounds for b in bs
        @inbounds for d in 1:dim        
            stored.F_tmp[d, b] = 0.0
        end
    end
    # k, i are bead indices 
    @inbounds for k in bs        
        rk = @view beads.X[:, k]
        if ∇V_ext !== nothing
            @inbounds for d in 1:dim
                stored.F_tmp[d, k] += ∇V_ext(rk)[d]
            end
        end
        if derV !== nothing
            vec_rki = vec_buffer
            @inbounds for i in bs
                i==k && continue
                ri = @view beads.X[:, i]
                distance!(rk, ri, L, vec_rki)                
                rki = norm(vec_rki)
                pbc && (rki > L/2) && continue
                @inbounds for d in 1:dim
                    stored.F_tmp[d, k] += derV(rki)*vec_rki[d]/rki
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
   

function get_Vx_F(PIMC::t_pimc, X::AbstractArray{T}, k::Int64) where T<:Number
    """Potential Vx and force F (used for [V,[T,V]]) for kth bead on a slice"""
    #
    # This is the *slow* version, used by coordinate scaling derivatives
    
    L = PIMC.L
    M = PIMC.M
    V_ext = PIMC.confinement_potential
    ∇V_ext = PIMC.grad_confinement_potential   
    V  = PIMC.pair_potential
    derV =  PIMC.der_pair_potential
    nbs = size(X,2) # mostly N        
    Fk = zeros(eltype(X[1,1]), dim)
    Vxk = zero(X[1,1])
    if V_ext !== nothing
        Vxk += V_ext(view(X, :, k)) 
        @inbounds @views Fk .+= ∇V_ext(view(X, :, k))
    end
    if V !== nothing
        vec_rki = zeros(eltype(X[1,1]), dim)        
        rk = @view X[:, k]            
        i_not_k = [i for i in 1:nbs if i!=k]
        @inbounds for i in i_not_k
            ri = @view X[:, i]
            distance!(rk, ri, L, vec_rki)                
            rki = norm(vec_rki)
            pbc && (rki > L/2) && continue # pbc cutoff
            @inbounds for d in 1:dim
                Fk[d] += derV(rki)*vec_rki[d]/rki
            end
            if i<k
                Vxk += V(rki) #  ∑_{i(<k)} V(r_ki)
            end
        end
    end
    Vxk, Fk
end

function Vx_and_VTVx(PIMC::t_pimc, X::AbstractArray{T}) where T<:Number
    """Potential V(x) and commutator [V[T,V]](x) = 2λ|F(x)|^2 = 2λ|∂_x V(x)|^2 for one slice"""    
    if PIMC.confinement_potential == nothing && PIMC.pair_potential == nothing
        return 0.0, 0.0
    end
    
    nbs = size(X,2) # mostly N
    Vx = zero(X[1,1])
    VTVx = zero(X[1,1])
    @inbounds for k in 1:nbs 
        V, F = get_Vx_F(PIMC, X, k) # F is force on kth bead
        VTVx += 2λ*sum(F.^2)
        Vx += V 
    end
    # add tail correction (default Vtail is zero)
    Vx += PIMC_Common.Vtail * nbs
    Vx, VTVx
end



function Evir_chin(PIMC::t_pimc, beads::t_beads, links::t_links)
    """Scaled potential derivative ∂β'[Vtilde_m(x_m^s; τ')]|β'=β, used in virial energy estimator """
    # 
    #
    if PIMC.confinement_potential == nothing && PIMC.pair_potential == nothing
        return 0.0
    end
    
    M = PIMC.M
    β = PIMC.β
    L = PIMC.L
    V_ext = PIMC.confinement_potential
    ∇V_ext = PIMC.grad_confinement_potential   
    V = PIMC.pair_potential
    derV =  PIMC.der_pair_potential
    
    r_ref = zeros(dim)    
    # co-moving centroid references
    function centroid_ref!(r_ref::AbstractArray{Float64, 2})
        """ co-moving centroid reference """
        dr = zeros(dim)
        active = findall(beads.active)
        @inbounds for id in active
            r_ref[:, id] .= 2*beads.X[:, id]
            # steps up and down from bead id
            id_up = id 
            id_do = id 
            # bead X[:, id] defines the continuous path
            rup = copy(beads.X[:, id]) # path up
            rdo = copy(beads.X[:, id]) # path down
            @inbounds for _ in 1:M-1
                id_up = links.next[id_up]
                id_do = links.prev[id_do]
                distance!(view(beads.X, :, id_up), view(rup,:), L, dr)
                rup +=  dr
                @. r_ref[:, id] += rup
                distance!(view(beads.X, :, id_do), view(rdo,:), L, dr)
                rdo += dr
                @. r_ref[:, id] += rdo
            end
        end
        r_ref .*= 1/(2*M)
    end

    
    r_ref = @view mat_buffer[:, 1:count(beads.active)] 
    centroid_ref!(r_ref)
    
    function Vtilde(βp)
        """ Vtilde_m(x_m^s; τ') """  
        # βp marks β'
        # here β is external constant
        res = 0.0
        τp = 3/M*βp # to be derived wrt. βp 
        s =  sqrt(βp/β) - 1 # to be derived wrt. β        
        dr = vec_buffer
        # sub-optimal, but the dual type has dynamic tags to pre-allocate outside function
        _Xs_buffer = Matrix{eltype(s)}(undef, dim, N) 
        @inbounds for m = 1:M
            bs = active_beads_on_slice(beads, m)
            nbs = length(bs)
            Xs = @view _Xs_buffer[:, 1:nbs]  #Matrix{eltype(s)}(undef, dim, nbs)  # scaled coordinates on slice m           
            @inbounds for i = 1:nbs
                id = bs[i]
                r_m = @view beads.X[:, id]
                # steps up and down from bead id
                distance!(r_m,  view(r_ref, :, id), L, dr)                
                r_m_s = r_m + s*dr
                Xs[:, i] .=  r_m_s
            end
            # V and VTV at scaled coordinates (slow version, can't use update and stored values)
            Vxs, VTVxs = Vx_and_VTVx(PIMC, Xs)
            res += V_m(m, Vxs, VTVxs, τp)
        end
        res  # Vtilde unit is energy
    end

    # deriv unit is energy
    deriv = β*ForwardDiff.derivative(Vtilde, β)  # derivative evaluated at β
    return deriv*3/M  
    
end



# Modified potential
# ==================
function V_m(m::Int64, Vx::T, VTVx::T, τ::T) where T<:Number
    """Chin Action modified potential with V and [V,[T,V]] on slice m"""
    if (m+1)%3 == 0        
        V_m = chin_v2*Vx + τ^2*chin_u0*(1-2*chin_a1) * VTVx # v2*W_(1-2a1)
    else
        V_m = chin_v1*Vx + τ^2*chin_u0*chin_a1 * VTVx   # v1*W_a1
    end
end

# Kinetic action
# ==============
function K(PIMC::t_pimc, beads::t_beads, links::t_links)
    """Kinetic action of the whole PIMC config"""
    β = PIMC.β
    L = PIMC.L
    K::Float64 = 0.0 

    active = findall(beads.active)
    @inbounds for id in active
        id_next = links.next[id]
        Δr2 = dist2(view(beads.X, :, id), view(beads.X, :, id_next), L)        
        Δτ = mod(beads.times[id_next] - beads.times[id], β)         
        K += dim/2*log(4π*λ*Δτ) + Δr2/(4*λ*Δτ) 
    end
    K
end

function K(PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    """Kinetic action of bead id to next and to previous"""    
    β = PIMC.β
    L = PIMC.L
    
    # link to next bead
    id_next = links.next[id]      
    Δr2  = dist2(view(beads.X, :, id), view(beads.X, :, id_next), L)
    Δτ = mod(beads.times[id_next] - beads.times[id], β)
    
    # exp(-K) is (4π*λ*Δτ)^{-dim/2}*exp(-Δr^2/(4*λ*Δτ))
    K = dim/2*log(4π*λ*Δτ) +  Δr2/(4*λ*Δτ)

    # link to previous bead
    id_prev = links.prev[id]
    Δr2  = dist2(view(beads.X, :, id), view(beads.X, :, id_prev), L)
    Δτ = mod(beads.times[id] - beads.times[id_prev], β)    
    K  += dim/2*log(4π*λ*Δτ) + Δr2/(4*λ*Δτ)
end



# Inter-action
# ============

counter = 0 
function update_stored_slice_data(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Move accepted: Update stored values of V, VTV and F from their tmp values"""
    global counter
    counter += 1
    if counter>100000
        # trigger full update in next call to U_stored to avoid error accumulation in updates
        init_stored(PIMC, beads)
        counter = 0
    end
    m = beads.ts[id]
    bs = active_beads_on_slice(beads, m)
    
    stored.V[m] = stored.V_tmp[m]   
    stored.VTV[m] = stored.VTV_tmp[m]
    @inbounds for d in 1:dim
        stored.F[d, bs] .= stored.F_tmp[d, bs]
    end

    if false
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
    
end


function init_stored(PIMC::t_pimc, beads::t_beads)
    """Computing and storing V, VTV and F for each slice from scratch"""
    println("Chin Action init_stored:: Initializing action-specific stored values")
    @inbounds for m in 1:PIMC.M
        bs = active_beads_on_slice(beads, m)
        compute_V!(PIMC, beads, bs, m) # sets stored.V_tmp[m]
        compute_F!(PIMC, beads, bs)    # sets stored.F_tmp[:, bs]
        stored.V[m] = stored.V_tmp[m]
        stored.F[:, bs] .= stored.F_tmp[:, bs]
        stored.VTV[m] = 2λ*sum(stored.F[:, bs].^2)
    end
    PIMC_Chin_Action.stored.set = true
end


function U_stored(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Inter-action before changes using stored V and VTV"""    
    if !PIMC_Chin_Action.stored.set
        error("Can't use U_stored before stored values are initialized in init_stored")
    end
    m = beads.ts[id]
    τ = PIMC.τ
    # U could be stored, but it's recomputation from store V and VTV is fast
    U = τ * V_m(m, stored.V[m], stored.VTV[m], τ)
    U
end

function U_update(PIMC::t_pimc, beads::t_beads, xold::SubArray{Float64}, id::Int64, act::Symbol)
    """Inter-action after *suggested* change xold -> beads.X[:, id]"""   
    if !PIMC_Chin_Action.stored.set
        error("Can't update before stored values are initialized in init_stored")
    end
    m = beads.ts[id]
    τ = PIMC.τ    
    bs = active_beads_on_slice(beads, m)

    
    compute_ΔV(PIMC, beads, xold, bs, id, act)
    ΔV = stored.ΔV[m] 
    ΔVTV = get_ΔVTV(PIMC, beads, xold, bs, id, act) # computes stored.ΔF[:, bs]
    ΔF = stored.ΔF
    
    Vx = stored.V[m] + ΔV  
    VTVx = stored.VTV[m] + ΔVTV
    
    Fx = copy(stored.F)
    @inbounds for d in 1:dim
        Fx[d, bs] += ΔF[d, bs]
    end

    Unew = τ * V_m(m, Vx, VTVx, τ)

    if false
        # check SLOW
        println("-"^80)
        @show act
        @show id, bs        
        compute_V!(PIMC, beads, bs, m) # new stored.V_tmp[m]   
        if !isapprox(stored.V_tmp[m], Vx, atol=1e-8)
            @show stored.V[m], ΔV
            @show stored.V_tmp[m], Vx, stored.V_tmp[m] - Vx
            error("ΔV is wrong")
        else        
            println("ΔV is ok")
        end

        F_tmp_orig = copy(stored.F_tmp[:, bs])
        compute_F!(PIMC, beads, bs) # new stored.F_tmp   
        if !isapprox(stored.F_tmp[:, bs], Fx[:, bs], atol=1e-8)
            for k in bs
                #@show k, stored.F_tmp[:, k], Fx[:, k]
                if !isapprox(stored.F_tmp[:, k], Fx[:, k], atol=1e-8)
                    @show stored.F_tmp[:, k] , Fx[:, k]
                    @show k, stored.F_tmp[:, k] .- Fx[:, k]
                    @show stored.F[:, k]
                end
            end
            error("ΔF is wrong")
        else
            println("ΔF is ok")
        end
        
        
        if !isapprox(2λ*sum(stored.F_tmp[:, bs].^2), VTVx, atol=1e-5)
            @show stored.VTV[m],  ΔVTV
            Fbs = @view stored.F_tmp[:, bs]
            @show 2λ*sum(Fbs.^2), VTVx, 2λ*sum(Fbs.^2)-VTVx        
            error("ΔVTV is wrong")
        else
            println("ΔVTV is ok")
        end
        stored.F_tmp[:, bs] .= F_tmp_orig
        #
    end
    
    # update tmp storage
    stored.V_tmp[m] = Vx    
    stored.VTV_tmp[m] = VTVx
    stored.F_tmp[:, bs] .= Fx[:, bs]

    if false
        # check SLOW
        Vx_, VTVx_ = Vx_and_VTVx(PIMC, view(beads.X, :, bs))
        if !isapprox(Vx_, Vx, atol=1e-8)
            @show Vx_, Vx
            error("wrong V")
        end
        if !isapprox( VTVx_, VTVx, atol=1e-8)
            @show VTVx_, VTVx
            error("wrong VTV")
        end
        
        
        UU =  U(PIMC, beads, id)
        if !isapprox(UU, Unew, atol=1e-8)        
            @show Unew, UU, Unew - UU
            error("U is wrong")
        end
    end
    
    Unew
end

function U(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Inter-action τ*U of bead id"""
    # SLOW use only for checking
    #error("not meant to use this")
    m = beads.ts[id]
    τ = PIMC.τ
    m = beads.ts[id]
    bs = active_beads_on_slice(beads, m)
    Vx, VTVx = Vx_and_VTVx(PIMC, view(beads.X,:,bs))
    U = τ * V_m(m, Vx, VTVx, τ)
    U
end

    

function U(PIMC::t_pimc, beads::t_beads, m::Int64, τ)
    """Inter-action τ*U of beads at slice m, variable τ to be used in AD"""
    # SLOW use only for checking
    #error("not meant to use this")
    bs = active_beads_on_slice(beads, m)
    Vx, VTVx = Vx_and_VTVx(PIMC, view(beads.X,:,bs))
    U = τ * V_m(m, Vx, VTVx, τ)
    U
end
        
function U(PIMC::t_pimc, beads::t_beads)
    """Inter-action τ*U of the whole PIMC"""
    # SLOW use only for checking
    #error("not meant to use this")
    τ = PIMC.τ    
    U::Float64 = 0.0 
    @inbounds for m in 1:PIMC.M
        bs = active_beads_on_slice(beads, m)
        Vx, VTVx = Vx_and_VTVx(PIMC, view(beads.X,:,bs))   
        U += V_m(m, Vx, VTVx, τ)        
    end
    τ*U
end



end
