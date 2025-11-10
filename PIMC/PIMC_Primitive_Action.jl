__precompile__(false)
module PIMC_Primitive_Action
using Printf
using LinearAlgebra: ⋅, dot
using ForwardDiff
using StaticArrays


push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Structs
using QMC_Statistics
using PIMC_Systems
using PIMC_Utilities: active_beads_on_slice, boson_virial_exchange_energy, E_pot_bead, E_pot_all

export U, K, U_stored, U_update, update_stored_slice_data
export meas_E_th, meas_E_vir, Ekin_th, init_action!, meas_virial_pressure

using PIMC_Reports: report_energy



# light-weight vector buffers, not thread-safe!
const vec_buffer =  MVector{dim, Float64}(undef) 
const vec_buffer2 = MVector{dim, Float64}(undef)
const vec_buffer3 = MVector{dim, Float64}(undef) 
const rref_buffer = Matrix{Float64}(undef, dim, N_slice_max*1000) # overkill

# Storage for V(x) on each slice
mutable struct t_stored
    set::Bool
    V::Vector{Float64}
    ΔV::Vector{Float64}
    V_tmp::Vector{Float64}
    function t_stored(M::Int)
        new(false, zeros(M), zeros(M), zeros(M))
    end
end
stored = t_stored(1) 

function init_action!(PIMC::t_pimc, beads::t_beads)
    global stored
    """Init Primitive Action"""
    println("Init Primitive Action")
    # evenly spaced
    M = PIMC.M
    β = PIMC.β
    τ = β/M    
    PIMC.τ = τ
    
    times = [(m-1)*τ for m in 1:M]

    # set bead times
    for id in beads.ids
        m = beads.ts[id]
        beads.times[id] = times[m]
    end
    stored = t_stored(M)
end


# Thermodynamic energy estimator
# ==============================

@inline norm(x) = sqrt(sum(abs2,x))



mutable struct t_epot
    ipimc::Int64
    Epot::Float64
end
Epot_store = t_epot(0, 0.0)

function Ekin_th(PIMC::t_pimc, beads::t_beads, links::t_links)
    τ = PIMC.τ
    L = PIMC.L
    M = PIMC.M
    Ekin = dim*N/(2*τ)    
    Δr2 = 0.0
    @inbounds @simd for id in beads.ids[beads.active]
        id_next = links.next[id]
        Δr2  += dist2(view(beads.X, :, id), view(beads.X, :, id_next), L)
    end
    Ekin += -Δr2 / (4*λ*τ^2*M)
    Ekin
end


function meas_E_th(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    """Thermodynamic energy estimator"""

   
    Ekin = Ekin_th(PIMC, beads, links)
    
    Epot = 0.0
    # use stored value if already computed
    #if Epot_store.ipimc == PIMC.ipimc
    #    Epot = Epot_store.Epot
    #else
        Epot = E_pot_all(PIMC, beads)
    #    Epot_store.ipimc = PIMC.ipimc
    #    Epot_store.Epot = Epot        
    #end
    
    Ekin /= N
    Epot /= N
    E = Ekin + Epot
   
    #if PIMC.ipimc < 0
        # save raw energy for correlation analysis or other checks
    #    open("E.raw", "a") do f
    #        println(f, E)
    #    end
    #end
    
    

    # Collect statistics
    add_sample!(meas.stat, [E, Ekin, Epot])
    if meas.stat.finished
        # block full, report
        report_energy(PIMC, meas)        
    end
end


function meas_E_vir(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement; opt::Bool=false)
    """Virial energy estimator"""

    M = PIMC.M    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
   
    Ekin::Float64 = dim*N/(2*β)  # exact distinguishable particle kinetic energy
    Eexc::Float64 = 0.0 
    if bose
        # exchange energy term        
        Eexc = boson_virial_exchange_energy(PIMC, beads, links)
        Ekin += Eexc
    end
    Evir = Evir_prim(PIMC, beads, links)
    Ekin += Evir

    Epot = 0.0
    # use stored value if already computed
    #if Epot_store.ipimc == PIMC.ipimc
    #    Epot = Epot_store.Epot
    #else
        Epot = E_pot_all(PIMC, beads)
    #    Epot_store.ipimc = PIMC.ipimc
    #    Epot_store.Epot = Epot        
    #end
    
    Eexc /= N
    Ekin /= N
    Epot /= N
    E = Ekin + Epot
    
    if opt
        return E
    end
    #open("E_raw","a") do f
    #    @printf(f, "%15.5f %15.5f %15.5f  %15.5f\n", E, Ekin, Epot, Eexc)
    #end
    
    # Collect statistics
    add_sample!(meas.stat, [E, Ekin, Epot])
    if meas.stat.finished
        # block full, report
        report_energy(PIMC, meas)        
    end
   
end


# Co-moving centroid reference for all active beads
function centroid_ref!(PIMC::t_pimc, beads::t_beads, links::t_links, rref::Matrix{Float64},
                       dr::MVector{dim, Float64} = vec_buffer,
                       rup::MVector{dim, Float64} = vec_buffer2,
                       rdo::MVector{dim, Float64} = vec_buffer3    
                       ) ::Nothing
    """ co-moving centroid reference using M beads up and down"""
    M = PIMC.M
    L = PIMC.L
    active = beads.ids[beads.active]
    
    @inbounds @views begin
        for b in active
            rref[:, b] .= 2*beads.X[:, b]
            # steps up and down from bead b
            bup = b 
            bdo = b 
            # bead X[:, id] defines the continuous path
            rup .= beads.X[:, b] # path up
            rdo .= beads.X[:, b] # path down
            for _ in 1:M-1
                bup = links.next[bup]
                bdo = links.prev[bdo]
                distance!(view(beads.X, :, bup), rup, L, dr)
                rup .+=  dr
                rref[:, b] .+= rup
                distance!(view(beads.X, :, bdo), rdo, L, dr)
                rdo .+= dr
                rref[:, b] .+= rdo
            end
        end
        rref .*= 1/(2*M)
    end # inbounds
    return nothing
end

function Evir_prim(PIMC::t_pimc, beads::t_beads, links::t_links,
                   rref::Matrix{Float64} = rref_buffer)
    """Scaled potential derivative ∂β'[Vtilde_m(x_m^s; τ')]|β'=β, used in virial energy estimator """
    # 
    # (prim) =  ∂β'[V(x_m^s)]|β'=β
    # x_m^s := (r_1^s, r_2^s, ... ) = (r_m^s), m=1:M*N
    # r_m^s := r_m + s*(r_m - r_m^*) , s = sqrt(β'/β)-1, ∂β'(s) = 1/(2β)
    # (∂β'[r_m^s]|β'=β = (r_m - r_m^*)/(2β))
    # 
    #
    if !has_potential(PIMC.confinement_potential)  && !has_potential(PIMC.pair_potential)
        return 0.0
    end
    
    M = PIMC.M
    V_ext = PIMC.confinement_potential
    ∇V_ext = PIMC.grad_confinement_potential   
    V = PIMC.pair_potential
    derV =  PIMC.der_pair_potential
    β = PIMC.β
    L = PIMC.L
    
    deriv = 0.0
    active = beads.ids[beads.active]
    
    # co-moving centroid references
    function centroid_ref!(rref::Matrix{Float64}) ::Nothing
        """ co-moving centroid reference """
        dr = zeros(dim)
        
        for id in active
            rref[:, id] .= 2*beads.X[:, id]
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
                @. rref[:, id] += rup
                distance!(view(beads.X, :, id_do), view(rdo,:), L, dr)
                rdo += dr
                @. rref[:, id] += rdo
            end
        end
        rref .*= 1/(2*M)
        return nothing
    end

    centroid_ref!(rref)
      
    
    function Vtilde(βp)
        # here β is external constant
        res = 0.0
        s =  sqrt(βp/β) - 1
        Xs = []  # scaled coordinates on all slices 
        id_to_Xs_index = Dict() # bead id to Xs index dictionary
        dr = zeros(dim)
        @inbounds for (i, id) in enumerate(beads.active)
            r_m = @view beads.X[:, id]
            # steps up and down from bead id
            distance!(r_m, view(rref, :, id), L, dr)
            r_m_s = r_m + s*dr
            push!(Xs, r_m_s)
            id_to_Xs_index[id] = i
        end
        # one-body potential
        if has_potential(PIMC.confinement_potential)
            res += sum(V_ext.(Xs))
        end
        # pair potential
        if has_potential(PIMC.pair_potential)
            @inbounds for t in 1:M
                # beads on slice t
                bs = active_beads_on_slice(t)
                nbs = length(bs) # N 
                # pairwise interaction on slice t, sum over j>i 
                @inbounds for i in 1:nbs-1
                    ix = id_to_Xs_index[bs[i]] # scaled coordinate index
                    @inbounds for j in i+1:nbs  
                        jx =  id_to_Xs_index[bs[j]]  # another scaled coordinate index
                        rij = dist(Xs[ix], Xs[jx], L) # scaled coordinate distance
                        pbc && rij > L/2 && continue # apply cutoff in pbc
                        res += V(rij)
                    end
                end
            end
        end
        res 
    end

    # using gradients instead of scaling and β-derivatives
    if has_potential(PIMC.confinement_potential)        
        dr = zeros(dim)
        @inbounds for id in active
            vecr = @view beads.X[:, id]
            ref  = @view rref[:, id]
            distance!(vecr, ref, L, dr)
            deriv += 0.5 * dr ⋅ ∇V_ext(vecr)
        end       
    end

    if has_potential(PIMC.pair_potential)
        
        #
        # sum_{m=1}^M  1/2*(xm-xm^*) ⋅ ∇_xm V(xm; τ)
        # = sum_{m=1}^M \sum_{k=1}^N  1/2*(r{m,k}-r_{m,k}^*) ⋅ ∇_r{m,k} V(xm; τ)
        # 
        #   ∇_rk V(xm; \tau) 
        # = ∇_rk sum_{i<j} V(r_ij) = sum_{i (\ne k)} V'(rki)*(rk-ri)/r_ki
        #                           :=  sum_{i (\ne k)} derV(rki)*(rk-ri)/rki
        
        ∇V = zeros(dim)
        vec_rki = zeros(dim)
        dr = zeros(dim)
        @inbounds for m in 1:M
            # beads on slice m
            bs = active_beads_on_slice(beads, m)
            nbs = length(bs) # mostly N 
            @inbounds for k in 1:nbs  # index in bs
                kk = bs[k] # bead id
                rk = @view beads.X[:, kk]
                ∇V .= 0
                i_not_k = [i for i in 1:nbs if i!=k]
                @inbounds for i in i_not_k
                    ii = bs[i] # bead id
                    ri = @view beads.X[:, ii]
                    distance!(rk, ri, L, vec_rki)                
                    rki = norm(vec_rki)
                    pbc && (rki > L/2) && continue # apply cutoff in pbc                
                    ∇V += derV(rki)*vec_rki/rki
                end
                deriv += 0.5 * (rk - rref[:, kk]) ⋅ ∇V
            end
        end
    end

    # testing that the computed values are the same (they are)
    #AD_deriv = β*ForwardDiff.derivative(Vtilde, β)  # derivative evaluated at β
    #@show AD_deriv/M, deriv/M 
    
    return deriv/M
    
    
    
end

const F_buffer = Matrix{Float64}(undef, dim, N_slice_max)



function meas_virial_pressure(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement,
                              vec_rki::MVector{dim, Float64} = vec_buffer,
                              F::Matrix{Float64} = F_buffer; opt::Bool=false
                              ) ::Float64
    M = PIMC.M
    L = PIMC.L
    β = PIMC.β
   
    ∇V_ext = PIMC.grad_confinement_potential   
    V´ = PIMC.der_pair_potential
    
    vir_pot::Float64 = 0.0
    vir_exc::Float64 = 0.0
    #
    # Potential term
    #
    @inbounds begin
        for m in 1:M
            bs = active_beads_on_slice(beads, m)
            nbs = length(bs)
            F .= 0.0
            for k in 1:nbs
                rk = @view beads.X[:, bs[k]]
                if has_potential(∇V_ext)
                    for d in 1:dim
                        F[d, k] += ∇V_ext(rk)[d]
                    end
                end
                if has_potential(V´)
                    for i in 1:nbs                        
                        i==k && continue
                        ri = @view beads.X[:, bs[i]]
                        distance!(rk, ri, L, vec_rki)                
                        rki = norm(vec_rki)
                        if pbc
                            if rki > L/2
                                continue
                            end
                        end
                        dVki::Float64 = V´(rki)/rki
                        F[:, k] .+= dVki*vec_rki
                    end
                end
            end
            for i in 1:nbs
                vir_pot += dot(view(beads.X, :, bs[i]), view(F, :, i))
            end            
        end
    end # inbounds
    #
    # Exchange term
    #
    if bose       
        # Exc may be stored
        if Eexc_store.ipimc == PIMC.ipimc && Eexc_store.iworm == PIMC.iworm
            Eexc =  Eexc_store.Eexc
        else
            Eexc = boson_virial_exchange_energy(PIMC, beads, links)
            Eexc_store.ipimc = PIMC.ipimc
            Eexc_store.iworm = PIMC.iworm
            Eexc_store.Eexc = Eexc
        end
        vir_exc  = -Eexc*β        
    end 




    
    Ω = L^dim # volume, cubic box
    
    # P = N*k*T/Ω  | ideal gas pressure
    #    + 2/(dim*Ω) < 1/M ∑_m=1^M  ∑_m=1^M (x_{M+m}-x_m)⋅(x_{M+m-1}-x_{M+m})/(4π t_{M+m-1}τ) | exchange 
    #                  + τ/2 ∑_m=1^M x_m^* ⋅ ∂ V/∂x_m >
    #  :=  N*k*T/Ω  + 2/(dim*Ω) * vir 
    #       
    P = N/(β*Ω) + 2/(dim*Ω) * (vir_exc + vir_pot)
    if opt
        return P
    end
    
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

  
function compute_ΔV(PIMC::t_pimc, beads::t_beads, rsold::AbstractArray{Float64}, bs::Vector{Int64}, id::Int64, act::Symbol)
    """Computes change in potential energy ΔV to stored.ΔV[m]"""
     
    L = PIMC.L
    V_ext = PIMC.confinement_potential
    Vpair = PIMC.pair_potential
    
    ΔV = 0.0
    if act == :move
        # move from rsold -> rs := beads.X[:, id] 
        rs = view(beads.X, :, id)
        if has_potential(V_ext)
            ΔV += V_ext(rs) - V_ext(rsold) 
        end
        if has_potential(Vpair)
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
        if has_potential(V_ext)
            ΔV +=  - V_ext(rsold) 
        end
        if has_potential(Vpair)
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
        if has_potential(V_ext)
            ΔV += V_ext(rs) 
        end
        if has_potential(Vpair)
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


function compute_V!(PIMC::t_pimc,  beads::t_beads, bs::Vector{Int64}, m::Int64)
    """ Potential energy on slice, update stored.V_tmp[m]"""
    L = PIMC.L
    V_ext = PIMC.confinement_potential
    V_pair = PIMC.pair_potential
    V = 0.0
    nbs = length(bs)
    # k, i are running indices in bs
    @inbounds for k in 1:nbs
        rk = view(beads.X, :, bs[k])
        if has_potential(V_ext)
            V += V_ext(rk) 
        end
        if has_potential(V_pair)
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


# Kinetic action
# ==============
function K(PIMC::t_pimc, beads::t_beads, links::t_links)
    """Kinetic action of the whole PIMC config"""
    M = PIMC.M    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
   

    Δr2 = 0.0
    @inbounds for id in beads.ids[beads.active]
        id_next = links.next[id]
        Δr2  += dist2(view(beads.X, :, id), view(beads.X, :, id_next), L)        
    end
    K = dim/2*log(4π*λ*τ) + Δr2/(4*λ*τ) 
end

function K(PIMC::t_pimc, beads::t_beads, links::t_links, id::Int64)
    """Kinetic action of bead id"""
    
    β = PIMC.β
    τ = PIMC.τ
    L = PIMC.L
   
    
    # exp(-K) is (4π*λ*Δτ)^{-dim/2}*exp(-Δr^2/(4*λ*Δτ))
    # => -ln(K) = dim/2*(4π*λ*Δτ) + Δr^2/(4*λ*Δτ)    
    # this was per link, here
    # K = dim*(4π*λ*Δτ) + (Δr_id_id_nextt^2 + Δr_id_prev_id^2)/(4*λ*Δτ) 
    id_next = links.next[id]
    id_prev = links.prev[id]
    Δr2  = dist2(view(beads.X, :, id), view(beads.X, :, id_next), L)
    Δr2  += dist2(view(beads.X, :, id), view(beads.X, :, id_prev), L)
    K = dim*log(4π*λ*τ) +  Δr2/(4*λ*τ)
end

# Inter-action
# ============
counter = 0
function update_stored_slice_data(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Move accepted: Update stored values of V, VTV and F from their tmp values"""
    global counter
    counter += 1
    if counter>10000
        # trigger full update in next call to U_stored to avoid error accumulation in updates
        init_stored(PIMC, beads)
        counter = 0
    end
    m = beads.ts[id]
    bs = active_beads_on_slice(beads, m)    
    stored.V[m] = stored.V_tmp[m]
end

function init_stored(PIMC::t_pimc, beads::t_beads)
    """Computing and storing V for each slice from scratch"""
    @inbounds for m in 1:PIMC.M
        bs = active_beads_on_slice(beads, m)
        compute_V!(PIMC, beads, bs, m) # sets stored.V_tmp[m]
        stored.V[m] = stored.V_tmp[m]
    end
    stored.set = true
end


function U_stored(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Inter-action before changes using stored V"""
    # U could be stored, but it's recomputation from stored V is fast
    
    m = beads.ts[id]
    if !stored.set
        init_stored(PIMC, beads)        
    end
    U = PIMC.τ * stored.V[m]
end



function U_update(PIMC::t_pimc, beads::t_beads, xold::AbstractArray{Float64}, id::Int64, act::Symbol, fake::Bool=false) ::Float64
    """Inter-action after *suggested* change xold -> beads.X[:, id]"""
    m = beads.ts[id]
    if !stored.set
        # trigger error if U_update is called before U_stored
        error("Can't update before stored values are initialized in init_stored called from U_stored")
    end
    bs = active_beads_on_slice(beads, m)    
    compute_ΔV(PIMC, beads, xold, bs, id, act)
    ΔV = stored.ΔV[m]     
    Vx = stored.V[m] + ΔV
    Unew = PIMC.τ * Vx
end

function U(PIMC::t_pimc, beads::t_beads)
    """Inter-action τ*U of the whole PIMC config"""    
    U = PIMC.τ * E_pot_all(PIMC, beads)
end

function U(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Inter-action τ*U of bead id"""
    U = PIMC.τ * E_pot_bead(PIMC, beads, id)
end


end
