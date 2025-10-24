__precompile__(false)
module PIMC_Measurements

using Random, LinearAlgebra
using Printf


# local modules:
push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Utilities
using PIMC_Structs: t_pimc, t_beads, t_links, t_measurement
using QMC_Statistics
using PIMC_Systems
using PIMC_Action_Interface

export add_measurement!
export rho_s
export meas_density_profile, meas_superfluid_fraction, meas_radial_distribution
export meas_cycle_count, meas_obdm, meas_head_tail_histogram
export E_pot_bead, E_pot_all
export boson_virial_exchange_energy
export meas_static_structure_factor

function add_measurement!(PIMC::t_pimc, frequency::Int64, name::Symbol,
                          exe::Function, datasize::Int64, blocksize::Int64,
                          filename::String)
    stat = init_stat(datasize, blocksize)
    measurement = t_measurement(frequency, name, exe, stat, filename)
    push!(PIMC.measurements, measurement)    
    println("Added measurement: ",name," frequency ",frequency)
end

# lightweight buffers, not thread safe!
const vec_buffer = Vector{Float64}(undef, dim)
const vec_buffer2 = Vector{Float64}(undef, dim)

# "Symmetric" slices where measurements are done
# needed for Chin Action, which has 3 slices per one τ 
# Compile-time dispatch
@inline function get_ms(::PrimitiveAction, M::Int64)
    return 1:M
end
@inline function get_ms(::ChinAction, M::Int64)
    return 2:3:M
end
@inline function get_ms(M::Int64)
    A = PIMC_Common.action
    return get_ms(A(), M)
end



# Measurement functions
# =====================


function meas_density_profile(PIMC::t_pimc, beads::t_beads, dummy::t_links, meas::t_measurement)
    """Computes density profile"""
    datasize = length(meas.stat.sample.data)
    dens_profile = zeros(datasize) 
    dens_grid = range(-10.0, 10.0, length=datasize) # hard-wired limits
    M = PIMC.M
    n = 0
    @inbounds for id in beads.ids[beads.active]
        n += 1
        r = beads.X[1, id] # x profile
        bin = searchsortedfirst(dens_grid, r)
        # bin = findfirst(b -> b > r, dens_grid)  # slower
        if bin == 1 && r < dens_grid[1]  # avoid overpopulating bin 1
            bin = nothing
        end        
        if bin !== nothing && 1<=bin<=datasize # searchsortedfirst may return bin>datasize
            dens_profile[bin] += 1
        end
    end
    bin_width = (dens_grid[end] - dens_grid[1]) / length(dens_grid)
    dens_profile = dens_profile ./ (n*bin_width)
    
    add_sample!(meas.stat, dens_profile)
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks  = get_stats(meas.stat)
        open(meas.filename, "w") do f
            @inbounds for i in 1:datasize
                println(f, dens_grid[i]," ",ave[i]," " ,std[i])
            end
        end
        
        # set HDF5 data Dict
        xs = [x for x in dens_grid]
        results[string(meas.name)] = Dict(
            "x" => xs, 
            "rho" => ave,
            "std" => std
        )
    end   
end


mutable struct t_head_tail_histo    
    nbins::Int64
    ndata::Int64
    histo::Vector{Float64}
end
const head_tail = t_head_tail_histo(1000, 0, zeros(1000)) # buffer, should be big enough     

function meas_head_tail_histogram(PIMC::t_pimc, beads::t_beads, dummy::t_links, meas::t_measurement)
    
    datasize = length(meas.stat.sample.data)
    M = PIMC.M    
    t_H = beads.ts[PIMC.head]
    t_T = beads.ts[PIMC.tail]
    bin = mod1(t_T-t_H, M)
    
    if bin<=datasize
        head_tail.ndata += 1
        head_tail.histo[bin] += 1
    end

    # collect some histogram data per sample
    if head_tail.ndata == 100
        @inbounds @views head_tail.histo[1:datasize] ./= 100
        add_sample!(meas.stat, head_tail.histo[1:datasize])
        head_tail.ndata = 0
        @inbounds @views head_tail.histo[1:datasize] .= 0
    end
    
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks  = get_stats(meas.stat)
        # file output
        if meas.filename != ""
            open(meas.filename, "w") do f
                @inbounds for i in 1:datasize
                    @printf(f, "%15d %15.5f %15.5f\n", i, ave[i], std[i])
                end
            end
        end
        is = [i for i in 1:datasize]
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "i" => is,
            "histo" => ave,
            "std" => std
        )
    end    
end


#  
# One-body density matrix (OBDM) ρ_1(r) for homogeneous isotropic fluids
#
const obdm_buffer = Vector{Float64}(undef, 1000) # fixed size buffer

function meas_obdm(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement,
                   obdm_buffer::Vector{Float64}=obdm_buffer)
    """One-body density matrix: histogram of same-time Head-Tail distances"""
    #  calls directly PIMC_Action_Interface.U_stored and PIMC_Action_Interface.U_stored to avoid dependency loop
    
    if PIMC.head == -1
        return # no worm
    end
    M::Int64 = PIMC.M
    ms = get_ms(M)
    
    if beads.ts[PIMC.tail] ∉ ms
        return # tail is not on a symmetric slice (any slice in Primitive, some Chin Action slices)
    end
    
    L::Float64 = PIMC.L    
    β::Float64 = PIMC.β
    μ::Float64 = PIMC.μ
    
    datasize = length(meas.stat.sample.data)
    if datasize>length(obdm_buffer)
        error("buffer too small")
    end
    obdm = @view obdm_buffer[1:datasize]
    @inbounds @views obdm .= 0.0
    
    stepr = 0.02 # default guess
    if pbc
        stepr = L/2/datasize 
    end
    # fake close, head on the same slice as tail, but at distance r from tail
    rT = copy(beads.X[:, PIMC.tail])
    rH = @view beads.X[:, PIMC.head]
    
    
    # need beads from head to fake head (which is at the tail slice m)
    k = mod1(beads.ts[PIMC.tail] - beads.ts[PIMC.head], M)
    idlist = Vector{Int64}(undef, k+1)
    new_bead_indices = 2:k # generated beads, from head+1 to fake head -1
    
    m = beads.ts[PIMC.head]
    idlist[1] = PIMC.head   
    @inbounds for i in new_bead_indices
        m = mod1(m+1, M)
        ind = findfirst(b -> !beads.active[b], beads.at_t[m]) # find inactive bead on slice m
        if ind===nothing
            error("No free beads on slice")
        end
        id = beads.at_t[m][ind] 
        idlist[i] = id        
    end
    idlist[end] = PIMC.tail
    rfakeH = @view beads.X[:, PIMC.tail]
    Δτ = mod(beads.times[PIMC.tail] - beads.times[PIMC.head], β) # head -> fake head   
    nreps = 5 
    # pick fake head on the same slice as tail and generate free-particle path head->fake head
    # temporarily move particle from tail to fake head to measure <Ψ^†(r')Ψ(r)>
    #
    @views beads.active[idlist[new_bead_indices]] .= true 
    #@inbounds begin
    
    
        Uold = 0.0    
        for id in idlist[2:end]
            Uold += PIMC_Action_Interface.U_stored(PIMC, beads, id) # picks slice index based on bead id
        end
        # possibly unnecessary, but activate new beads to be sure 
        
    
        for rep in 1:nreps # a few directions
            # dir is a random direction (assumes isotropic homogeneous fluid)
            dir = randn(dim)
            dir ./= norm(dir)
            for bin in 1:datasize
                r = (bin-1)*stepr         
                @views rfakeH .= rT + r*dir 
                if pbc
                    periodic!(rfakeH, L) 
                end            
                # sample free-particle path head -> fake head
                generate_path!(PIMC, beads, links, idlist)            
                Unew = 0.0
                buf = zeros(dim)
                for id in idlist[new_bead_indices]
                    # dangerous: U_update changes tmp storage, unless fake=true
                    if !beads.active[id]
                        println("not active!")
                    end
                    Unew += PIMC_Action_Interface.U_update(PIMC, beads, view(buf, :), id, :add; fake=true) # dummy old position
                end                
                # new U when particle moves from rT -> rfakeH := beads.X[:, PIMC.tail]
                Unew += PIMC_Action_Interface.U_update(PIMC, beads, rT, PIMC.tail, :move, fake=true)
                
                
                ΔU = Unew - Uold
                ΔU = clamp(ΔU, -100.0, 100.0) # avoid overflow
                # obdm ∝ <exp(-ΔS)>, rho_0(head, fake head) is left from staging
                ρ₀_val = rho_0(rH, rfakeH, λ, Δτ, L)                
                val = exp(-ΔU + μ*Δτ) * ρ₀_val
                if val>10.0
                    open("obdm_warn.log", "a") do io
                        println(io, "$(PIMC.ipimc) $bin $val $ΔU $(Δτ) $(rH) $(rfakeH) $(rT) $ρ₀_val")
                    end
                end
                # bin it            
                obdm[bin] += val
            end                    
        #end # inbounds
        
        # move tail back to original position
        @views beads.X[:, PIMC.tail] .= rT
        # undo temporary links 
        links.next[PIMC.head] = -1
        links.prev[PIMC.tail] = -1        
        obdm ./= nreps
    end # inbounds block

    @views beads.active[idlist[new_bead_indices]] .= false  
    
    add_sample!(meas.stat, obdm)

       
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks  = get_stats(meas.stat)
        # normalization
        # obdm(r) := rho_1(r) = Z_G/Z <exp(-ΔS)>_r
        # Z_G/Z = N_G/N_Z = (time spent in G-sector)/(time spent in Z-sector)        
        obdm_norm = worm_stats.N_open_acc * worm_stats.N_close_try/(worm_stats.N_open_try * worm_stats.N_close_acc * worm_C)
        ave .*= obdm_norm
        std .*= obdm_norm
       
        rs = zeros(datasize)
        @inbounds for i in 1:datasize
            rs[i] = (i-1)*stepr
        end
        
        open(meas.filename, "w") do f
            @inbounds for i in 1:datasize
                @printf(f, "%15.5f %15.5e %15.5e\n", rs[i], ave[i], std[i])
            end
        end
        
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "r" => rs,
            "obdm" => ave,
            "std" => std
        )
    end
end
      

#
# Radial distribution function g(r)
#


function meas_radial_distribution(PIMC::t_pimc, beads::t_beads, dummy::t_links, meas::t_measurement)

    datasize = length(meas.stat.sample.data)
    M = PIMC.M
    L = PIMC.L
    
    g = zeros(datasize)
    stepr = L/2/datasize
    n = 0
    ms = get_ms(M)

    @inbounds for m in ms       
        bs = active_beads_on_slice(beads, m)
        nbs = length(bs)
        id_pairs = [(bs[i], bs[j]) for i in 1:nbs-1 for j in i+1:nbs]
        @inbounds for (id1, id2) in id_pairs            
            r1 = @view beads.X[:, id1]
            r2 = @view beads.X[:, id2]            
            Δr = dist(r1, r2, L)
            bin = floor(Int64, Δr/stepr) + 1 
            if bin<=datasize
                n += 1
                g[bin] += 2
            end
        end
    end

    # Collect statistics
    add_sample!(meas.stat, g)
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks  = get_stats(meas.stat)
        # file output
        rho = N/L^dim
        c =  dim==3 ? 4/3*pi*N/L^3 : pi*N/L^2
        stepg = L/2/datasize
        rs = zeros(datasize)
        gs = zeros(datasize)
        stds = zeros(datasize)
        @inbounds for i in 1:datasize
            rlo = (i-1)*stepg 
            rhi = i*stepg 
            r =  rlo + 0.5*stepg
            a  =  length(ms) * N * c * (rhi^dim-rlo^dim) 
            rs[i] = r
            gs[i] = ave[i]/a
            stds[i] = std[i]/a           
        end
        if meas.filename != ""
            open(meas.filename, "w") do f
                @inbounds for i in 1:datasize
                    println(f, rs[i]," ", gs[i], " ", stds[i])
                end
            end
        end
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "r" => rs,
            "g" => gs,
            "std" => stds
        )
    end
end


#
# Static structure factor S(k)
#
mutable struct t_ks_kvecs
    ks  ::Vector{Float64}
    kvecs ::Matrix{Float64}
end

k_store = t_ks_kvecs(zeros(1), zeros(1,1))

function get_kvecs(L::Float64, nk::Int64)
    global k_store
    if length(k_store.ks)>1
        return k_store.ks, k_store.kvecs
    end
    Δk = 2π/L        
    ks = Vector{Float64}(undef, nk)
    kvecs = Matrix{Float64}(undef, dim, nk)

    done = false
    ik = 1
    if dim==3
        for i = 1:nk
            qx = (i-1)*Δk
            for j = 1:i
                qy = (j-1)*Δk
                for k = 1:j
                    qz = (k-1)*Δk
                    q = sqrt(qx^2+qy^2+qz^2)
                    if q < 1e-6
                        continue # skip q=0
                    end
                    # skip duplicates
                    if any(x->isapprox(x, q), ks) 
                        continue
                    end   
                    ks[ik] = q
                    @inbounds @views kvecs[:, ik] .= [qx, qy, qz]
                    ik += 1
                    if ik>nk
                        done = true
                        break
                    end
                end
                if done
                    break                
                end
            end
            if done
                break                
            end
        end
    elseif dim==2
        for i = 1:nk
            qx = (i-1)*Δk
            for j = 1:i
                qy = (j-1)*Δk                
                q = sqrt(qx^2+qy^2)
                if q < 1e-6
                    continue # skip q=0
                end
                # skip duplicates
                if any(x->isapprox(x, q), ks) 
                    continue
                end   
                ks[ik] = q
                @inbounds @views kvecs[:, ik] .= [qx, qy]
                ik += 1
                if ik>nk
                    done = true
                    break
                end
            end
            if done
                break                
            end
        end
    elseif dim==1
        for i = 1:nk
            qx = (i-1)*Δk
            q = sqrt(qx^2)
            if q < 1e-6
                continue # skip q=0
            end
            # skip duplicates
            if any(x->isapprox(x, q), ks) 
                continue
            end   
            ks[ik] = q
            @inbounds @views kvecs[:, ik] .= [qx]
            ik += 1
            if ik>nk
                break
            end
        end
    end
    
    sort_indices = sortperm(ks)
    ks = ks[sort_indices]
    @inbounds @views kvecs .= kvecs[:,sort_indices]
    
    k_store.ks = ks
    k_store.kvecs = kvecs
    return ks, kvecs
end


function meas_static_structure_factor(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement)
    """Static structure factor S(k) for PBC"""

    if !pbc
        return
    end
    
    # In each dimension,
    # wave vectors are 0, Δk, 2Δk, ... , kmax
    # Δk = 2π/L
    #
    # ρ_k = sum_{i=1}^N exp(i k⋅r_i)  density operator
    # S(k) = <ρ_k^† ρ_k> / N
    #
    M = PIMC.M
    L = PIMC.L
    
    nk = length(meas.stat.sample.data) # number of k values
    ks, kvecs = get_kvecs(L, nk)

    ρ = complex(zeros(nk))
    Sk = zeros(nk)
    ms = get_ms(M)
    for m in ms
        bs = active_beads_on_slice(beads, m)
        @inbounds ρ .= 0.0
        for b in bs
            r = @view beads.X[:, b]
            for i in 1:nk
                ki = @view kvecs[:,i]
                ρ[i] += exp(im * (ki ⋅ r))
            end
        end
        @inbounds @views @. Sk += real(ρ * conj(ρ))
    end
    @inbounds @views Sk ./= length(ms)*N
    add_sample!(meas.stat, Sk)

    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks, bdata  = get_stats(meas.stat)
        if meas.filename != ""
            open(meas.filename, "w") do f
                for i = 1:nk
                    @printf(f,"%8.3f  %8.5f  %-8.5f\n", ks[i], ave[i], std[i]) 
                end
            end
        end
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "Sk" => ave,
            "k" => ks,
            "std" => std
        )
        
    end
    
end



function meas_superfluid_fraction(PIMC::t_pimc, beads::t_beads, links::t_links, meas::Union{t_measurement, Nothing}= nothing)
    """Computes superfluid fraction from ⟨W^2⟩"""
    M = PIMC.M
    β = PIMC.β
    L = PIMC.L

    NN = count(beads.active)/M 
    # superfluid fraction    
    # rhos/rho = mL^2/(βħ^2N) <W^2> (3D, in the Book)
    #          = L^2 /(2 β (ħ^2/(2m)) N) <W^2>
    #          = L^2 /(2 β λ N) <W^2>
    #          = <(Px - x)^2/L^2> L^2 /(6 β λ N)
    #          = <(Px - x)>^2> /(6 β λ N) , so L cancels in superfluid fraction
    
    W = zeros(dim)
    dr = zeros(dim)
    #
    for id in beads.ids[beads.active]
        id_next = links.next[id]
        distance!(view(beads.X, :, id_next),  view(beads.X, :, id), L, dr) 
        W += dr
    end
       
    Wsq = sum(W.^2)/dim
    rhos = Wsq/(2*β*λ*NN)

    if meas === nothing
        @printf("rhos = %8.3f\n", rhos)
        return
    end

    add_sample!(meas.stat, rhos)
    
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks, bdata  = get_stats(meas.stat)
        @printf("rho_s: %8.5f ± %-8.5f \n", ave, std)
        if meas.filename != ""
            open(meas.filename, "a") do f
                # T rhos 
                @printf(f,"%8.3f  %8.5f  %-8.5f  %8.5f \n", 1/β, ave, std, bdata) 
            end
        end
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "rhos" => ave,
            "std" => std
        )
        
    end
    
end

        #
        

function meas_cycle_count(PIMC::t_pimc, beads::t_beads, links::t_links, meas::Union{t_measurement, Nothing}= nothing)
    """counts exchange cycles in current beads"""

    M = PIMC.M
    cc = zeros(Int64, N)
    # pick a bead at a time on slice t=1 and follow link until return; count full M cycles
    done = Int64[]
    active = beads.ids[beads.active]
    bs = [id for id in active if beads.ts[id]==1]
    

    @inbounds for id1 in bs
        id1 ∈ done && continue
        push!(done, id1)
        # follow links
        id = id1
        m = 1
        while true
            id = links.next[id]
            id == id1 && break
            if beads.ts[id]==1
                push!(done, id)
                m += 1 # passes slice t=1, one cycle full                
            end           
        end
        cc[m] += 1
    end

    if meas === nothing
        println("cycle counts:")
        @inbounds for i in 1:length(cc)
            cc[i]>0 && @printf("%d %8.5f \n", i, cc[i])
        end
        return
    end
    
    # Collect statistics
    add_sample!(meas.stat, cc)
    if meas.stat.finished
        # block full, report
        ave, std, _, _ = get_stats(meas.stat)
        # screen and file output
        println("cycle counts:")
        open(meas.filename, "w") do f            
            @inbounds for i in 1:length(cc)
                cc[i]>0 && @printf(f, "%d %8.5f \n", i, cc[i])
                cc[i]>0 && @printf("%d %8.5f \n", i, cc[i])
            end
        end
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "cc" => cc,
        )
    end
end

 


       
#
# Generic energy computations, not measurements
# =============================================
#
function E_pot_bead(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Potential energy of bead id"""    
    Epot::Float64 = 0.0 
    # an inactive bead has no potential energy
    beads.active[id] == false && (return Epot)

    r1 = @view beads.X[:, id]
    if PIMC.confinement_potential !== nothing        
        Epot += PIMC.confinement_potential(r1)
    end
    if PIMC.pair_potential !== nothing
        V = PIMC.pair_potential
        L = PIMC.L
        t = beads.ts[id]        
        # active beads on slice t not equal to id
        bs = filter(x -> x ≠ id, active_beads_on_slice(beads, m))
        nbs = length(bs)
        id_pairs = [(bs[i], bs[j]) for i in 1:nbs-1 for j in i+1:nbs]
        @inbounds for (id1, id2) in id_pairs            
            r1 = @view beads.X[:, id1]
            r2 = @view beads.X[:, id2]            
            Δr = dist(r1, r2, L)
            bin = floor(Int64, Δr/stepr) + 1 
            if bin<=datasize
                n += 1
                g[bin] += 2
            end
        end
    end

    # Collect statistics
    add_sample!(meas.stat, g)
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks  = get_stats(meas.stat)
        # file output
        rho = N/L^dim
        c =  dim==3 ? 4/3*pi*N/L^3 : pi*N/L^2
        stepg = L/2/datasize
        rs = zeros(datasize)
        gs = zeros(datasize)
        stds = zeros(datasize)
        @inbounds for i in 1:datasize
            rlo = (i-1)*stepg 
            rhi = i*stepg 
            r =  rlo + 0.5*stepg
            a  =  length(ms) * N * c * (rhi^dim-rlo^dim) 
            rs[i] = r
            gs[i] = ave[i]/a
            stds[i] = std[i]/a           
        end
        if meas.filename != ""
            open(meas.filename, "w") do f
                @inbounds for i in 1:datasize
                    println(f, rs[i]," ", gs[i], " ", stds[i])
                end
            end
        end
        # set HDF5 data Dict
        results[string(meas.name)] = Dict(
            "r" => rs,
            "g" => gs,
            "std" => stds
        )
    end
end

#
# Generic energy computations, not measurements
# =============================================
#
function E_pot_bead(PIMC::t_pimc, beads::t_beads, id::Int64)
    """Potential energy of bead id"""    
    Epot::Float64 = 0.0 
    # an inactive bead has no potential energy
    beads.active[id]==false && (return Epot)

    r1 = @view beads.X[:, id]
    if PIMC.confinement_potential !== nothing        
        Epot += PIMC.confinement_potential(r1)
    end
    if PIMC.pair_potential !== nothing
        V = PIMC.pair_potential
        L = PIMC.L
        t = beads.ts[id]        
        # active beads on slice t not equal to id
        bs = filter(x -> x ≠ id, active_beads_on_slice(beads, t))
        #
        # vectorized form
        #    nbs = length(bs)
        #    Xv = @view beads.X[:, bs]
        #    # beads on different world lines, use min. image dist for pbc
        #    dXs  = [dist(Xv[:, i], r1, L) for i in 1:nbs]
        #    Epot += sum(Vs(dXs))            
        
        # non-vectorized form is a bit faster
        if pbc
            @inbounds for id2 in bs
                # beads on different world lines, use min. image dist for pbc
                r = dist(r1, view(beads.X, :, id2), L)
                r > L/2 && continue #cutoff
                Epot += V(r)
            end
            Epot += PIMC_Common.Vtail # add tail correction, if any
        else
            @inbounds for id2 in bs                   
                Epot += V(norm(r1 - view(beads.X, :, id2)))
            end
        end
        
    end
    return Epot
end

function E_pot_all(PIMC::t_pimc, beads::t_beads)
    """Potential energy of the whole system"""

    Epot = 0.0
    ms = get_ms(PIMC.M)
    for m in ms
        # all active beads on slice m
        bs = active_beads_on_slice(beads, m)
        Xv = @view beads.X[:, bs]        
        if PIMC.confinement_potential !== nothing
            Epot += sum(PIMC.confinement_potential(Xv))            
        end
        if PIMC.pair_potential !== nothing
            V = PIMC.pair_potential
            L = PIMC.L
            #t1 = @elapsed begin            
            nbs = length(bs)               
            if pbc
                # beads on different world lines, use min. image dist for pbc                
                # cutoff at L/2
                dXs = Float64[]  
                @inbounds for i in 1:nbs-1
                    @inbounds for j in i+1:nbs
                        d = dist(view(Xv, :, i), view(Xv, :, j), L)
                        if d ≤ L/2
                            push!(dXs, d)
                        end
                    end
                end
            else
                dXs  = [norm(Xv[:, i]-Xv[:, j]) for i in 1:nbs-1 for j in i+1:nbs]                
            end
            Epot += sum(V.(dXs))
            #=
            #end
            #t2 = @elapsed begin
            # non-vectorized form, almost always slower
            nbs = length(bs)
            Etest = 0
            id_pairs = [(bs[i], bs[j]) for i in 1:nbs-1 for j in i+1:nbs]
            for (id1, id2) in id_pairs        
                r1 = @view beads.X[:, id1]
                r2 = @view beads.X[:, id2]
                # beads on different world lines, use min. image dist for pbc
                r12 = dist(r1, r2, L)
                if r12 ≤ L/2
                    Etest += V(r12)
                end
            end
            Epot += Etest
        
            #@show Etest/N, sum(V.(dXs))# , t1, t2, t1<t2
            =#
            
        end          
    end
    if PIMC.pair_potential !== nothing
        Epot += PIMC_Common.Vtail*N # add tail correction, if any
    end
    return Epot/length(ms) # works for both prim and chin
end



function boson_virial_exchange_energy(PIMC::t_pimc, beads::t_beads, links::t_links)
    """Boson virial estimator exchange energy term for co-moving centroid reference."""

    if !bose
        error("Boson virial estimator works only for bosons :)")
    end
    M = PIMC.M
    L = PIMC.L
    β = PIMC.β
    τ = PIMC.τ
    
    # exchange energy term
    
    dx_long = zeros(dim)  # pre-allocate 
    dx_short = zeros(dim) 
    r = zeros(dim)
    
    Δτ = 0.0 
    Eexc = 0.0
    dr = zeros(dim)
    for idm in beads.ids[beads.active]
        r_m = @view beads.X[:,idm] # X_m 
        r .= r_m  # displacements are relative to r (updated along the way)
        id = idm
        for m in 1:M
            id = links.next[id]
            distance!(view(beads.X, :, id), view(r,:), L, dr)
            if m==M-1
                # X_M+m - X_M+m-1, short distance vector
                dx_short .= dr
                Δτ = mod(beads.times[id] - beads.times[links.prev[id]], β)        
            end
            r .+= dr
        end
        r_Mm = r # X_M+m
        # long continuous path vector (on same slice), *not* min. image distance         
        dx_long .= r_Mm - r_m  # X_M+m - X_m
        Eexc += dx_long ⋅ dx_short/Δτ  
    end
    Eexc *= -1/(4*λ*M*β)
    Eexc
end


end
