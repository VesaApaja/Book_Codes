__precompile__(false)
module PIMC_Measurements

using Random, LinearAlgebra
using StaticArrays
using Printf
using BenchmarkTools


# local modules:
push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Utilities
using PIMC_Structs: t_pimc, t_beads, t_links, t_measurement, check_links
using QMC_Statistics
using PIMC_Systems

using PIMC_Action_Interface

export add_measurement!
export rho_s
export meas_density_profile, meas_superfluid_fraction, meas_radial_distribution
export meas_cycle_count, meas_obdm, meas_head_tail_histogram
export meas_static_structure_factor, meas_static_structure_factor_old
export meas_consistency

function add_measurement!(PIMC::t_pimc, frequency::Int, name::Symbol,
                          exe::Function, datasize::Int, blocksize::Int,
                          filename::String)
    stat = init_stat(datasize, blocksize)
    sname = String(name)
    measurement = t_measurement(frequency, name, sname, exe, stat, filename)
    push!(PIMC.measurements, measurement)    
    println("Added measurement: ",name," frequency ",frequency)
end



# lightweight buffers, not thread safe!
const vec_buffer = Vector{Float64}(undef, dim)
const vec_buffer2 = Vector{Float64}(undef, dim)



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
        results[meas.sname] = Dict(
            "x" => xs, 
            "rho" => ave,
            "std" => std
        )
    end   
end


mutable struct t_head_tail_histo    
    nbins::Int
    ndata::Int
    histo::Vector{Float64}
end
const head_tail_buffer = t_head_tail_histo(1000, 0, zeros(1000)) # buffer, should be big enough     

function meas_head_tail_histogram(PIMC::t_pimc, beads::t_beads, dummy::t_links, meas::t_measurement,
                                  head_tail::t_head_tail_histo = head_tail_buffer
                                  )

    
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
        @inbounds @views begin
            head_tail.histo[1:datasize] ./= 100
            add_sample!(meas.stat, head_tail.histo[1:datasize])
            head_tail.ndata = 0
            head_tail.histo[1:datasize] .= 0
        end
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
        results[meas.sname] = Dict(
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
const ave_buffer = Vector{Float64}(undef, 1000) # fixed size buffer
const std_buffer = Vector{Float64}(undef, 1000) # fixed size buffer
const rT_buffer = MVector{dim, Float64}(undef)
const dir_buffer =  MVector{dim, Float64}(undef)
const xold_buffer = MVector{dim, Float64}(undef)
const idlist_buffer = Vector{Int}(undef, M_max)

function meas_obdm(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement,
                   obdm_buf::Vector{Float64} = obdm_buffer,
                   ave_buf::Vector{Float64} = ave_buffer,
                   std_buf::Vector{Float64} = std_buffer,
                   rT::MVector{dim, Float64} = rT_buffer,
                   dir::MVector{dim, Float64} = dir_buffer,
                   xold::MVector{dim, Float64} = xold_buffer,
                   idlist_buf::Vector{Int} = idlist_buffer
                   )
    """One-body density matrix: histogram of same-time head-tail distances"""
    
    if PIMC.head == -1
        return # no worm
    end
    M = PIMC.M
    ms = get_ms(M)

    if beads.ts[PIMC.tail] ∉ ms || beads.ts[PIMC.head] ∉ ms
        # Head and tail should be on a physical slice (any slice in Primitive, some Chin Action slices)
        # Just double checking, this should be taken care already in worm moves 
        return 
    end
    
    
    L = PIMC.L    
    β = PIMC.β
    μ = PIMC.μ

        
    datasize = length(meas.stat.sample.data)
    #@assert datasize<length(obdm_buf) "buffer too small"
    obdm = @view obdm_buf[1:datasize]
    @inbounds @views obdm .= 0.0
    
    stepr = 0.02 # default
    if pbc
        stepr = L/2/datasize  
    end
    #
    # fake close, head on the same slice as tail, but at distance r from tail
    #
    # need new beads from head to fake head (which is at the tail slice m)
    k = mod1(beads.ts[PIMC.tail] - beads.ts[PIMC.head], M)
    idlist = @view idlist_buf[1:k+1]
    # idlist:
    #  1    2   3 ...  k    k+1
    # head                  fake head
    # old  new  new ...     sample first 
        
    m = beads.ts[PIMC.head]
    idlist[1] = PIMC.head
    # find free beads for new path segment
    @inbounds for i in 2:k
        m = mod1(m+1, M)
        idlist[i] = beads.inactive_at_t[m][1]
    end
    idlist[end] = PIMC.tail
    # store head and tail
    store_X!(beads, PIMC.head, PIMC.tail)

    
    Δτ = wrapβ(beads.times[PIMC.tail] - beads.times[PIMC.head], β) # head -> fake head
    #
    # inter-action of moved beads
    Uold = 0.0    
    @inbounds for b in idlist[2:end]
        Uold += U_stored(PIMC, beads, b) # picks slice index based on bead id
    end    
        
    # pick fake head on the same slice as tail and generate free-particle path head->fake head
    # temporarily move particle from tail to fake head to measure <Ψ^†(r')Ψ(r)>
    #
    midbeads = @view idlist[2:end-1]
    activate_beads!(beads, midbeads)
    
    @inbounds for d in 1:dim
        rT[d]= beads.X[d, PIMC.tail] # fixed reference point
    end
    # from now on, fake head is stored to PIMC.tail
    @inbounds begin       
        nreps = 5
        for rep in 1:nreps # a few directions
            # dir is a random direction (assumes isotropic homogeneous fluid)
            randn!(dir)
            dir ./= norm(dir)
            for bin in 1:datasize
                r = (bin-1)*stepr
                # fake head at distance r from tail
                for d in 1:dim
                    # -------------------------
                    x = rT[d] + r*dir[d]
                    if pbc
                        x -= L * floor((x + L/2) / L)
                    end                        
                    beads.X[d, PIMC.tail] = x
                    # -------------------------
                end
                # sample path head -> fake head
                # --------------------------------------------               
                generate_path!(PIMC, beads, links, idlist)
                # --------------------------------------------                
                Unew = 0.0           
                for id in midbeads
                    # dangerous: U_update changes tmp storage unless fake=true
                    Unew += U_update(PIMC, beads, xold, id, :add; fake=true) # dummy old position xold
                end                
                # new U when particle moves from rT -> rfakeH := beads.X[:, PIMC.tail]
                Unew += U_update(PIMC, beads, rT, PIMC.tail, :move; fake=true) 
                ΔU = Unew - Uold           
                ΔU = clamp(ΔU, -100.0, 100.0) # avoid overflow                
                obdm[bin] += exp(-ΔU + μ*Δτ)*rho_0(beads.X, PIMC.head, PIMC.tail, λ, Δτ, L) 
            end
        end 

        
        # undo temporary links created by generate_path!
        links.next[PIMC.head] = -1
        links.prev[PIMC.tail] = -1        
        obdm ./= nreps
        
    end # inbounds
    
    
    # restore head and tail
    restore_X!(beads,  PIMC.head, PIMC.tail)
    # deactivate path segment
    deactivate_beads!(beads, midbeads)
        
    add_sample!(meas.stat, obdm)

   
    if meas.stat.finished
        # block full, report
        # tried to use ave = @view obdm_buf[1:datasize], but the hdf5 output gets wrong!
        ave = @view ave_buf[1:datasize]
        std = @view std_buf[1:datasize]
        get_stats!(meas.stat, ave, std)
        # normalization
        # obdm(r) := rho_1(r) = Z_G/Z <exp(-ΔS)>_r
        # Z_G/Z = N_G/N_Z = (time spent in G-sector)/(time spent in Z-sector)        
        obdm_norm = worm_stats.N_open_acc * worm_stats.N_close_try/(worm_stats.N_open_try * worm_stats.N_close_acc * worm_C)
        @inbounds @views begin
            ave .*= obdm_norm
            std .*= obdm_norm
        end
        
        rs = collect(0:stepr:(datasize-1)*stepr) # small allocation
        
        open(meas.filename, "w") do f
            @inbounds for i in 1:datasize
                @printf(f, "%15.5f %15.5e %15.5e\n", rs[i], ave[i], std[i])
            end
        end
        
        # set HDF5 data Dict
        results[meas.sname] = Dict(
            "r" => rs,
            "obdm" => ave,
            "std" => std
        )
    end
end
      

#
# Radial distribution function g(r)
#

const g_buffer = Vector{Float64}(undef, 1000) # should be enough
const gs_buffer = Vector{Float64}(undef, 1000) 
const rs_buffer = Vector{Float64}(undef, 1000)
const stds_buffer = Vector{Float64}(undef, 1000)


function meas_radial_distribution(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement,
                                  g_buf::Vector{Float64} =  g_buffer,
                                  gs_buf::Vector{Float64} =  gs_buffer,
                                  rs_buf::Vector{Float64} =  rs_buffer,
                                  stds_buf::Vector{Float64} =  stds_buffer,
                                  )

    datasize = length(meas.stat.sample.data)
    M = PIMC.M
    L = PIMC.L
    
    g = @view g_buf[1:datasize]
    @inbounds @views g .= 0.0
    stepr = L/2/datasize
    n = 0
    ms = get_ms(M)

    @inbounds begin
        for m in ms       
            bs = beads.active_at_t[m]
            nbs = length(bs)
            for i in 1:nbs-1
                r1 = @view beads.X[:, bs[i]]
                for j in i+1: nbs
                    r2 = @view beads.X[:, bs[j]]            
                    Δr = dist(r1, r2, L)
                    bin = floor(Int, Δr/stepr) + 1 
                    if bin<=datasize
                        n += 1
                        g[bin] += 2
                    end
                end
            end
        end
    end

    # Collect statistics
    add_sample!(meas.stat, view(g, :))
    
    if meas.stat.finished
        # block full, report
        rs = @view rs_buf[1:datasize]
        gs = @view gs_buf[1:datasize]
        stds = @view stds_buf[1:datasize]
        #
        input_σ2, Nblocks  = get_stats!(meas.stat, gs, stds)
        # file output
        rho = N/L^dim
        c =  dim==3 ? 4/3*pi*N/L^3 : pi*N/L^2        
        @inbounds for i in 1:datasize
            rlo = (i-1)*stepr 
            rhi = i*stepr 
            r =  rlo + 0.5*stepr
            a  =  length(ms) * N * c * (rhi^dim-rlo^dim) 
            rs[i] = r
            gs[i] /= a
            stds[i] /= a           
        end
        if meas.filename != ""
            open(meas.filename, "w") do f
                @inbounds for i in 1:datasize
                    println(f, rs[i]," ", gs[i], " ", stds[i])
                end
            end
        end
        # set HDF5 data Dict
        results[meas.sname] = Dict(
            "r" => rs,
            "g" => gs,
            "std" => stds
        )
    end
    
end


#
# Static structure factor S(k)
#
mutable struct t_kstore
    ks ::Vector{Float64}
    bin2i :: Vector{Vector{Int}}
    kvecs ::Matrix{Float64}
end

const kstore_ref =  Ref{Union{Nothing, t_kstore}}(nothing)


function get_kvecs(nk::Int, Δk::Float64, krec::Float64, kmax::Float64, dk::Float64)
    # lazy init
    if kstore_ref[] === nothing
        ks_b = Vector{Float64}(undef, (nk+1)^dim)
        kvecs_b = Matrix{Float64}(undef, dim, (nk+1)^dim)
        
        if dim==3
            i = 0
            for n1 in 0:nk, n2 in 0:nk, n3 in 0:nk                
                n1 == 0 && n2 == 0 && n3 == 0 && continue # skip trivial k=0
                kvec = [n1, n2, n3] .* Δk
                k = norm(kvec)
                k>krec && continue
                i += 1   
                kvecs_b[:, i] .= kvec
                ks_b[i] = k
            end
            
        elseif dim==2
            i = 0
            for n1 in 0:nk, n2 in 0:nk
                n1 == 0 && n2 == 0 && continue 
                n1^2 + n2^2 > nk^2 && continue
                kvec = [n1, n2] .* Δk
                k = norm(kvec)
                k>krec && continue
                i += 1
                kvecs_b[:, i] .= kvec
                ks_b[i] = k
                
            end
        else
            # dim==1
            i = 0
            for n1 in 0:nk
                n1 == 0 && continue
                kvec = [n1*Δk]
                k = norm(kvec)
                k>krec && continue
                i += 1
                kvecs_b[:, i] .= kvec
                ks_b[i] = k
                
            end
        end
        # fill k>krec k-vectors
        # random directions
        dirs = []
        dir = MVector{dim, Float64}(undef)
        for i in 1:3
            randn!(dir)
            dir ./= norm(dir)
            push!(dirs, dir)
        end
        k = krec
        while k<5*kmax # overkill (hopefully) to get nk values in the end
            k += dk
            for dir in dirs
                i += 1
                kvecs_b[:, i] .= k*dir
                ks_b[i] = k
            end
        end
        maxind = i
        ks = Vector{Float64}(undef, maxind)
        kvecs = Matrix{Float64}(undef, dim, maxind)
        ks .= ks_b[1:maxind]
        kvecs .= kvecs_b[:, 1:maxind]
        sort_indices = sortperm(ks)
        ks = ks[sort_indices]
        @inbounds @views kvecs .= kvecs[:, sort_indices]
        bin2i = Vector{Vector{Int}}()
        i = 1
        while i ≤ maxind
            j = i
            while j ≤ maxind && isapprox(ks[j], ks[i])
                j += 1
            end
            push!(bin2i, collect(i:j-1))
            length(bin2i)==nk && break
            i = j            
        end
        ks_nodup = Vector{Float64}(undef, nk)
        ks_nodup[1] =  ks[1]
        j = 1
        for i in 2:maxind
            isapprox(ks[i], ks_nodup[j]) && continue
            j += 1
            ks_nodup[j] = ks[i]
            j==nk && break
        end
        ks = ks_nodup
        if length(bin2i) != nk || length(ks) != nk
            @show length(bin2i),  length(ks), nk
            error("couldn't get nk k-vectors, compute more k-vectors after krec")
        end
        kstore_ref[] = t_kstore(ks, bin2i, kvecs)        
    end
    return kstore_ref[]
end


const Sk_buffer = Vector{Float64}(undef, 1000) # overkill

function meas_static_structure_factor(PIMC::t_pimc, beads::t_beads, links::t_links, meas::t_measurement,
                                      Sk_buf::Vector{Float64} = Sk_buffer)
    """Static structure factor S(k) for PBC"""

    if !pbc
        return
    end
    L = PIMC.L
    M = PIMC.M
    
    # In each dimension,
    # wave vectors are 0, Δk, 2Δk, ... , kmax
    # Δk = 2π/L
    #
    # ρ_k = sum_{i=1}^N exp(i k⋅r_i)  density operator
    # S(k) = <ρ_k^† ρ_k> / N
    #      =  < sum_{i=1}^N exp(-i k⋅r_i) sum_{j=1}^N exp(i k⋅r_j)> /N
    #      =  < sum_{i=1}^N (cos(k⋅r_i) - i*sin(k⋅r_i)) * sum_{j=1}^N (cos(k⋅r_j) + i*sin(k⋅r_j))
    # mark re = sum_{i=1}^N (cos(k⋅r_i)
    #      im = sum_{i=1}^N (sin(k⋅r_i)
    # to get
    # S(k) = < (re - i*im) * (re + i*im)>/N
    #      = <re^2 + im^2>/N
    # average also over physical slices
    #
    nk = length(meas.stat.sample.data) 
    Sk = @view Sk_buf[1:nk]
    # 
    # get k-vectors and their lengths k
    # =================================
    Δk = 2π/L
    krec = 4*Δk # for k<krec use all reciprocal lattice vectors
    kmax = 10.0 
    dk = kmax/nk # fixed k-step after krec, kvecs in a few random directions
    kstore = get_kvecs(nk, Δk, krec, kmax, dk)  
    ks = kstore.ks
    bin2i= kstore.bin2i
    kvecs = kstore.kvecs
    #
    kvec = MVector{dim, Float64}(undef)
    ms = get_ms(M) # physical slices
    len_ms = length(ms)
    
    @inbounds begin               
        for bin in 1:nk
            Sk_bin = 0.0
            for i in bin2i[bin]
                kvec .= kvecs[:, i]                               
                for m in ms
                    re = 0.0
                    im = 0.0
                    nn = 0
                    bs = beads.active_at_t[m]
                    for b in bs
                        θ = dot(kvec, view(beads.X, :, b)) # k⋅r_i
                        re += cos(θ)
                        im += sin(θ)
                        nn += 1
                    end
                    Sk_bin += (re^2 + im^2)/nn
                end   
            end
            nkvec = length(bin2i[bin])            
            Sk[bin] = Sk_bin/(nkvec*len_ms)
        end
    end
    
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
        results[meas.sname] = Dict(
            "Sk" => ave,
            "k" => ks,
            "std" => std
        )
        
    end
    
end


const rho_s_W_buffer = MVector{dim, Float64}(undef)
const rho_s_dr_buffer = MVector{dim, Float64}(undef)

function meas_superfluid_fraction(PIMC::t_pimc, beads::t_beads, links::t_links, meas::Union{t_measurement, Nothing}= nothing,
                                  W::MVector{dim, Float64} = rho_s_W_buffer,
                                  dr::MVector{dim, Float64} = rho_s_dr_buffer
                                  )
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

    @inbounds @views W .= 0.0
    #
    @inbounds for id in beads.ids[beads.active]
        id_next = links.next[id]
        distance!(view(beads.X, :, id_next), view(beads.X, :, id), L, dr) 
        @views W .+= dr
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
        results[meas.sname] = Dict(
            "rhos" => ave,
            "std" => std
        )
        
    end
    
end

#
const cc_buffer = Vector{Int}(undef, N_slice_max)        

function meas_cycle_count(PIMC::t_pimc, beads::t_beads, links::t_links, meas::Union{t_measurement, Nothing}= nothing,
                          cc_buf::Vector{Int} = cc_buffer 
                          ) 
    """counts exchange cycles in current beads"""

    M = PIMC.M
    cc = @view cc_buf[1:N]
    @inbounds @views cc .= 0
    # pick a bead at a time on slice t=1 and follow link until return; count full M cycles
    
    bs = beads.active_at_t[1] # active beads on slice 1   
    done = falses(maximum(bs)) # BitVector, record done beads on slice 1
    @inbounds for id1 in bs
        done[id1] && continue
        done[id1] = true
        # follow links
        id = id1
        m = 1
        while true
            id = links.next[id]
            id == id1 && break
            if beads.ts[id]==1
                done[id] = true
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
        results[meas.sname] = Dict(
            "cc" => cc,
        )
    end
end

function meas_consistency(PIMC::t_pimc, beads::t_beads, links::t_links, meas::Union{t_measurement, Nothing}= nothing)
    """Consistency check for noninteracting particles sampling"""
    
    β = PIMC.β
    M = PIMC.M
    L = PIMC.L
    #
    # kinetic_term = ∑_{m=1}^M (x_m - x_{m+1})^2/(4λτt_m) (pbc uses min. image distances)
    #
    kinetic_term::Float64 = 0.0
    @inbounds begin
        for id in beads.ids[beads.active]
            id_next = links.next[id]
            Δr2 = dist2(beads.X, id, id_next, L)        
            Δτ  = wrapβ(beads.times[id_next] - beads.times[id], β)
            kinetic_term += Δr2/Δτ
        end
    end
    kinetic_term *= 1/4λ
    #
    # exchange_term = -(x_{M+1}-x_1) ⋅ (x_{M+1} - x_M)(4λτt_M)
    #               = - 1/M * ∑_{m=1}^M (x_{M+m} - x_m) ⋅ (x_{M+m}-x_{M+m-1})/(4λτt_{M+m-1})
    #
    exchange_term::Float64 = β * boson_virial_exchange_energy(PIMC, beads, links)
    #
    # 
    # <kinetic_term + exchange_term> = dim*N*(M-1)/2
    # 
    add_sample!(meas.stat, SVector{2, Float64}(kinetic_term, exchange_term) )
    
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks, bdata  = get_stats(meas.stat)
        if meas.filename != ""
            open(meas.filename, "a") do f
                # T <kinetic_term>/N + <exchange_term>/N  dim*(M-1)/2
                @printf(f,"%8.3f  %8.5f  %-8.5f  %8.5f  %-8.5f  %8.5f \n", 1/β, ave[1]/N, std[1]/N,
                        ave[2]/N, std[2]/N, dim*(M-1)/2) 
            end
            # compare dim*(M-1)/2 - <kinetic_term>/N with <exchange_term>/N in gnuplot
            # gnuplot> plot 'consistency.Nonint...dat' u 0:($6-$2):($3) w errorl,'' u 0:4:5 w errorl
        end
        # set HDF5 data Dict
        results[meas.sname] = Dict(
            "kinetic" => ave[1]/N,
            "std_kinetic" => std[1]/N,
            "exchange" => ave[2]/N,
            "std_exchange" => std[2]/N,
            "rhs" => dim*(M-1)/2
        )
        
    end
end


function meas_consistency(PIMC::t_pimc, beads::t_beads, links::t_links, meas::Union{t_measurement, Nothing}= nothing)
    """Consistency check for noninteracting particles sampling"""
    
    β = PIMC.β
    M = PIMC.M
    L = PIMC.L
    #
    # kinetic_term = ∑_{m=1}^M (x_m - x_{m+1})^2/(4λτt_m) (pbc uses min. image distances)
    #
    kinetic_term::Float64 = 0.0
    @inbounds begin
        for id in beads.ids[beads.active]
            id_next = links.next[id]
            Δr2 = dist2(beads.X, id, id_next, L)        
            Δτ  = wrapβ(beads.times[id_next] - beads.times[id], β)
            kinetic_term += Δr2/Δτ
        end
    end
    kinetic_term *= 1/4λ
    #
    # exchange_term = -(x_{M+1}-x_1) ⋅ (x_{M+1} - x_M)(4λτt_M)
    #               = - 1/M * ∑_{m=1}^M (x_{M+m} - x_m) ⋅ (x_{M+m}-x_{M+m-1})/(4λτt_{M+m-1})
    #
    exchange_term::Float64 = β * boson_virial_exchange_energy(PIMC, beads, links)
    #
    # 
    # <kinetic_term + exchange_term> = dim*N*(M-1)/2
    # 
    add_sample!(meas.stat, SVector{2, Float64}(kinetic_term, exchange_term) )
    
    if meas.stat.finished
        # block full, report
        ave, std, input_σ2, Nblocks, bdata  = get_stats(meas.stat)
        if meas.filename != ""
            open(meas.filename, "a") do f
                # T <kinetic_term>/N + <exchange_term>/N  dim*(M-1)/2
                @printf(f,"%8.3f  %8.5f  %-8.5f  %8.5f  %-8.5f  %8.5f \n", 1/β, ave[1]/N, std[1]/N,
                        ave[2]/N, std[2]/N, dim*(M-1)/2) 
            end
            # compare dim*(M-1)/2 - <kinetic_term>/N with <exchange_term>/N in gnuplot
            # gnuplot> plot 'consistency.Nonint...dat' u 0:($6-$2):($3) w errorl,'' u 0:4:5 w errorl
        end
        # set HDF5 data Dict
        results[meas.sname] = Dict(
            "kinetic" => ave[1]/N,
            "std_kinetic" => std[1]/N,
            "exchange" => ave[2]/N,
            "std_exchange" => std[2]/N,
            "rhs" => dim*(M-1)/2
        )
        
    end
end










end
