__precompile__(false)
module PIMC_Utilities

push!(LOAD_PATH,".")
using PIMC_Common
using PIMC_Structs:t_beads, t_links, t_pimc

export distance_check
export store_X!, restore_X!, storage
export rho_0, active_beads_on_slice, bisection_segment!
export backup_state!, restore_state!

export save_paths
export argparse



# =====================================
# ArgParse would do this, but I hate to load yet another package

function argparse(known_args ::Vector{String})
    if length(ARGS)==0; return ; end

    got_args = Dict() 
    for arg in ARGS
        if occursin("=",arg)
            parts = split(arg, "=", keepempty=false)
            var = parts[1]
            if var ∉ known_args
                @show(known_args)
                println("Error reading command line arguments, variable $var is unknown")
                exit(1)
            end
            try
                value = parts[2]                
                if occursin(".",value)
                    value = parse(Float64,value)
                else
                    try
                        value = parse(Int64,value)
                    catch
                        # assume it's a string
                        value = String(value)
                    end
                end
                #println("command line: ",var,"=",value)
                got_args[var]=value
            catch
                println("Error reading command line arguments, use var=val")
                exit(1)
            end
        end
    end
    return got_args
end
# ======================================




@inline function active_beads_on_slice(beads::t_beads, m::Int64)
    mask = beads.active[beads.at_t[m]]  
    return beads.at_t[m][mask]      
end

# Temporary storage for a few bead positions in worm moves and in bisection

struct t_storage
    buffer::Matrix{Float64} # Preallocated storage
    k::Base.RefValue{Int64}   # number of stored values
end

function t_storage(K::Int64)
    t_storage(zeros(dim, K),  Ref(0))
end
storage = t_storage(max(10, 10000)) # should be enough  


function store_X!(storage::t_storage, beads::t_beads, idlist::AbstractVector{Int64})
    k = length(idlist)
    k > size(storage.buffer, 2) && error("Storage too small")
    storage.k[] = k           # set number of stored elements
    @inbounds for i = 1:k, d = 1:dim
        storage.buffer[d, i] = beads.X[d, idlist[i]] 
    end
end

# Restore values if move is rejected
function restore_X!(storage::t_storage, beads::t_beads,  idlist::AbstractVector{Int64})
    k = storage.k[]   # get number of stored elements
    @inbounds for i = 1:k, d = 1:dim
        beads.X[d, idlist[i]] = storage.buffer[d, i]
    end
end


function backup_state!(beads_backup::t_beads, links_backup::t_links, 
                       beads::t_beads, links::t_links)
    # these are already fixed: beads.ids, beads.ts, beads.at_t, beads.times       
    @inbounds beads_backup.X .= beads.X
    @inbounds beads_backup.active .= beads.active
    @inbounds links_backup.next .= links.next
    @inbounds links_backup.prev .= links.prev
end

function restore_state!(beads::t_beads, links::t_links, 
                        beads_backup::t_beads, links_backup::t_links)
    @inbounds beads.X .= beads_backup.X
    @inbounds beads.active .= beads_backup.active 
    @inbounds links.next .= links_backup.next
    @inbounds links.prev .= links_backup.prev    
end


@inline function bisection_segment!(PIMC::t_pimc, beads::t_beads, links::t_links, idlist::AbstractVector{Int64})
    """Bisection of path between known beads idlist[1] and idlist[end] for any number of slices"""
    n = length(idlist)
    if n==2
        # no beads to generate, just link'm
        links.next[idlist[1]] = idlist[2]
        links.prev[idlist[2]] = idlist[1]
        return
    end
    if n<2
        error("Can't bisect path without two end points")
    end
    stack = [(1, n)] # known indices in bead list idlist
    β = PIMC.β
    L = PIMC.L
   
    while !isempty(stack)
        (i, j) = pop!(stack)
        if j - i <= 1
            continue  # no points needed between beads idlist[i] and idlist[j]
        end
        m = div(i + j, 2)

        τ_ij = mod(beads.times[idlist[j]] - beads.times[idlist[i]], β)
        τ_im = mod(beads.times[idlist[m]] - beads.times[idlist[i]], β)
        τ_mj = mod(beads.times[idlist[j]] - beads.times[idlist[m]], β) 
        τ = τ_im * τ_mj / τ_ij
        X = @view beads.X[:, idlist[m]]
        @inbounds for d in 1:dim
            xi = beads.X[d, idlist[i]]
            xj = beads.X[d, idlist[j]]
            dx = xj - xi
            if pbc
                dx -= L * round(dx/L)
            end
            # interpolated midpoint
            mean = xi + dx * (τ_im/τ_ij)            
            # without pbc, mean = (τ_mj * xi + τ_im * xj) / τ_ij
            # with pbc, replace xj with xi+dx 
            # (τ_mj * xi + τ_im * xj)/τ_ij ->  (τ_mj * xi + τ_im * (xi+dx))/τ_ij = xi + dx * (τ_im/τ_ij)  
            X[d] = mean + sqrt(2λ*τ) * randn()
            if pbc
                X[d] = mod(X[d] + L/2, L) - L/2
            end
        end
        # push left and right halves to stack 
        push!(stack, (i, m))
        push!(stack, (m, j))
    end
    for i in 1:length(idlist)-1
        links.next[idlist[i]] = idlist[i+1]
        links.prev[idlist[i+1]] = idlist[i]
    end
end



@inline function rho_0(r1::SubArray{Float64}, r2::SubArray{Float64}, λ::Float64, τ12::Float64, L::Float64)    
    """Free-particle density matrix ⟨r1 | e^(-τ T) | r2⟩ for one particle"""
    Δr2 = dist2(r1, r2, L)    
    return (4π*λ*τ12)^(-dim/2) * exp(-Δr2 /(4*λ*τ12))  
end


function distance_check(PIMC::t_pimc, beads::t_beads, links::t_links)
    for id in beads.active
        next = links.next[id]
        if next>0
            Δr = dist(beads.X[:, id], beads.X[:, next], PIMC.L)
            r  = norm(beads.X[:, id] - beads.X[:, next])
            if !isapprox(Δr, r)
                @show id, next, Δr, r
                error("min. image and euclidian distances are not the same for bead and next bead")
            end
        end
    end
end




function save_paths(PIMC::t_pimc, beads::t_beads, links::t_links, col::Int64=1)

    L = PIMC.L
    
    mode = col==1 ? "w" : "a"
    open("paths", mode) do f
        # full world lines x(τ) in 1D, just coordinates in 2D and 3D
        for id in beads.active
            #if dim==1
                println(f,  beads.X[1, id], " ", beads.ts[id], " ", col)
            #else
            #    println(f,  beads.X[1, id], " ", beads.X[2, id], " ",  col)
            #end
            
            id_next = links.next[id]            
            if id_next > 0
                if beads.ts[id_next]==1
                    println(f," ")
                end
                #if dim==1
                    println(f,  beads.X[1, id_next], " ", beads.ts[id_next], " ",  col)
                #else
                #    println(f,  beads.X[1, id_next], " ",  beads.X[2, id_next], " ",  col)
                #end
            end
            println(f," ")            
        end
    end
    #if col==2
    #    println("saved paths (press enter if stopped)")
    #    readline()
    #end
end


end
