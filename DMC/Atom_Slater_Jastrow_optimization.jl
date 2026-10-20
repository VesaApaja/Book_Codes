#
# Optimization of basis set parameters
# For example, in order to optimize B atom using triple-zeta STO basis, run
# 
# julia Atom_Slater_Jastrow_optimization.jl atom=B basis_set=STO_Tzeta
#
#    raw output to file Eopt_B_STO_Tzeta
#
#    optimized input is stored to file atom_data/B_STO_Tzeta_opt,
#    which can be copied to atom_data/B_STO_Tzeta and used as input.
#

using Printf
using StaticArrays
using Statistics
using ForwardDiff: gradient as ADgrad 
# dot product:
using LinearAlgebra: ⋅  

# slightly faster (?) norm 
@inline norm(x) = sqrt(sum(abs2,x))

# local modules:
push!(LOAD_PATH,".")
using Common
using Jastrow: φ_T_J as φ_T_J_in, F_J as F_J_in
using VMCstep
using Utilities
using QMC_Statistics
using SlaterDeterminants
using LinearOptimization
using STOOrbitals
using CSF

# for testing purposes only:
#using Random
#Random.seed!(1234)
#@warn "using fixed random seed"



possible_args=["atom","basis_set"]
arg_dict = argparse(possible_args)
# use command line values or defaults
const atom = get(arg_dict, "atom", "")
const basis_set = get(arg_dict, "basis_set", "STO")


const atom_data = parse_atom_data(atom)

# Print atom data
println("Atom: ", atom_data.atom)
println("Number of electrons (N): ", atom_data.N)
println("Number of spin-up electrons (Nup): ", atom_data.Nup)
println("Number of spin-down electrons (Ndo): ", atom_data.Ndo)
println("Atomic number (Z): ", atom_data.Z)
println("Exact energy (Eexact): ", atom_data.Eexact)

const N = atom_data.N
const Z = atom_data.Z

jpara = parse_Jastrow_data(atom, basis_set)

println("Jastrow parameters:")
@show jpara.param

#sanity check
if N>1
    if jpara.param[2] == 0.0 || jpara.param[3] == 0.0
        @warn("You have either β1 or β2 equal to 0.0")
    end
end

# keep Jastrow α=0 to not contaminate STO basis 
jpara.param[1] == 0.0
# keep Jastrow α12 = 1.0 because of cusp condition; only if there are more than one electron
if N>1
    jpara.param[4] == 1.0
else
    jpara.param[4] == 0.0
end


# Select atomic orbital functions
SlaterDeterminants.init(basis_set)

aos = parse_basis_set_data(atom, basis_set)



# Read lc coefficients and nl-combinations from file  
cnls = parse_coeffients_data(atom, basis_set)

# create linear combination of ao's
csfs = read_csf_specs( "atom_data/"*atom*"_"*basis_set)

lc_, npara_, nonlin_, lc_ups_, lc_dos_ =
    set_linear_combination_of_orbitals(
        atom_data, aos, csfs
    )

# fix lists of up and down orbitals
const lc = lc_
const lc_ups = lc_ups_
const lc_dos = lc_dos_

# sanity check
for aos in lc.orbitals
    if length(aos) != N
        @show length(aos),N
        error("lc.orbitals must have length N to accommondate all electrons")
    end
end

# more checks
for (i, det) in enumerate(lc.orbitals)
    println("\nDeterminant $i, coefficient = $(det_coeff(lc, i))")
    for ao in det
        println("  $(ao.orbital_type)  $(ao.spin)")
    end
end



# Jastrow parameters
npara_ += 4  # hard-wired ...
const npara = npara_

const nonlin = nonlin_   # If you want to set this empty, use Int64[] 

optfile = string("atom_data/"*atom*"_"*basis_set*"_opt")
println("Optimized values will be saved to $optfile")


suffix_ = string(PROGRAM_FILE,"_",atom,"_opt")

const suffix = suffix_ ::String
const Efile = string("E.",suffix)
const VMCfile = string("VMC.",suffix)


println("="^40)
@show(suffix)
println("Energy output file: ",Efile)
println("   VMC output file: ",VMCfile)
println("="^40)

rm(Efile, force=true)
rm(VMCfile, force=true)

# bind Jastrow functions
φ_T_J(x ::MMatrix{dim, N, Float64}, par) = φ_T_J_in(x, par, s)
F_J(x ::MMatrix{dim, N, Float64}, par) = F_J_in(x, par, s)


# 2 for parallel spins, 4 for antiparallel
const s = spinfactor(atom_data.Nup, atom_data.Ndo)


# collect parameters to one struct
wf_params = WfParams(lc, jpara)


println(npara," parameters in φ_T")
println(length(nonlin)," nonlinear parameters in φ_T")



@inline function E_correlated(walker ::Vector{Walker}, φ_T ::Function, wf_params_new::WfParams)
    E = 0.0        
    r = 0.0
    r2 = 0.0
    for w ∈ walker
        wratio = φ_T(w.R, wf_params_new)^2 / w.Ψ^2
        E += EL(w.R, wf_params_new) * wratio
        r += wratio
        r2 += wratio^2
    end
    Ecorr = E/r
    Neff = r^2 / r2
    return Ecorr, Neff
end



@inline function EL(R ::MMatrix{dim, N, Float64}, wf_params::WfParams)
    return EL(R, wf_params.lc, wf_params.jpara)
end

@inline function EL(R ::MMatrix{dim, N, Float64}, lc ::LinearCombination, jpara ::JastrowParameters)
    
    VL = 0.0 

    ∇DD = Matrix{Float64}(undef,dim,N)
    for j in 1:N
        for k in 1:dim
            ∇DD[k,j] = 0.0
        end
    end
    TL = 0.0
    DDsum = 0.0
    for (i,aos) in enumerate(lc.orbitals)        
        ups = aos[lc_ups[i]]
        dos = aos[lc_dos[i]]       
        # see comments before F(R)           
        DD_occu = det_coeff(lc, i) * D_up(R, ups) * D_do(R, dos)
        TL += -D* DD_occu*( ∇2_up(R, ups) + ∇2_do(R, dos) )
        ∇DD +=   DD_occu *( F_up(R, ups) + F_do(R, dos) )
        DDsum +=  DD_occu
    end
    TL = TL/DDsum
    ∇DD = ∇DD/(DDsum*2)
    
    ∇J   = F_J(R, jpara)/2     # ∇ e^J/J = ∇J = F/2
    
    TL += -D*2* ∇DD ⋅ ∇J
    
    α, β1, β2, α12 = jpara.param
    ∇2J  = 0.0
    for k in 1:N
        rk  = norm(R[:,k]) 
        ∇2J += -α*2/rk #  2 is dim-1
        VL += -Z/rk
        for i in k+1:N
            ski = s[k,i]
            rki_vec = R[:,k]-R[:,i] 
            rki = norm(rki_vec) 
            VL += 1/rki
            if ski==2
                ∇2J += 4* α12/(ski*(1+β1*rki)^3)/rki
            else
                ∇2J += 4* α12/(ski*(1+β2*rki)^3)/rki
            end
        end        
    end    
    TL += -D * ( sum(∇J.^2) + ∇2J ) 
    EL = TL + VL    
    EL    
end

#
# The e-n cusp condition is 
# c1*ζ1 + c2*ζ2 + ... = (Z-α)(c1+c2+...)
# with Jastrow factor parameter α
# For a single 1s orbital this reduces to ζ1 = Z.
# 

function force_en_cusp_condition!(wf_params ::WfParams)
    cs = []
    ζs = []
    for aos in wf_params.lc.orbitals
        done = false
        for ao in aos
            if [ao.n, ao.l] == [1, 0] 
                # this is 1s orbital
                cs = ao.coefficients
                ζs = ao.exponents
                done = true
                break
            end                
        end
        if done
            break
        end
    end
    α = wf_params.jpara.param[1]
    # sum(cs.*ζs) = (Z-α)*sum(cs), but not very accurate for many-electron systems

    #cs[end] = ((Z-α)*sum(cs[1:end-1]) - sum(cs[1:end-1].*ζs[1:end-1]) )/(ζs[end]-(Z-α))
    
    x  =  (Z-α)*sum(cs)/sum(cs.*ζs)
    println("cusp condition factor $x")
    if x < 0.0
        @warn "failed to enforce e-n cusp condition, would lead to ζ<0"
        @show  sum(cs.*ζs) - (Z-α)*sum(cs)
        return
    end
    ζs .*= x
    orbitals = []
    for aos in wf_params.lc.orbitals
        new_aolist = AtomicOrbital[]                
        for ao in aos
            if [ao.n, ao.l] == [1, 0]
                ao.exponents .= ζs
                ao.coefficients .= cs
            end
            push!(new_aolist, ao)
        end
        push!(orbitals, new_aolist)
    end
    wf_params.lc.orbitals .= orbitals
    
end

@inline function φ_T(R ::MMatrix{dim, N, Float64}, wf_params ::WfParams)
    return φ_T(R, wf_params.lc, wf_params.jpara)
end

# total Slater-Jastrow trial wavefunction

@inline function φ_T(R ::MMatrix{dim, N, Float64}, lc ::LinearCombination, jpara ::JastrowParameters) 
    DD = 0.0    
    for (i,aos) in enumerate(lc.orbitals)
        ups = aos[lc_ups[i]]
        dos = aos[lc_dos[i]]
        DD += det_coeff(lc, i) * D_up(R, ups) * D_do(R, dos)
    end
    isfinite(DD) || error("φ_T: non-finite determinant sum")
    φ_T = φ_T_J(R, jpara) * DD
end


# version with default parameters
@inline @fastmath function φ_T(R ::MMatrix{dim, N, Float64})
    return φ_T(R, wf_params)
end

function G_i_numerical(R ::MMatrix{dim, N, Float64}, wf_params ::WfParams, G :: Function)
    G0 = G(R, wf_params)
    h = 1e-6
    G_i = zeros(npara)
    Δα = @MVector zeros(npara)
    for i in 1:npara
        Δα[i] = h
        wf_params = update_wf_params(wf_params, Δα)
        G_i[i] =  (G(R, wf_params) - G0)/h
        wf_params = update_wf_params(wf_params, -Δα)
        Δα .= 0.0
    end
    G_i  
end


@inline function G_i_AD(
    R::MMatrix{dim,N,Float64},
    wf_params::WfParams,
    G::Function
) where {dim,N}

    lc = wf_params.lc

    # --------------------------------------
    # AD with respect to LC parameters
    # --------------------------------------

    function gradient_lc(lc)

        # Exclude the fixed reference CSF coefficient
        ncsf_free = length(lc.csf_coeffs) - 1
        para_concat = copy(lc.csf_coeffs[2:end])

        # Identify the unique orbital parameter sets
        keys = Tuple{Int,Int}[]
        lens = Int[]

        for aos in lc.orbitals
            for ao in aos
                key = (ao.n, ao.l)

                if key ∉ keys
                    push!(keys, key)
                    push!(lens, length(ao.exponents))

                    append!(para_concat, ao.exponents)
                    append!(para_concat, ao.coefficients)
                end
            end
        end

        gradients = ADgrad(para_concat) do x

            # Reconstruct the CSF coefficients
            csf_coeffs = vcat(
                lc.csf_coeffs[1:1],
                x[1:ncsf_free]
            )

            pos = ncsf_free + 1

            # Reconstruct unique orbital parameters
            # Shared between all occurrences of each orbital
            radial = Dict{
                Tuple{Int,Int},
                Tuple{Vector{eltype(x)},Vector{eltype(x)}}
            }()

            for (key, len) in zip(keys, lens)

                new_exponents = x[pos:pos+len-1]
                pos += len

                new_coefficients = x[pos:pos+len-1]
                pos += len

                radial[key] = (new_exponents, new_coefficients)
            end

            @assert pos == length(x) + 1

            # Reconstruct every determinant's orbitals
            orbitals = [
                [
                    AtomicOrbital(
                        ao.atom,
                        ao.orbital_type,
                        ao.n,
                        ao.l,
                        radial[(ao.n,ao.l)][1],
                        radial[(ao.n,ao.l)][2],
                        ao.spin
                    )
                    for ao in aos
                ]
                for aos in lc.orbitals
            ]

            # The CSF-to-determinant mapping stays fixed
            new_lc = LinearCombination(
                orbitals,
                csf_coeffs,
                lc.det_csf,
                lc.det_weight
            )

            return G(R, new_lc, wf_params.jpara)
        end

        return gradients
    end

    # --------------------------------------
    # AD with respect to Jastrow parameters
    # --------------------------------------

    grad_lc = gradient_lc(lc)

    grad_jpara = ADgrad(wf_params.jpara.param) do x
        new_jpara = JastrowParameters(x)
        G(R, lc, new_jpara)
    end

    # Parameter order:
    # independent CSF coefficients,
    # orbital exponents and coefficients,
    # Jastrow parameters
    return vcat(grad_lc, grad_jpara)
end

function save_parameters(outfile::String, wf_params::WfParams)

    lc = wf_params.lc

    open(outfile, "w") do f

        println(f, atom, " ", basis_set)
        println(f, "*")

        order = [
            "1s", "2s", "2p", "3s", "3p",
            "4s", "3d", "4p", "5s", "4d",
            "5p", "6s", "4f", "5d", "6p"
        ]

        # ---------------------------------
        # Orbital parameters
        # ---------------------------------

        for orbital in order
            found = false

            for aos in lc.orbitals
                for ao in aos

                    # Also matches 2px, 2py, 2pz
                    if startswith(ao.orbital_type, orbital)

                        len = length(ao.exponents)
                        mytype = orbital[end]

                        println(f, len, "  ", mytype)

                        for i in 1:len
                            @printf(f, "%15.10f %15.10f\n",
                                    ao.exponents[i],
                                    ao.coefficients[i])
                        end

                        found = true
                        break
                    end
                end

                found && break
            end
        end

        println(f, "*")

        # ---------------------------------
        # Jastrow parameters
        # ---------------------------------

        println(f, "# Jastrow params")

        α, β1, β2, α12 = wf_params.jpara.param

        @printf(f, "α = %15.10f\n", α)
        @printf(f, "β1 = %15.10f\n", β1)
        @printf(f, "β2 = %15.10f\n", β2)
        @printf(f, "α12 = %15.10f\n", α12)

        # ---------------------------------
        # Optimized CSFs
        # ---------------------------------

        write_optimized_csfs(f, csfs, lc)

        # ---------------------------------
        # Raw wave-function parameters
        # ---------------------------------

        println(f)
        println(f, "# raw output of wf_params")

        # Independent CSF coefficients
        for c in lc.csf_coeffs
            @printf(f, "%8.5f ", c)
        end

        # Unique orbital parameters
        seen = Set{Tuple{Int,Int}}()

        for aos in lc.orbitals
            for ao in aos
                key = (ao.n, ao.l)

                if key ∉ seen
                    push!(seen, key)

                    for ζ in ao.exponents
                        @printf(f, "%8.5f ", ζ)
                    end

                    for c in ao.coefficients
                        @printf(f, "%8.5f ", c)
                    end
                end
            end
        end

        # Jastrow parameters
        for p in wf_params.jpara.param
            @printf(f, "%8.5f ", p)
        end

        println(f)
    end

    return nothing
end


function thermalize(walker ::Vector{Walker}, nsteps ::Int64, vmc_params ::VMC_Params)
    # thermalize
    EL_ave = 0.0
    EL2_ave = 0.0
  
    Nw = length(walker)
    @inbounds for iw in 1:Nw 
        w = walker[iw]
        @inbounds for i in 1:nsteps
            # make sure walkers have Ψ set, or LinearOptimization fails
            w.Ψ = vmc_step!(w.R, vmc_params, (x->φ_T(x, wf_params)))
            if i%10 == 0
                adjust_step!(vmc_params)
            end
        end
        # make sure walkers have E set
        E = EL(w.R, wf_params)
        w.E = E
        EL_ave += E            
        EL2_ave += E^2   
    end        
    EL_ave /= Nw
    EL2_ave /= Nw
    EL_sigma = sqrt(abs(EL2_ave - EL_ave^2))

    return walker, EL_ave, EL_sigma 
end

function solve_Δα_wf_params!(walker ::Vector{Walker},
                             Ψ_is ::Matrix{Float64},
                             EL_is ::Matrix{Float64},
                             ELs ::Vector{Float64},                             
                             nonlin ::Vector{Int64},
                             fix_fact ::MVector,
                             a_opt ::Float64,
                             Δα ::MVector,
                             wf_params ::WfParams) 

    Nw::Int = length(walker)
    npara::Int = length(Δα)

    for attempt in 1:30
        # ************************************************************
        solve_Δα!(walker, Ψ_is, EL_is, ELs, a_opt, npara, nonlin, Δα)
        # ************************************************************

        # possibly fix some parameters
        Δα .*= fix_fact

        mα = maximum(abs.(Δα))
        if mα>1e-2
            # too large change in parameters
            a_opt *= 2.0
            continue
        end        
        wf_params_upd = update_wf_params(wf_params, Δα)
        
        #force_en_cusp_condition!(wf_params_tmp)

        # normalize 1st lc coeff to 1.0
        #norm_params!(wf_params, basis_set, Z)

        #= count nodal sign changes
        nflips = 0
        
        @inbounds for iw in 1:Nw
            R = walker[iw].R
            
            ψ_old = φ_T(R, wf_params_orig)
            ψ_new = φ_T(R, wf_params_tmp)
            
            if !isfinite(ψ_new)
                error("Non-finite trial wave function")
            end
            
            if ψ_old != 0.0 && ψ_new != 0.0 &&
                signbit(ψ_old) != signbit(ψ_new)
                nflips += 1
            end
        end
        
        if nflips > 0
            @info "Nodal changes" nflips Nw fraction=nflips/Nw
        end
        =#

    
        return a_opt, Δα, wf_params_upd
    end
end

function find_a_opt(a_opts, E_opts)

    # Sort by increasing a_opt
    p = sortperm(a_opts)
    aa = Float64.(a_opts[p])
    EE = Float64.(E_opts[p])

    all(isfinite, aa) && all(>(0), aa) ||
        error("Invalid a_opts")
    all(isfinite, EE) ||
        error("Invalid E_opts")

    length(unique(aa)) == 3 ||
        error("Need three distinct a_opts")

    imin = argmin(EE)
    a_best = aa[imin]

    # Interpolate only if the middle point
    # has lower energy than both endpoints.
    if imin != 2
        return a_best
    end

    # Work in logarithmic coordinates,
    # centered at the middle point.
    x1 = log(aa[1] / aa[2])
    x3 = log(aa[3] / aa[2])

    s1 = (EE[1] - EE[2]) / x1
    s3 = (EE[3] - EE[2]) / x3

    A = (s3 - s1) / (x3 - x1)
    B = s1 - A*x1

    if !isfinite(A) || A <= 0
        return a_best
    end

    xmin = -B / (2A)

    # Do not extrapolate outside the bracket.
    if !isfinite(xmin) || !(x1 < xmin < x3)
        return a_best
    end

    return aa[2] * exp(xmin)
end

function run_VMC!(walker ::Vector{Walker}, vmc_params ::VMC_Params) ::Tuple{Float64, Float64}
    Nw = length(walker)
    println("thermalizing $Nw walkers using VMC")
    @inbounds for iw in 1:Nw
        R = walker[iw].R
        @inbounds for i in 1:100
            vmc_step!(R, vmc_params, φ_T)
            if i%10 == 0
                adjust_step!(vmc_params)
            end
        end
    end
    println("VMC       step = $(vmc_params.step)")
    println("VMC acceptance = $(vmc_params.Naccept*100.0/vmc_params.Ntry)")

    #
    # VMC
    #
    Estat = init_stat(1, 10)
    while true
        E_ave = 0.0 
        @inbounds for iw in 1:Nw 
            R = walker[iw].R
            vmc_step!(R, vmc_params, φ_T)
            E = EL(R, wf_params) 
            E_ave += E            
        end

        add_sample!(Estat, E_ave/Nw)
        adjust_step!(vmc_params)
        if Estat.finished            
            Eb_ave, Eb_std, E_inputvar2, Nb = get_stats(Estat)            
            @printf("VMC <E> = %.10f +/- %.10f \n", Eb_ave, Eb_std)
            # exit VMC
            if (Eb_std<0.001 && Nb>5) || Nb==20
                output_MCresult(Eb_ave, Eb_std)
                open(VMCfile,"a") do f
                    @printf(f,"%.10f  %.10f \n", Eb_ave, Eb_std)
                end
                open(optfile,"a") do f
                    output_MCresult(f, Eb_ave, Eb_std)
                end
                return Eb_ave, Eb_std
            end
            
            
        end        
    end
end

function main()
    #
    # Main program 
    # 
    # 
    vmc_params = VMC_Params(0,0,2.0)

    
    init_this(Nw,Nwx) = init_walkers(Nw, Nwx, dim, N; φ_T=φ_T, scale=2.5) 
    vmc_step_this!(R) = vmc_step!(R, vmc_params, φ_T)


    # -----------------
    # Check EL 
    Nw = 5
    print_params(wf_params,"wf_params")
    walker = init_this(Nw,Nw)
    (walker, _, _) = thermalize(walker, 500, vmc_params)

    println("Checking numerically EL against φ_T")
    
    for i in 1:Nw
        R = walker[i].R
        E = EL(R, wf_params)
        num_check_EL(R, E, φ_T, x->V(x, N, Z))
    end
    println("Full EL check passed")    
    println("checks passed")
    # ---------------------
    

    #
    # Optimization
    #
    opt_logfile = "Eopt_"*atom*"_"*basis_set
    
    println("Begin Optimization, see file $opt_logfile")
    rm(opt_logfile, force=true)
    rm("Eopt_EL_"*atom*"_"*basis_set, force=true)

    
    Δα = @MVector zeros(npara)

    # Fix some parameters
    # set 0 factor for fixed parameters, 1 else
    fix_fact = @MVector ones(npara)

    
    @info "Fixing Jastrow α and α12" 
    fix_fact[end-3] = 0.0 # fix Jastrow α to prevent contaminating STO exponents
    fix_fact[end] = 0.0 # fix Jastrow α12, it's determined by the cusp condition
    
    if occursin("EMA",basis_set)
        @info "EMA, fixing nonlinear parameters" 
        # fix exponents for EMA
        fix_fact[nonlin] .= 0.0
    end

    Nw = 100 # very small set to begin with, tweak if necessary
    
    while true
        
        
        Ψ_is  = zeros(npara, Nw)
        EL_is = zeros(npara, Nw)
        ELs   = zeros(Nw)
        
        println("="^80)
        @show(Nw)
        println("="^80)
        
        walker = init_walkers(Nw, Nw, dim, N; φ_T=(x->φ_T(x, wf_params)), scale=1.5)

        
        println("-"^80)
        
        # thermalize over given number of VMC steps
        walker, EL_ave, EL_sigma = thermalize(walker, 200, vmc_params)
        
        @printf("k = %4d  Nw = %d <EL> = %.10f σ(EL) = %.10f \n", -1, Nw, EL_ave, EL_sigma)           
        a_opt = 10.0
        print_params(wf_params, "wf_params")
        
        ntries = 0
        ndownhill = 0
        EL_corr_ave, Neff = E_correlated(walker, φ_T, wf_params)
        open(opt_logfile,"a") do f
            println(f, EL_ave, " ", EL_sigma, " ", EL_corr_ave, " ", Neff)
        end       
        
        ok = true
        for k ∈ 1:10 # hardcoded, break early if correlated number of configs Neff gets too low
            println("-"^80)
            global wf_params
            
            if ok
                @info "updating gradients Ψ_is EL_is and local energies ELs using current wf_params"
                # NB: don't change walker Ψ here, it must remain the one used in sampling configs
                #     otherwise the correlated EL is wrong
                for iw ∈ 1:Nw
                    R = walker[iw].R                    
                    Ψ_is[:,iw]  = G_i_AD(R, wf_params, φ_T)              
                    EL_is[:,iw] = G_i_AD(R, wf_params, EL)
                    # for testing:
                    #Ψ_is[:,iw] = G_i_numerical(R, wf_params, φ_T)              
                    #EL_is[:,iw]= G_i_numerical(R, wf_params, EL)
                    ELs[iw] =  EL(R, wf_params)                    
                end
            end
            
            if ok
                #
                # solve wf_params and correlated E for three choices of a_opt
                # -----------------------------------------------------------
                a_opts = a_opt*[0.1, 1.0, 10.0]
                E_opts = zeros(3)
                wf_params_tmp = deepcopy(wf_params) # just to be sure orig wf_params doesn't change here
                
                for i_opt ∈ 1:3
                    a_opt_tmp = a_opts[i_opt]
                    a_opt_tmp, Δα, wf_params_upd = solve_Δα_wf_params!(walker, Ψ_is, EL_is, ELs, nonlin, fix_fact, a_opt_tmp, Δα, wf_params_tmp)
                    a_opts[i_opt] = a_opt_tmp # may have changed in solve_ in preventing too large changes
                    E_opts[i_opt], _ = E_correlated(walker, φ_T, wf_params_upd)
                end
                println("Correlated energies for three values of a_opt:")
                @show E_opts
                @show a_opts
                
                a_opt = find_a_opt(a_opts, E_opts)

                # check that this a_opt is better than any in a_opts
                a_opt, Δα, wf_params_upd = solve_Δα_wf_params!(walker, Ψ_is, EL_is, ELs, nonlin, fix_fact, a_opt, Δα, wf_params_tmp)
                EL_corr_ave_upd, Neff = E_correlated(walker, φ_T, wf_params_upd)

                if EL_corr_ave_upd > minimum(E_opts)
                    # bad guess for a_opt, revert to best in list                    
                    i = argmin(E_opts)
                    a_opt = a_opts[i]
                    EL_corr_ave_upd  = E_opts[i]
                end
                
            else
                # last one failed, so need to push a_opt higher
                a_opt *= 10.0
            end
            @show a_opt
            
            
            # solve wf_params for a_opt found above
            # -------------------------------------
            
            a_opt, Δα, wf_params_upd = solve_Δα_wf_params!(walker, Ψ_is, EL_is, ELs, nonlin, fix_fact, a_opt, Δα, wf_params)
            
            
            print_params(wf_params,"wf_params")          
            print_params(Δα,"Δα")
            print_params(wf_params_upd,"wf_params_upd")
            EL_corr_ave_upd, Neff = E_correlated(walker, φ_T, wf_params_upd)
            @show EL_corr_ave_upd, Neff, Nw
            if Neff<0.5*Nw
                @info "Correlated sampling unrealiable: effective number of configs is too low"
            end
            # -------------------------------------
            
           
            
            EL_ave_upd = 0.0
            EL2_ave_upd = 0.0           
            for w in walker
                E = EL(w.R, wf_params_upd)
                EL_ave_upd += E
                EL2_ave_upd += E^2                
            end
            EL_ave_upd /= Nw
            EL2_ave_upd /= Nw
            EL_sigma_upd = sqrt(abs(EL2_ave_upd - EL_ave_upd^2))
            #
            #
            
            ok = true           
            ntries += 1
            
            @show EL_corr_ave, EL_corr_ave_upd
            
            # decision to accept/reject based on correlated energy and reliability of correlated sampling 
            if EL_corr_ave_upd <= EL_corr_ave && Neff >= 0.5*Nw
                # BETTER
                ndownhill += 1                
                EL_ave = EL_ave_upd
                EL_sigma = EL_sigma_upd
                EL_corr_ave =  EL_corr_ave_upd
                wf_params = deepcopy(wf_params_upd)
                @info "better, new wf_params"
            else                
                # WORSE
                ok = false                
            end
            open(opt_logfile,"a") do f
                println(f, EL_ave, " ", EL_sigma, " ", EL_corr_ave)
            end
            
            
            @printf("***** Accepted  %.5f %% \n", ndownhill*100.0/ntries)
            mα = maximum(abs.(Δα))
            @printf("k = %4d  Nw = %d <EL> = %.10f σ(EL) = %.10f max Δα = %.3e  a_opt = %.6f, ok = %s\n",
                    k, Nw, EL_ave, EL_sigma, mα, a_opt, ok)

           
            
            if mα<1e-10
                break
            end
            # short thermalization; new configs and new EL_ave_old
            # may be good for small Nw
            #println("running short VMC using current wfparams")
            #walker, EL_ave_old, EL_sigma_old = thermalize(walker, 10, vmc_params)
            
        end
        # add newline 
        open(opt_logfile,"a") do f
            println(f) 
        end
        
        GC.gc() # manual carbage collection
        save_parameters(optfile, wf_params)
        # this may be omitted, takes a while:
        #if ndownhill>0
        #    Eb_av, Eb_std = run_VMC!(walker, vmc_params)
        #    if Eb_std<1e-10
        #        println("perfect trial wave function found")
        #        break
        #    end
        #end
        # Increase number of configs
        Nw = Int(floor(Nw*1.5))
        if Nw>500000 # hardwired upper limit
            break
        end
    end
    
    println("Optimization done")
    println("Results in file",optfile*"_tmp")
    println("Move it to ",optfile," if you're satisfied")
    println("Running a few VMC steps")
    run_VMC!(walker, vmc_params)
end

@time main()
