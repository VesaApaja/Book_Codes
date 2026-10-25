module STOOrbitals
using StaticArrays
using ForwardDiff
#import LinearAlgebra: norm

using Common
using GSL  # polynomials

# slightly faster norm 
@inline norm(x) = sqrt(sum(abs2,x))

#=
Here Slater-type atomic orbitals (STO) are
ϕ_j(rvec) = r^{n-1} exp(-ζ_j r) S_{lm}(θ,ϕ) , 


 1s   n=0, l=0, m=0 
 2s   n=1, l=0, m=0
 2p   n=2, l=1, m=-1,0,1
 3s   n=3, l=0, m=0
 3p   n=3, l=1, m=-1,0,1   
in filling order 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p


The first ones are
s:
Y_{00}(θ,ϕ)     = 1/2*sqrt(1/pi)

p:
Y_{1,-1}(θ,ϕ)   = 1/2*sqrt(3/(2pi)) sin(θ) e^{-iϕ}
Y_{10}(θ,ϕ)     = 1/2*sqrt(3/pi) cos(θ) 
Y_{11}(θ,ϕ)     = -1/2*sqrt(3/(2pi)) sin(θ) e^{iϕ}

d:
Y_{2,-2}(θ,ϕ)   = 1/4*sqrt(15/(2pi)) sin^2(θ) e^{-2iϕ}
Y_{2,-1}(θ,ϕ)   = 1/2*sqrt(15/(2pi)) sin(θ) cos(θ) e^{-iϕ}
Y_{20}(θ,ϕ)     = 1/4*sqrt(5/pi) (3 cos^2(θ)-1)
Y_{21}(θ,ϕ)     = -1/2*sqrt(15/(2pi)) sin(θ)cos(θ) e^{iϕ}
Y_{2,-2}(θ,ϕ)    = 1/4*sqrt(15/(2pi)) sin^2(θ) e^{2iϕ}



The real spherical harmonics  S_{lm} (θ,ϕ) are

                  sqrt(2) (-1)^m Re Y_{l|m|}(θ,ϕ) , for m>0
S_{lm}(θ,ϕ) =     Y_{l0}(θ,ϕ)            , for m=0
                  sqrt(2) (-1)^m Im Y_{lm}(θ,ϕ) , for m<0


s:
S_{00}(θ,ϕ)     = 1/2*sqrt(1/pi)                  ϕ_1(rvec) = c_1 exp(-ζ_1 r) 1/2*sqrt(1/pi)

p: (actually px and py are linear combinations of m=1 and m=-1 states, but I arbitrarily assign one m=-1 to px and m=1 to py) 
S_{1,-1}(θ,ϕ)   = 1/2*sqrt(3/pi) sin(θ)sin(ϕ)   =>  2py : r*S_{1,-1}(θ,ϕ) = 1/2*sqrt(3/pi)*y  
S_{10}(θ,ϕ)     = 1/2*sqrt(3/pi) cos(θ)         =>  2pz : r*S_{10}(θ,ϕ)   = 1/2*sqrt(3/pi)*z   
S_{11}(θ,ϕ)     = 1/2*sqrt(3/pi) sin(θ)cos(ϕ)   =>  2px : r*S_{11}(θ,ϕ)   = 1/2*sqrt(3/pi)*x  

d:
S_{2,-2}(θ,ϕ)   = 1/4*sqrt(15/pi) sin^2(θ)sin(2ϕ) 
S_{2,-1}(θ,ϕ)   = 1/4*sqrt(15/pi) sin(2θ)sin(ϕ)
S_{2,0}(θ,ϕ)    = 1/4*sqrt(5/pi) (3cos^2(θ)-1)
S_{2,1}(θ,ϕ)    = 1/4*sqrt(15/pi) sin(2θ)cos(ϕ)
S_{2,2}(θ,ϕ)    = 1/4*sqrt(15/pi) sin(θ)cos(2ϕ)

Atom states are eigenstate of S^2 and S_z, so for p-states etc. take linear combinations,
1s: ϕ_1(rvec) = c_1 * exp(-ζ_1 r) 
2s: ϕ_2(rvec) = c_2 * exp(-ζ_2 r) 
2p: ϕ_3(rvec) = c_3 * exp(-ζ_3 r) r (S_{1,-1}+S_{10}+S_{11})(θ,ϕ)
          = c_3 * exp(-ζ_3 r) * (x+y+z)
ϕ_4(rvec) = c_4 * exp(-ζ_4 r) r^2 (S_{2,-2}+S_{2-1}+S_{20}+S_{21}+S_{22})(θ,ϕ)

=#




# R_nl(r) = N r^{n-1} exp(-ζr)
#  and multiply with S_lm(θ,ϕ)

#The real spherical harmonics S_{lm} (θ,ϕ) are 
#                  sqrt(2) (-1)^m Re Y_{l|m|}(θ,ϕ) , for m>0
#S_{lm}(θ,ϕ) =     Y_{l0}(θ,ϕ)            , for m=0
#                  sqrt(2) (-1)^m Im Y_{lm}(θ,ϕ) , for m<0


#using SphericalHarmonics
#Y = computeYlm(pi/2, pi/3, lmax = 1, m_range = SphericalHarmonics.ZeroTo, SHType = SphericalHarmonics.RealHarmonics())

# Spherical harmonics
# Condon–Shortley phase (-1.0)^m 
# use abs(m) in Legendre polynomials

#=
function Y(l ::Int64, m ::Int64, θ ::T, ϕ ::Z) where T<:Number where  Z<:Number
    if m==0
        Ylm = sf_legendre_Plm(l,0,cos(θ))
    elseif m>0
        Ylm = (-1.0)^m*sf_legendre_Plm(l,abs(m),cos(θ))* exp(im*m*ϕ)
    else
        Ylm = (-1.0)^m*sf_legendre_Plm(l,abs(m),cos(θ))* (-1.0)* exp(-im*m*ϕ)
    end
    return Ylm    
end



@inline function S(l ::Int64, m ::Int64, θ ::T, ϕ ::Z) where T<:Number where  Z<:Number
    if l==0
        S = 1/2*sqrt(1/pi)
    elseif l==1
        if m==-1
            # 2py
            S = 1/2*sqrt(3/pi)*sin(θ)*sin(ϕ)
        elseif m==0
            # 2pz
            S = 1/2*sqrt(3/pi)*cos(θ)
        elseif m==1
            # 2px
            S = 1/2*sqrt(3/pi)*sin(θ)*cos(ϕ)
        else
            error("illegal m")
        end
    elseif l==2
        if m==-2
            S = 1/4*sqrt(15/pi)*sin(θ)^2*sin(2ϕ)
        elseif m==-1
            S =  1/4*sqrt(15/pi)*sin(2θ)*sin(ϕ)
        elseif m==0
            S = 1/4*sqrt(5/pi)*(3cos(θ)^2-1)
        elseif m==1
            S = 1/4*sqrt(15/pi)*sin(2θ)*cos(ϕ)
        elseif m==2
            S = 1/4*sqrt(15/pi)*sin(θ)*cos(2ϕ)
        else
            error("illegal m")
        end
    else
        error("orb l>2 not implemented")
    end
    #=
    if m>0
        S = sqrt(2.0) * (-1)^m * real(Y(l,m,θ,ϕ))
    elseif m==0
        S = Y(l,m,θ,ϕ)
    else
        S = sqrt(2.0) * (-1)^m * imag(Y(l,m,θ,ϕ))
    end
    =#
    return S
end
=#


# normalization of radial STO
@inline function norm_STO(n, l, ζ)
    z32 = ζ * sqrt(ζ)

    if n == 1 && l == 0       # 1s
        return z32 / sqrt(π)
    elseif n == 2 && l == 0   # 2s
        return ζ*z32 / sqrt(3π)
    elseif n == 2 && l == 1   # 2p
        return ζ*z32 / sqrt(π)
    else
        error("STO normalization not implemented")
    end
end


# this was slightly off, n+1 -> 2n+1
# doesn't really matter that much, optimization changes the overall coefficient to comply
# @inline norm_STO(n,ζ) = sqrt.( (2*ζ).^(n+1)/factorial(2n) )

# Orbitals
# ========

@inline function orb(rvec::MVector, ao::AtomicOrbital)
    r = norm(rvec)

    n = ao.n
    l = ao.l
    ζ = ao.exponents
    c = ao.coefficients
    t = ao.orbital_type

    @inbounds radial =
        c[1] * norm_STO(n, l, ζ[1]) * exp(-ζ[1]*r)
    
    @inbounds for k in 2:length(ζ)
        radial += c[k] * norm_STO(n, l, ζ[k]) * exp(-ζ[k]*r)
    end
    
    # Angular/radial prefactor
    factor = if t == "1s"
        one(r)
    elseif t == "2s"
        r
    elseif t == "2px"
        rvec[1]
    elseif t == "2py"
        rvec[2]
    elseif t == "2pz"
        rvec[3]
    else
        error("Unknown orbital type: $t")
    end

    return factor * radial
end


@inline function ∇orb(rvec::MVector{3}, ao::AtomicOrbital)

    r = norm(rvec)
    x, y, z = rvec

    n = ao.n
    ζ = ao.exponents
    c = ao.coefficients

    denom = factorial(2n)

    # First primitive; establishes numeric type,
    # including ForwardDiff.Dual when appropriate.
    @inbounds begin
        ζk = ζ[1]
        a = c[1] * norm_STO(n, ao.l, ζk) * exp(-ζk*r)
        A = a
        B = ζk*a

        for k in 2:length(ζ)
            ζk = ζ[k]
            a = c[k] * norm_STO(n, ao.l, ζk) * exp(-ζk*r)
            A += a
            B += ζk*a
        end
    end

    t = ao.orbital_type
    q = B/r

    if t == "1s"
        g = -q
        return SVector(g*x, g*y, g*z)

    elseif t == "2s"
        g = A/r - B
        return SVector(g*x, g*y, g*z)

    elseif t == "2px"
        g = -x*q
        return SVector(A + g*x, g*y, g*z)

    elseif t == "2py"
        g = -y*q
        return SVector(g*x, A + g*y, g*z)

    elseif t == "2pz"
        g = -z*q
        return SVector(g*x, g*y, A + g*z)

    else
        error("Unknown orbital type: $t")
    end
end



# ∇^2 R_{nl}(r) =  (n(n-1)/r^2 - 2nζ/r + ζ^2) R_{nl}(r)
@inline function ∇2orb(rvec::MVector{3}, ao::AtomicOrbital)

    r = norm(rvec)

    n = ao.n
    ζ = ao.exponents
    c = ao.coefficients

    denom = factorial(2n)

    # Accumulate A, B and C.
    # First primitive establishes the numeric type,
    # including ForwardDiff.Dual when applicable.
    @inbounds begin
        ζk = ζ[1]        
        a = c[1] * norm_STO(n, ao.l, ζk) * exp(-ζk*r)

        A = a
        B = ζk*a
        C = ζk*ζk*a

        for k in 2:length(ζ)
            ζk = ζ[k]
            a = c[k] * norm_STO(n, ao.l, ζk) * exp(-ζk*r)

            A += a
            B += ζk*a
            C += ζk*ζk*a
        end
    end

    invr = inv(r)

    t = ao.orbital_type

    if t == "1s"
        return C - 2*B*invr

    elseif t == "2s"
        return r*C - 4*B + 2*A*invr

    elseif t == "2px"
        return rvec[1] * (C - 4*B*invr)

    elseif t == "2py"
        return rvec[2] * (C - 4*B*invr)

    elseif t == "2pz"
        return rvec[3] * (C - 4*B*invr)

    else
        error("Unknown orbital type: $t")
    end
end



end
