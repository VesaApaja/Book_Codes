#
# Routines for collecting measurement samples from QMC simultion
# 
# Initialization: For a single measured value,
#    stat = init_stat(1, blocksize)
# and for a measured array of ndata values,
#    stat = init_stat(ndata, blocksize) 
# When stat.finished is true, and one block is done, you can
# pick up the QMC error estimate with
#     average, std, input_σ2, Nblocks  = get_stats(stat) 
# where input_σ2 is the variance^2 of the raw input data
#

module QMC_Statistics

#using FFTW
using Statistics

export init_stat, add_sample!, get_stats, t_Stat, integrated_autocorr
export get_stats!

mutable struct t_StatData
    n    ::Int64  
    data ::Vector{Float64}
    data2 ::Vector{Float64}
    input_σ2 :: Float64
end

mutable struct t_Stat    
    nblocks   ::Int64
    blocksize ::Int64
    finished  ::Bool
    sample    ::t_StatData
    datablock :: Vector{t_StatData}
    t_Stat() = new()
end



function init_stat(datasize::Int64, blocksize::Int64; numblocks::Int64=100)
    stat = t_Stat()
    stat.blocksize = blocksize
    stat.nblocks = 0    
    stat.sample = t_StatData(0, zeros(datasize), zeros(datasize), 0.0)
    # data blocks
    stat.datablock = Vector{t_StatData}(undef, numblocks) 
    # init numblocks data blocks
    for j in 1:numblocks
        stat.datablock[j] = t_StatData(
            0,
            zeros(length(stat.sample.data)),
            zeros(length(stat.sample.data)),
            0.0
        )
    end    
    stat.finished = false
    return stat
end

function add_sample!(stat ::t_Stat, dat)
    stat.finished  = false
    @inbounds for i in eachindex(dat)
        x = dat[i]
        stat.sample.data[i]  += x
        stat.sample.data2[i] += x*x
    end
    stat.sample.n += 1
    
    if stat.sample.n == stat.blocksize
        # one full block collected, move average to block data
        # input data σ^2:
        N = stat.sample.n
        d = 0.0
        d2 = 0.0
        for i in eachindex(stat.sample.data)
            d += stat.sample.data[i]
            d2 += stat.sample.data2[i]
        end
        d /= N
        d2 /= N
        input_σ2 = d2-d^2
        #
        stat.nblocks += 1
        # resize if necessary
        if stat.nblocks > length(stat.datablock)
            old =  length(stat.datablock)
            resize!(stat.datablock, old + 100) # unintialized elements in the end
            # preallocate 
            for j in old+1:length(stat.datablock)
                stat.datablock[j] = t_StatData(
                    0,
                    zeros(length(stat.sample.data)),
                    zeros(length(stat.sample.data)),
                    0.0
                )
            end
        end
        blk = stat.datablock[stat.nblocks]
        # fill block averages        
        @inbounds for i in eachindex(stat.sample.data)
            blk.data[i] = stat.sample.data[i] / N
            blk.data2[i] = 0.0   
        end
        blk.input_σ2 = input_σ2
        # old, before preallocation:        
        #stat.datablock[stat.nblocks] = t_StatData(0, stat.sample.data./N, zeros(length(stat.sample.data)), input_σ2)
        
        @inbounds for i in eachindex(stat.sample.data)
            stat.sample.data[i]  = 0.0
            stat.sample.data2[i] = 0.0
        end
        stat.sample.n = 0
        stat.finished = true
    end  
end



function get_stats(stat ::t_Stat)
    N = stat.nblocks
    if N==0
        println("get_stats: no data")
        return 0, 0, 0, 0
    end
    if length(stat.datablock[1].data)==1
        ave_1, std_1, input_σ2_1, N_1 = get_stats_1(stat ::t_Stat)
        return ave_1, std_1, input_σ2_1, N_1,  stat.datablock[N].data[1] 
    end
    ave = similar(stat.datablock[1].data) 
    ave2 = similar(ave)
    ave .= 0
    ave2 .= 0
    input_σ2 = 0.0
    for i = 1:N
        d = stat.datablock[i].data
        @. ave += d
        @. ave2 += d^2
        input_σ2 += stat.datablock[i].input_σ2 
    end    
    @. ave /= N
    @. ave2 /= N
    input_σ2 /= N
    var = copy(ave)
    var2 = copy(ave)
    std = copy(ave)
    @. var2 = abs(ave2 - ave^2)
    @. var = sqrt(var2)
    @. std = var/sqrt(N)    
    return ave, std, input_σ2, N, stat.datablock[N].data 
end


# use buffers
const ave2_buffer = Vector{Float64}(undef, 1000) # should be enough

function get_stats!(stat::t_Stat, ave::AbstractVector{Float64}, std::AbstractVector{Float64},
                   ave2_buf::Vector{Float64} = ave2_buffer,
                   )
    N = stat.nblocks
    if N==0
        println("get_stats: no data")
        return 0, 0, 0, 0
    end
    if length(stat.datablock[1].data)==1
        a, s, input_σ2_1, N_1 = get_stats_1(stat ::t_Stat)
        ave[1] = a
        std[1] = s
        return input_σ2_1, N_1        
    end
    ave2 = @view ave2_buf[1:length(ave)]
    @inbounds @views ave .= 0
    @inbounds @views ave2 .= 0
    input_σ2 = 0.0
    
    @inbounds begin
        for i = 1:N
            d = stat.datablock[i].data
            for j in eachindex(d)
                ave[j] += d[j]
                ave2[j] += d[j]^2
            end
            input_σ2 += stat.datablock[i].input_σ2 
        end
        @views ave ./= N
        @views ave2 ./= N
        input_σ2 /= N
    end
    for i in eachindex(ave)
        var = sqrt(abs(ave2[i] - ave[i]^2))
        std[i] = var/sqrt(N)
    end
    return input_σ2, N #, stat.datablock[N].data 
end


function get_stats_1(stat ::t_Stat)
    N = stat.nblocks
    if N==0
        println("get_stats: no data")
        return nothing
    end
    ave = 0.0 
    ave2 = 0.0
    input_σ2 = 0.0 
    for i = 1:N
        d = stat.datablock[i].data[1]
        ave += d
        ave2 += d^2
        input_σ2 += stat.datablock[i].input_σ2 
    end 
    ave /= N
    ave2 /= N
    input_σ2 /= N
    var2 = abs(ave2 - ave^2)
    var = sqrt(var2) 
    std = var/sqrt(N)
    return ave, std, input_σ2, N
end


function integrated_autocorr(data::Vector{Float64}; maxlag::Int=0)
    N = length(data)
    println("QMC_Statistics  integrated_autocorr estimating τ_int using $N values")
    ave = sum(data)/N
    ave2 = sum(data.^2)/N
    σ2 = abs(ave2 - ave^2)
    if maxlag == 0
        maxlag = N ÷ 2   # default
    end
    # ρ[k] is autocovariance at lag k
    ρ = zeros(maxlag)
    for k in 1:maxlag
        ρ[k] = sum((data[1:N-k] .- ave) .* (data[1+k:N] .- ave)) / ((N - k) * σ2)
    end
    #=
    # FFT version needs packages FFTW and Statistics
    # assumes wrap-around correlations, so the result deviates for large k from the sum version
    function autocorr_fft(x)
        x = x .- mean(x)
        N = length(x)
        f = real(ifft(abs2.(fft(x))))
        f = f ./ ((N:-1:1) .* var(x)) # (N:-1:1) is [N,N-1,...,1]
        return f
    end
    ρ_fft = autocorr_fft(data)
    #    
    open("energy_autocorr","w") do f
        for (r, r_fft) in zip(ρ, ρ_fft)
            println(f, "$r  $r_fft")
        end
    end
    exit()
    =# 
    # Stop summing when ρ becomes negative
    # ρ is computed for just one sample, so it's not a smooth curve
    τ_int = 1 + 2*sum(ρ[ρ .> 0])
    return τ_int
end


# not used
function extract_block_averages(stat::t_Stat)
    """get raw data as averages over each block, for all blocks so far"""
    N = stat.nblocks
    M = length(stat.datablock[1].data)
    blocks = Matrix{Float64}(undef, M, N)  # Each column = one block average
    for i in 1:N
        blocks[:, i] .= stat.datablock[i].data
    end
    return blocks
end


end
