# Blocking statistics for scalar or vector QMC measurements.
#
# Usage:
#   stat = init_stat(1, 100)           # scalar observations
#   stat = init_stat(3, 100)           # vector observations of length 3
#   add_sample!(stat, observation)
#   if stat.finished
#       average, std_err, input_σ2, Nblocks, lastblock = get_stats(stat)
#   end
#
# std_err is the standard error calculated from completed block means,
# assuming blocks are sufficiently long to be approximately uncorrelated.
# input_σ2 is the mean WITHIN-BLOCK population variance of the raw
# observations, calculated separately for each vector component.
# Incomplete blocks are not included in get_stats.

module QMC_Statistics

export init_stat, add_sample!, get_stats, t_Stat

mutable struct t_StatData
    n::Int
    data::Vector{Float64}       # running mean during sampling; completed block mean otherwise
    data2::Vector{Float64}      # running sum of squared deviations (Welford's M2)
    input_σ2::Vector{Float64}   # within-block variance of each component
end

mutable struct t_Stat
    nblocks::Int
    blocksize::Int
    finished::Bool
    sample::t_StatData
    datablock::Vector{t_StatData}
end

function init_stat(datasize::Integer, blocksize::Integer; numblocks::Integer=100)
    datasize > 0 || throw(ArgumentError("datasize must be positive"))
    blocksize > 0 || throw(ArgumentError("blocksize must be positive"))
    numblocks >= 0 || throw(ArgumentError("numblocks must be nonnegative"))

    sample = t_StatData(0, zeros(datasize), zeros(datasize), zeros(datasize))
    datablock = Vector{t_StatData}(undef, numblocks)
    return t_Stat(0, Int(blocksize), false, sample, datablock)
end

# Add one scalar observation. The scalar interface is deliberately restricted
# to datasize == 1 (avoids accidentally broadcasting a scalar to all channels).
function add_sample!(stat::t_Stat, dat::Real)
    length(stat.sample.data) == 1 ||
        throw(DimensionMismatch("scalar observation requires datasize == 1"))
    return _add_sample!(stat, (Float64(dat),))
end

# Add one vector observation; no temporary vector is allocated.
function add_sample!(stat::t_Stat, dat::AbstractVector{<:Real})
    length(dat) == length(stat.sample.data) ||
        throw(DimensionMismatch("observation length must match datasize"))
    return _add_sample!(stat, dat)
end

function _add_sample!(stat::t_Stat, dat)
    s = stat.sample
    n = s.n + 1

    # Welford's numerically stable online mean and M2 update.
    @inbounds for j in eachindex(s.data)
        x = Float64(dat[j])
        delta = x - s.data[j]
        s.data[j] += delta / n
        s.data2[j] += delta * (x - s.data[j])
    end
    s.n = n
    stat.finished = false

    if n == stat.blocksize
        stat.nblocks += 1
        if stat.nblocks > length(stat.datablock)
            resize!(stat.datablock, length(stat.datablock) + 100)
        end

        # Population variance (M2/n), matching the original input_σ2 meaning.
        input_var = similar(s.data2)
        @inbounds for j in eachindex(input_var)
            input_var[j] = max(0.0, s.data2[j] / n)
        end
        stat.datablock[stat.nblocks] = t_StatData(0, copy(s.data),
                                                 Float64[], input_var)

        fill!(s.data, 0.0)
        fill!(s.data2, 0.0)
        s.n = 0
        stat.finished = true
    end
    return nothing
end

function get_stats(stat::t_Stat)
    N = stat.nblocks
    m = length(stat.sample.data)

    # Always return five values; preserve scalar vs vector shape.
    if N == 0
        if m == 1
            return NaN, NaN, NaN, 0, NaN
        else
            return fill(NaN, m), fill(NaN, m), fill(NaN, m), 0, fill(NaN, m)
        end
    end

    ave = zeros(m)
    input_var = zeros(m)
    for i in 1:N
        b = stat.datablock[i]
        @inbounds for j in 1:m
            ave[j] += b.data[j]
            input_var[j] += b.input_σ2[j]
        end
    end
    ave ./= N
    input_var ./= N

    # Stable second pass over block means.
    std_err = fill(NaN, m) # with only one block, the error is undetermined
    if N > 1
        ss = zeros(m)
        for i in 1:N
            d = stat.datablock[i].data
            @inbounds for j in 1:m
                ss[j] += (d[j] - ave[j])^2
            end
        end
        @inbounds for j in 1:m
            std_err[j] = sqrt(ss[j] / (N * (N - 1)))
        end
    end

    lastblock = stat.datablock[N].data
    if m == 1
        return ave[1], std_err[1], input_var[1], N, lastblock[1]
    end
    return ave, std_err, input_var, N, copy(lastblock)
end

end # module QMC_Statistics
