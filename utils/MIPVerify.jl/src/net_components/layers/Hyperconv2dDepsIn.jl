using Base.Cartesian

using JuMP

export HyperConv2dDepsIn

struct SamePadding end
Base.show(io::IO, p::SamePadding) = print(io, "same")
struct ValidPadding end
Base.show(io::IO, p::ValidPadding) = print(io, "valid")

FixedPadding = Union{Int,Tuple{Int,Int},Tuple{Int,Int,Int,Int}}
Padding = Union{SamePadding,ValidPadding,FixedPadding}

"""
$(TYPEDEF)

Represents 2-D convolution operation.

`p(x)` is shorthand for [`conv2d(x, p)`](@ref) when `p` is an instance of
`Conv2d`.

## Fields:
$(FIELDS)
"""
mutable struct HyperConv2dDepsIn{T<:JuMPReal,U<:JuMPReal,V<:Integer} <: Layer
    filter::Array{T,4}
    bias::Array{U,1}
    stride::V
    padding::Padding
    filter2::Array{T,4}
    bias2::Array{U,1}
    stride2::V
    padding2::Padding
    filtero::Array{T,4}
    biaso::Array{U,1}
    strideo::V
    paddingo::Padding

    function HyperConv2dDepsIn{T,U,V}(
        filter::Array{T,4},
        bias::Array{U,1},
        stride::V,
        padding::Padding,

        filter2::Array{T,4},
        bias2::Array{U,1},
        stride2::V,
        padding2::Padding,

        filtero::Array{T,4},
        biaso::Array{U,1},
        strideo::V,
        paddingo::Padding,

    ) where {T<:JuMPReal,U<:JuMPReal,V<:Integer}
        (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)
        bias_out_channels = length(bias)
        @assert(
            filter_out_channels == bias_out_channels,
            "For this convolution layer, number of output channels in filter, $filter_out_channels, does not match number of output channels in bias, $bias_out_channels."
        )
        return new(filter, bias, stride, padding, filter2, bias2, stride2, padding2, filtero, biaso, strideo, paddingo)
    end

end

function HyperConv2dDepsIn(
    filter::Array{T,4},
    bias::Array{U,1},
    stride::V,
    padding::Padding,

    filter2::Array{T,4},
    bias2::Array{U,1},
    stride2::V,
    padding2::Padding,

    filtero::Array{T,4},
    biaso::Array{U,1},
    strideo::V,
    paddingo::Padding,

) where {T<:JuMPReal,U<:JuMPReal,V<:Integer}
    HyperConv2dDepsIn{T,U,V}(filter, bias, stride, padding,filter2, bias2, stride2, padding2,filtero, biaso, strideo, paddingo)
end

function HyperConv2dDepsIn(
    filter::Array{T,4},
    bias::Array{U,1},
    stride::V,

    filter2::Array{T,4},
    bias2::Array{U,1},
    stride2::V,

    filtero::Array{T,4},
    biaso::Array{U,1},
    strideo::V,

) where {T<:JuMPReal,U<:JuMPReal,V<:Integer}
    HyperConv2dDepsIn{T,U,V}(filter, bias, stride, SamePadding(),filter2, bias2, stride2, SamePadding(),filtero, biaso, strideo, SamePadding())
end

function HyperConv2dDepsIn(filter::Array{T,4}, bias::Array{U,1},filter2::Array{T,4}, bias2::Array{U,1},filtero::Array{T,4}, biaso::Array{U,1}) where {T<:JuMPReal,U<:JuMPReal}
    HyperConv2dDepsIn(filter, bias, 1, SamePadding(),filter2, bias2, 1, SamePadding(),filtero, biaso, 1, SamePadding())
end

"""
$(SIGNATURES)

Convenience function to create a [`Conv2d`](@ref) struct with the specified filter
and zero bias.
"""
function HyperConv2dDepsIn(filter::Array{T,4},filter2::Array{T,4},filtero::Array{T,4}) where {T<:JuMPReal}
    bias_out_channels::Int = size(filter)[4]
    bias = zeros(bias_out_channels)
    bias_out_channels2::Int = size(filter2)[4]
    bias2 = zeros(bias_out_channels2)
    bias_out_channelso::Int = size(filtero)[4]
    biaso = zeros(bias_out_channelso)

    HyperConv2dDepsIn(filter, bias,filter2, bias2,filtero,biaso)
end

function check_size(params::HyperConv2dDepsIn, sizes::NTuple{4,Int})::Nothing
    check_size(params.filter, sizes)
    check_size(params.bias, (sizes[end],))
end

function Base.show(io::IO, p::HyperConv2dDepsIn)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(p.filter)
    stride = p.stride
    padding = p.padding
    print(
        io,
        "Conv2d($filter_in_channels, $filter_out_channels, kernel_size=($(filter_height), $(filter_width)), stride=($(stride), $(stride)), padding=$(padding))",
    )
end

# TODO (vtjeng): Figure out how to actually mutate the underlying value of s
# OR avoid all this confusion
function add_to_expression!(s::Real, input_val::Real, filter_val::Real)
    return s + input_val * filter_val
end

function add_to_expression!(s::JuMP.GenericAffExpr, input_val, filter_val)
    return JuMP.add_to_expression!(s, input_val, filter_val)
end

function add_to_expression_min!(s::Real, input_val_min::Real, filter_val_min::Real, input_val_max::Real, filter_val_max::Real)
    return s + min(input_val_min * filter_val_min, input_val_min * filter_val_max, input_val_max * filter_val_min, input_val_max * filter_val_max)
end

function add_to_expression_max!(s::Real, input_val_min::Real, filter_val_min::Real, input_val_max::Real, filter_val_max::Real)
    return s + max(input_val_min * filter_val_min, input_val_min * filter_val_max, input_val_max * filter_val_min, input_val_max * filter_val_max)
end

function compute_output_parameters(
    in_height::Int,
    in_width::Int,
    filter_height::Int,
    filter_width::Int,
    stride::Int,
    padding::FixedPadding,
)::Tuple{NTuple{2,Int},NTuple{2,Int}}
    (top_padding, bottom_padding, left_padding, right_padding) = compute_padding_values(padding)
    out_height_raw = (in_height + top_padding + bottom_padding - filter_height) / stride
    out_height = round(Int, out_height_raw, RoundDown) + 1
    out_width_raw = (in_width + left_padding + right_padding - filter_width) / stride
    out_width = round(Int, out_width_raw, RoundDown) + 1

    output_size = (out_height, out_width)
    filter_offset = (top_padding, left_padding)
    return (output_size, filter_offset)
end

function compute_output_parameters(
    in_height::Int,
    in_width::Int,
    filter_height::Int,
    filter_width::Int,
    stride::Int,
    padding::SamePadding,
)::Tuple{NTuple{2,Int},NTuple{2,Int}}
    out_height = round(Int, in_height / stride, RoundUp)
    out_width = round(Int, in_width / stride, RoundUp)
    pad_along_height = max((out_height - 1) * stride + filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + filter_width - in_width, 0)
    filter_height_offset = round(Int, pad_along_height / 2, RoundDown)
    filter_width_offset = round(Int, pad_along_width / 2, RoundDown)

    output_size = (out_height, out_width)
    filter_offset = (filter_height_offset, filter_width_offset)
    return (output_size, filter_offset)
end

function compute_output_parameters(
    in_height::Int,
    in_width::Int,
    filter_height::Int,
    filter_width::Int,
    stride::Int,
    padding::ValidPadding,
)::Tuple{NTuple{2,Int},NTuple{2,Int}}
    out_height = round(Int, (in_height + 1 - filter_height) / stride, RoundUp)
    out_width = round(Int, (in_width + 1 - filter_width) / stride, RoundUp)
    return ((out_height, out_width), (0, 0))
end

function compute_padding_values(padding::Int)::NTuple{4,Int}
    return (padding, padding, padding, padding)
end

function compute_padding_values(padding::NTuple{2,Int})::NTuple{4,Int}
    (y_padding, x_padding) = padding
    return (y_padding, y_padding, x_padding, x_padding)
end

function compute_padding_values(padding::NTuple{4,Int})::NTuple{4,Int}
    return padding
end

"""
$(SIGNATURES)

Computes the result of convolving `input` with the `filter` and `bias` stored in `params`.

Mirrors `tf.nn.conv2d` from the `tensorflow` package, with
`strides = [1, params.stride, params.stride, 1]`.

Supports three types of padding:
- 'same':  Specify via `SamePadding()`. Padding is added so that the output has the same size as the input.
- 'valid': Specify via `FixedPadding()`. No padding is added.
- 'fixed': Specify via:
  - A single integer, interpreted as padding for both axes
  - A tuple of two integers, interpreted as (y_padding, x_padding)
  - A tuple of four integers, interpreted as (top, bottom, left, right)

# Throws
* AssertionError if `input` and `filter` are not compatible.
"""
function compute_conv(input_vec, filter_vec, bias_vec, output_size, stride, filter_height_offset, filter_width_offset)

    output_vec = zeros(output_size)
    @nloops 4 i output_vec begin
        (@nref 4 output_vec i) = bias_vec[i_4]
        @nloops 3 j filter_vec begin
            x = (i_2 - 1) * stride + j_1 - filter_height_offset
            y = (i_3 - 1) * stride + j_2 - filter_width_offset
            input_index = (i_1, x, y, j_3)
            if checkbounds(Bool, input_vec, input_index...)
                # Effectively zero-padding the input.
                (@nref 4 output_vec i) = add_to_expression!(
                    (@nref 4 output_vec i),
                    input_vec[input_index...],
                    filter_vec[j_1, j_2, j_3, i_4],
                )
            end
        end
    end
    return output_vec

end


function compute_conv_min(input_vec_min,input_vec_max, filter_vec_min, filter_vec_max, bias_vec, output_size, stride, filter_height_offset, filter_width_offset)

    output_vec = zeros(output_size)
    @nloops 4 i output_vec begin
        (@nref 4 output_vec i) = bias_vec[i_4]
        @nloops 3 j filter_vec_max begin
            x = (i_2 - 1) * stride + j_1 - filter_height_offset
            y = (i_3 - 1) * stride + j_2 - filter_width_offset
            input_index = (i_1, x, y, j_3)
            if checkbounds(Bool, input_vec_min, input_index...)
                # Effectively zero-padding the input.
                (@nref 4 output_vec i) = add_to_expression_min!(
                    (@nref 4 output_vec i),
                    input_vec_min[input_index...],
                    filter_vec_min[j_1, j_2, j_3, i_4],
                    input_vec_max[input_index...],
                    filter_vec_max[j_1, j_2, j_3, i_4],
                )
            end
        end
    end
    return output_vec

end

function compute_conv_max(input_vec_min,input_vec_max, filter_vec_min, filter_vec_max, bias_vec, output_size, stride, filter_height_offset, filter_width_offset)

    output_vec = zeros(output_size)
    @nloops 4 i output_vec begin
        (@nref 4 output_vec i) = bias_vec[i_4]
        @nloops 3 j filter_vec_max begin
            x = (i_2 - 1) * stride + j_1 - filter_height_offset
            y = (i_3 - 1) * stride + j_2 - filter_width_offset
            input_index = (i_1, x, y, j_3)
            if checkbounds(Bool, input_vec_min, input_index...)
                # Effectively zero-padding the input.
                (@nref 4 output_vec i) = add_to_expression_max!(
                    (@nref 4 output_vec i),
                    input_vec_min[input_index...],
                    filter_vec_min[j_1, j_2, j_3, i_4],
                    input_vec_max[input_index...],
                    filter_vec_max[j_1, j_2, j_3, i_4],
                )
            end
        end
    end
    return output_vec

end


function compute_conv_min_max(input_vec_min, input_vec_max, filter_vec_min, filter_vec_max, bias_vec, output_size, stride, filter_height_offset, filter_width_offset)

    output_vec_min = zeros(output_size)
    output_vec_max = zeros(output_size)

    for out_idx in 1:output_size[1], out_jdx in 1:output_size[2]
        min_val = bias_vec[out_idx, out_jdx]
        max_val = bias_vec[out_idx, out_jdx]

        for f_h in 1:filter_height_offset, f_w in 1:filter_width_offset
            in_i = (out_idx - 1) * stride[1] + f_h
            in_j = (out_jdx - 1) * stride[2] + f_w

            # Compute all possible combinations of input and filter intervals
            products = [
                input_vec_min[in_i, in_j] * filter_vec_min[f_h, f_w],
                input_vec_min[in_i, in_j] * filter_vec_max[f_h, f_w],
                input_vec_max[in_i, in_j] * filter_vec_min[f_h, f_w],
                input_vec_max[in_i, in_j] * filter_vec_max[f_h, f_w]
            ]

            # The minimum value will be the smallest product, and the maximum value will be the largest product
            min_val += minimum(products)
            max_val += maximum(products)
        end

        output_vec_min[out_idx, out_jdx] = min_val
        output_vec_max[out_idx, out_jdx] = max_val
    end

    return output_vec_min, output_vec_max
end

function Hyperconv2dDepsIn(input::Array{T,4}, params::HyperConv2dDepsIn{U,V}) where {T<:JuMPReal,U<:JuMPReal,V<:JuMPReal}

    if T <: JuMPLinearType || U <: JuMPLinearType || V <: JuMPLinearType
        info(MIPVerify.LOGGER, "Applying $(params) ... ")
    end

    global layer_counter
    global upper_bound_prev
    global lower_bound_prev
	global u_for_spread
	global l_for_spread
    global diff_
    global all_bounds_of_original
    global I_z_prev_up
    global I_z_prev_down

    z_prev_up   = all_bounds_of_original[layer_counter+1][1]
    z_prev_down = all_bounds_of_original[layer_counter+1][2]

    println("In Conv2dDepsIn.")

    I_w_filter_up = params.filter2-params.filtero
    I_w_filter_down = params.filter-params.filtero
    w_filter_1 = params.filter2


    filter = params.filter
    bias_vec = params.biaso
    stride = params.stride
    padding = params.padding
    (batch, in_height, in_width, input_in_channels) = size(z_prev_up)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)

    @assert(
        input_in_channels == filter_in_channels,
        "Number of channels in input, $input_in_channels, does not match number of channels, $filter_in_channels, that filters operate on."
    )
    # Considered using offset arrays here, but could not get it working.
    ((out_height, out_width), (filter_height_offset, filter_width_offset)) =
        compute_output_parameters(in_height, in_width, filter_height, filter_width, stride, padding)
    output_size = (batch, out_height, out_width, filter_out_channels)

    result_min_1 = compute_conv_min(z_prev_down,z_prev_up, I_w_filter_down, I_w_filter_up, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)
    result_max_1 = compute_conv_max(z_prev_down,z_prev_up, I_w_filter_down, I_w_filter_up, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)
    result_min_2 = compute_conv_min(I_z_prev_down,I_z_prev_up, w_filter_1, w_filter_1, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)
    result_max_2 = compute_conv_max(I_z_prev_down,I_z_prev_up, w_filter_1, w_filter_1, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)

    I_z_down_in = result_min_1 .+ result_min_2
    I_z_up_in = result_max_1 .+ result_max_2

    #ReLU:
    I_z_m_up   = max.(0.0,I_z_up_in)
    I_z_m_down = -max.(0.0,-I_z_down_in)

    #Next
    I_z_prev_up = I_z_m_up
    I_z_prev_down = I_z_m_down
    println(size(I_z_prev_up))
    println(size(input))
    #println("I_z_prev_up:", I_z_prev_up|> Flatten([1, 2, 3, 4]))
    #println("I_z_prev_down:", I_z_prev_down|> Flatten([1, 2, 3, 4]))

    diff_ = I_z_prev_up-I_z_m_down

    return input
end


#=
function Conv2dDepsIn(input::Array{T,4}, params::Conv2dDepsIn{U,V}) where {T<:JuMPReal,U<:JuMPReal,V<:JuMPReal}

    if T <: JuMPLinearType || U <: JuMPLinearType || V <: JuMPLinearType
        info(MIPVerify.LOGGER, "Applying $(params) ... ")
    end

    global layer_counter
    global all_bounds_of_original
    global I_z_prev_up
    global I_z_prev_down

    z_prev_up   = all_bounds_of_original[layer_counter+1][1]
    z_prev_down = all_bounds_of_original[layer_counter+1][2]

    println("In Conv2dDepsIn.")

    I_w_filter_up = params.filter2-params.filtero
    I_w_filter_down = params.filter-params.filtero
    w_filter_1 = params.filter2


    filter = params.filter
    bias_vec = params.biaso
    stride = params.stride
    padding = params.padding
    (batch, in_height, in_width, input_in_channels) = size(z_prev_up)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)

    @assert(
        input_in_channels == filter_in_channels,
        "Number of channels in input, $input_in_channels, does not match number of channels, $filter_in_channels, that filters operate on."
    )
    # Considered using offset arrays here, but could not get it working.
    ((out_height, out_width), (filter_height_offset, filter_width_offset)) =
        compute_output_parameters(in_height, in_width, filter_height, filter_width, stride, padding)
    output_size = (batch, out_height, out_width, filter_out_channels)

    result_min_1 = compute_conv_min(z_prev_down,z_prev_up, I_w_filter_down, I_w_filter_up, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)
    result_max_1 = compute_conv_max(z_prev_down,z_prev_up, I_w_filter_down, I_w_filter_up, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)

    result_min_2 = compute_conv_min(I_z_prev_down,I_z_prev_up, w_filter_1, w_filter_1, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)
    result_max_2 = compute_conv_max(I_z_prev_down,I_z_prev_up, w_filter_1, w_filter_1, bias_vec*0, output_size, stride, filter_height_offset, filter_width_offset)

    I_z_down_in = result_min_1 .+ result_min_2
    I_z_up_in = result_max_1 .+ result_max_2

    #ReLU:
    I_z_m_up   = max.(0.0,I_z_up_in)
    I_z_m_down = -max.(0.0,-I_z_down_in)

    #Next
    I_z_prev_up = I_z_m_up
    I_z_prev_down = I_z_m_down
    println(size(I_z_prev_up))
    println(size(input))
    #println("I_z_prev_up:", I_z_prev_up|> Flatten([1, 2, 3, 4]))
    #println("I_z_prev_down:", I_z_prev_down|> Flatten([1, 2, 3, 4]))

    return input
end
=#



(p::HyperConv2dDepsIn)(x::Array{<:JuMPReal,4}) = Hyperconv2dDepsIn(x, p)
