using Base.Cartesian

using JuMP

export HyperConv2dDeps

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
mutable struct HyperConv2dDeps{T<:JuMPReal,U<:JuMPReal,V<:Integer} <: Layer
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

    function HyperConv2dDeps{T,U,V}(
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

function HyperConv2dDeps(
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
    HyperConv2dDeps{T,U,V}(filter, bias, stride, padding,filter2, bias2, stride2, padding2,filtero, biaso, strideo, paddingo)
end

function HyperConv2dDeps(
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
    HyperConv2dDeps{T,U,V}(filter, bias, stride, SamePadding(),filter2, bias2, stride2, SamePadding(),filtero, biaso, strideo, SamePadding())
end

function HyperConv2dDeps(filter::Array{T,4}, bias::Array{U,1},filter2::Array{T,4}, bias2::Array{U,1},filtero::Array{T,4}, biaso::Array{U,1}) where {T<:JuMPReal,U<:JuMPReal}
    HyperConv2dDeps(filter, bias, 1, SamePadding(),filter2, bias2, 1, SamePadding(),filtero, biaso, 1, SamePadding())
end

"""
$(SIGNATURES)

Convenience function to create a [`Conv2d`](@ref) struct with the specified filter
and zero bias.
"""
function HyperConv2dDeps(filter::Array{T,4},filter2::Array{T,4},filtero::Array{T,4}) where {T<:JuMPReal}
    bias_out_channels::Int = size(filter)[4]
    bias = zeros(bias_out_channels)
    bias_out_channels2::Int = size(filter2)[4]
    bias2 = zeros(bias_out_channels2)
    bias_out_channelso::Int = size(filtero)[4]
    biaso = zeros(bias_out_channelso)

    HyperConv2dDeps(filter, bias,filter2, bias2,filtero,biaso)
end

function check_size(params::HyperConv2dDeps, sizes::NTuple{4,Int})::Nothing
    check_size(params.filter, sizes)
    check_size(params.bias, (sizes[end],))
end

function Base.show(io::IO, p::HyperConv2dDeps)
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





function Hyperconv2dDeps(input::Array{T,4}, params::HyperConv2dDeps{U,V}) where {T<:JuMPReal,U<:JuMPReal,V<:JuMPReal}

    if T <: JuMPLinearType || U <: JuMPLinearType || V <: JuMPLinearType
        info(MIPVerify.LOGGER, "Applying $(params) ... ")
    end


    global layer_counter
    global upper_bound_prev
    global lower_bound_prev
	global u_for_spread
	global l_for_spread
    global diff_
    global I_z_prev_up
    global I_z_prev_down

    println(" In Hyperconv2dDeps")
    model = owner_model(input)
    av = JuMP.all_variables(model)

    I_z_prev_up_to_use = I_z_prev_up |> Flatten([1, 2, 3, 4])
    I_z_prev_down_to_use = I_z_prev_down |> Flatten([1, 2, 3, 4])
    for i in 1:length(I_z_prev_up_to_use)
         var1 = []
         for n in eachindex(av)
            if occursin("org_x_layer_"*string(layer_counter)*"_neuron_"*string(i)*"_", JuMP.name(av[n]))
                var1 = av[n]
                break
            end
         end

         var2 = []
         for n in eachindex(av)
            if occursin("hyper_x_layer_"*string(layer_counter)*"_neuron_"*string(i)*"_", JuMP.name(av[n]))
                var2 = av[n]
                break
            end
         end

        if !isempty(var1) && !isempty(var2)
            @constraint(model,var2 <= var1 + I_z_prev_up_to_use[i] )
            @constraint(model,var2 >= var1 + I_z_prev_down_to_use[i])
        end

    end


    return input
end

(p::HyperConv2dDeps)(x::Array{<:JuMPReal,4}) = Hyperconv2dDeps(x, p)
