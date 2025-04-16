export Hyperlinear

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`Hyperlinear`.

## Fields:
$(FIELDS)
"""
struct Hyperlinear{T<:Real,U<:Real} <: Layer
    matrix::Array{T,2}
    bias::Array{U,1}
    matrix2::Array{T,2}
    bias2::Array{U,1}

    function Hyperlinear{T,U}(matrix::Array{T,2}, bias::Array{U,1}, matrix2::Array{T,2}, bias2::Array{U,1}) where {T<:Real,U<:Real}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias,matrix2, bias2)
    end

end

function Hyperlinear(matrix::Array{T,2}, bias::Array{U,1},matrix2::Array{T,2}, bias2::Array{U,1}) where {T<:Real,U<:Real}
    Hyperlinear{T,U}(matrix, bias, matrix2, bias2)
end

function Base.show(io::IO, p::Hyperlinear)
    input_size = size(p.matrix)[1]
    output_size = size(p.matrix)[2]
    print(io, "Hyperlinear($input_size -> $output_size)")
end

function check_size(params::Hyperlinear, sizes::NTuple{2,Int})::Nothing
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end],))
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`.
"""
function matmul(x::Array{<:Real,1}, params::Hyperlinear)
    return transpose(params.matrix) * x .+ params.bias
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`. We write the computation out by hand when working with `JuMPLinearType`
so that we are able to simplify the output as the computation is carried out.
"""
function matmul(x::Array{T,1}, params::Hyperlinear{U,V}) where {T<:JuMPLinearType,U<:Real,V<:Real}
    Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
    (matrix_height, matrix_width) = size(params.matrix)
    (input_height,) = size(x)
    @assert(
        matrix_height == input_height,
        "Number of values in input, $input_height, does not match number of values, $matrix_height that Linear operates on."
    )
    model = owner_model(x)
    x_low =  transpose(params.matrix) * x .+ params.bias
    x_high = transpose(params.matrix2) * x .+ params.bias2
    println(size(x_low))
    println(size(x_high))
    input_range = CartesianIndices(size(x_high))
    x_out = map(
        i -> @variable(
            model,
            lower_bound = -1000000,
            upper_bound = 1000000
        ),
        input_range,
    )
    @constraint(model,x_out .<= x_high)
    @constraint(model,x_out .>= x_low)

    return x_out
end

(p::Hyperlinear)(x::Array{<:JuMPReal}) =
    "Hyperlinear() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |>
    ArgumentError |>
    throw

(p::Hyperlinear)(x::Array{<:JuMPReal,1}) = matmul(x, p)
