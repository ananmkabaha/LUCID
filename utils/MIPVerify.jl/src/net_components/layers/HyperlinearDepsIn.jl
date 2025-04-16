export HyperlinearDepsIn

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`HyperlinearDepsIn`.

## Fields:
$(FIELDS)
"""

struct HyperlinearDepsIn{T<:Real,U<:Real} <: Layer
    matrix::Array{T,2}
    bias::Array{U,1}
    matrix2::Array{T,2}
    bias2::Array{U,1}
    matrixo::Array{T,2}
    biaso::Array{U,1}

    function HyperlinearDepsIn{T,U}(matrix::Array{T,2}, bias::Array{U,1}, matrix2::Array{T,2}, bias2::Array{U,1}, matrixo::Array{T,2}, biaso::Array{U,1}) where {T<:Real,U<:Real}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias,matrix2, bias2,matrixo, biaso)
    end

end

function HyperlinearDepsIn(matrix::Array{T,2}, bias::Array{U,1},matrix2::Array{T,2}, bias2::Array{U,1},matrixo::Array{T,2}, biaso::Array{U,1}) where {T<:Real,U<:Real}
    HyperlinearDepsIn{T,U}(matrix, bias, matrix2, bias2, matrixo, biaso)
end

function Base.show(io::IO, p::HyperlinearDepsIn)
    input_size = size(p.matrix)[1]
    output_size = size(p.matrix)[2]
    print(io, "HyperlinearDepsIn($input_size -> $output_size)")
end

function check_size(params::HyperlinearDepsIn, sizes::NTuple{2,Int})::Nothing
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end],))
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`.
"""
function matmul(x::Array{<:Real,1}, params::HyperlinearDepsIn)
    return x
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`. We write the computation out by hand when working with `JuMPLinearType`
so that we are able to simplify the output as the computation is carried out.
"""


function interval_matrix_vector_multiplication(w_h_low, w_h_high, I_z_prev_min, I_z_prev_max)
    n = size(w_h_low, 1)  # Number of rows in w_h matrices
    m = size(w_h_low, 2)  # Number of columns in w_h matrices (and length of I_z_prev vectors)

    # Initialize the result vectors for the min and max bounds
    result_min = zeros(n)
    result_max = zeros(n)

    # Compute the matrix-vector product for the interval
    for i in 1:n
        min_val = 0.0
        max_val = 0.0
        for j in 1:m
            prod1 = w_h_low[i, j] * I_z_prev_min[j]
            prod2 = w_h_low[i, j] * I_z_prev_max[j]
            prod3 = w_h_high[i, j] * I_z_prev_min[j]
            prod4 = w_h_high[i, j] * I_z_prev_max[j]

            min_val += minimum([prod1, prod2, prod3, prod4])
            max_val += maximum([prod1, prod2, prod3, prod4])
        end
        result_min[i] = min_val
        result_max[i] = max_val
    end

    return result_min, result_max
end

#=
function matmul(x::Array{T,1}, params::HyperlinearDepsIn{U,V}) where {T<:JuMPLinearType,U<:Real,V<:Real}
    Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
    (matrix_height, matrix_width) = size(params.matrix)
    (input_height,) = size(x)
    global layer_counter
    global all_bounds_of_original
    global I_z_prev_up
    global I_z_prev_down
    z_prev_up   = all_bounds_of_original[layer_counter+1][1]
    z_prev_down = all_bounds_of_original[layer_counter+1][2]
    I_w_up   =  transpose(params.matrix2)-transpose(params.matrixo)
    I_w_down =  transpose(params.matrix)-transpose(params.matrixo)
    w_h_up   = transpose(params.matrixo)
    w_h_down = transpose(params.matrixo)
    w1 = transpose(params.matrix)
    result_min_1, result_max_1 = interval_matrix_vector_multiplication(I_w_down, I_w_up, z_prev_down, z_prev_up)
    result_min_2, result_max_2 = interval_matrix_vector_multiplication(w1, w1, I_z_prev_down, I_z_prev_up)
    I_z_down_in = result_min_1 .+ result_min_2
    I_z_up_in = result_max_1 .+ result_max_2
    #ReLU:
    I_z_m_up   = max.(0.0,I_z_up_in)
    I_z_m_down = -max.(0.0,-I_z_down_in)
    #next
    I_z_prev_up = I_z_m_up
    I_z_prev_down = I_z_m_down

    return x
end
=#

function matmul(x::Array{T,1}, params::HyperlinearDepsIn{U,V}) where {T<:JuMPLinearType,U<:Real,V<:Real}
    Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
    (matrix_height, matrix_width) = size(params.matrix)
    (input_height,) = size(x)
    global layer_counter
    global upper_bound_prev
    global lower_bound_prev
	global u_for_spread
	global l_for_spread
    global diff_
    global all_bounds_of_original
    global I_z_prev_up
    global I_z_prev_down

    println(" In HyperlinearDepsIn")

    model = owner_model(x)
    av = JuMP.all_variables(model)

    z_prev_up   = all_bounds_of_original[layer_counter+1][1]
    z_prev_down = all_bounds_of_original[layer_counter+1][2]
    I_w_up   =  transpose(params.matrix2)-transpose(params.matrixo)
    I_w_down =  transpose(params.matrix)-transpose(params.matrixo)
    w_h_up   = transpose(params.matrixo)
    w_h_down = transpose(params.matrixo)
    w1 = transpose(params.matrix)
    result_min_1, result_max_1 = interval_matrix_vector_multiplication(I_w_down, I_w_up, z_prev_down, z_prev_up)
    result_min_2, result_max_2 = interval_matrix_vector_multiplication(w1, w1, I_z_prev_down, I_z_prev_up)
    I_z_down_in = result_min_1 .+ result_min_2
    I_z_up_in = result_max_1 .+ result_max_2
    #ReLU:
    I_z_m_up   = max.(0.0,I_z_up_in)
    I_z_m_down = -max.(0.0,-I_z_down_in)
    #next
    I_z_prev_up = I_z_m_up
    I_z_prev_down = I_z_m_down
    diff_ = I_z_prev_up-I_z_prev_down

    return x
end

(p::HyperlinearDepsIn)(x::Array{<:JuMPReal}) =
    "HyperlinearDepsIn() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |>
    ArgumentError |>
    throw

(p::HyperlinearDepsIn)(x::Array{<:JuMPReal,1}) = matmul(x, p)
