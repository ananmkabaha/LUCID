export HyperlinearDeps

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`HyperlinearDeps`.

## Fields:
$(FIELDS)
"""

struct HyperlinearDeps{T<:Real,U<:Real} <: Layer
    matrix::Array{T,2}
    bias::Array{U,1}
    matrix2::Array{T,2}
    bias2::Array{U,1}
    matrixo::Array{T,2}
    biaso::Array{U,1}

    function HyperlinearDeps{T,U}(matrix::Array{T,2}, bias::Array{U,1}, matrix2::Array{T,2}, bias2::Array{U,1}, matrixo::Array{T,2}, biaso::Array{U,1}) where {T<:Real,U<:Real}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias,matrix2, bias2,matrixo, biaso)
    end

end

function HyperlinearDeps(matrix::Array{T,2}, bias::Array{U,1},matrix2::Array{T,2}, bias2::Array{U,1},matrixo::Array{T,2}, biaso::Array{U,1}) where {T<:Real,U<:Real}
    HyperlinearDeps{T,U}(matrix, bias, matrix2, bias2, matrixo, biaso)
end

function Base.show(io::IO, p::HyperlinearDeps)
    input_size = size(p.matrix)[1]
    output_size = size(p.matrix)[2]
    print(io, "HyperlinearDeps($input_size -> $output_size)")
end

function check_size(params::HyperlinearDeps, sizes::NTuple{2,Int})::Nothing
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end],))
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`.
"""
function matmul(x::Array{<:Real,1}, params::HyperlinearDeps)
    return x
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`. We write the computation out by hand when working with `JuMPLinearType`
so that we are able to simplify the output as the computation is carried out.
"""
function matmul(x::Array{T,1}, params::HyperlinearDeps{U,V}) where {T<:JuMPLinearType,U<:Real,V<:Real}
    Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
    (matrix_height, matrix_width) = size(params.matrix)
    (input_height,) = size(x)
    global layer_counter
    global upper_bound_prev
    global lower_bound_prev
	global u_for_spread
	global l_for_spread
    global diff_
    global I_z_prev_up
    global I_z_prev_down

    model = owner_model(x)
    av = JuMP.all_variables(model)

    println(" In HyperlinearDeps")
    println(layer_counter)
    model = owner_model(x)
    av = JuMP.all_variables(model)

    vec_1 = []
    for n in eachindex(av)
        if occursin("org_x_layer_"*string(layer_counter), JuMP.name(av[n]))
            append!(vec_1,n)
        end
    end
    vec_2 = []
    for n in eachindex(av)
        if occursin("hyper_x_layer_"*string(layer_counter), JuMP.name(av[n]))
            append!(vec_2,n)
        end
    end
    @constraint(model,av[vec_2] .<= av[vec_1] .+ I_z_prev_up)
    @constraint(model,av[vec_2] .>= av[vec_1] .+ I_z_prev_down)

    return x
end



(p::HyperlinearDeps)(x::Array{<:JuMPReal}) =
    "HyperlinearDeps() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |>
    ArgumentError |>
    throw

(p::HyperlinearDeps)(x::Array{<:JuMPReal,1}) = matmul(x, p)
