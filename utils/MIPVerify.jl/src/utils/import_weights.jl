export get_matrix_params, get_hyper_network_conv_deps_params, get_hyper_network_conv_deps_paramsIn, get_conv_params, get_example_network_params,get_hyper_network_params,get_hyper_network_deps_params,get_hyper_network_deps_paramsIn, get_hyper_network_conv_params

"""
$(SIGNATURES)

Helper function to import the parameters for a layer carrying out matrix multiplication
    (e.g. fully connected layer / softmax layer) from `param_dict` as a
    [`Linear`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{2, Int}`: Tuple of length 2 corresponding to the expected size
   of the weights of the layer.

"""
function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
)::Linear

    params = Linear(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
    )

    check_size(params, expected_size)

    return params
end

"""
$(SIGNATURES)

Helper function to import the parameters for a convolution layer from `param_dict` as a
    [`Conv2d`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{4, Int}`: Tuple of length 4 corresponding to the expected size
    of the weights of the layer.

"""
function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = ValidPadding(),
)::Conv2d

    params = Conv2d(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
    )

    check_size(params, expected_size)

    return params
end



function get_hyper_network_params(
    param_dict::Dict{String},
    param_dict2::Dict{String},
    layer_name::String,
    expected_size::NTuple{2,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
)::Hyperlinear

    params = Hyperlinear(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        param_dict2["$layer_name/$matrix_name"],
        dropdims(param_dict2["$layer_name/$bias_name"], dims = 1),
    )

    check_size(params, expected_size)

    return params
end

function get_hyper_network_conv_params(
    param_dict::Dict{String},
    param_dict2::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = ValidPadding(),
)::HyperConv2d

    params = HyperConv2d(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
        param_dict2["$layer_name/$matrix_name"],
        dropdims(param_dict2["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
    )

    check_size(params, expected_size)

    return params
end

function get_hyper_network_conv_deps_params(
    param_dict::Dict{String},
    param_dict2::Dict{String},
    param_dicto::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = ValidPadding(),
)::HyperConv2dDeps

    params = HyperConv2dDeps(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
        param_dict2["$layer_name/$matrix_name"],
        dropdims(param_dict2["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
        param_dicto["$layer_name/$matrix_name"],
        dropdims(param_dicto["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,

    )

    check_size(params, expected_size)

    return params
end

function get_hyper_network_conv_deps_paramsIn(
    param_dict::Dict{String},
    param_dict2::Dict{String},
    param_dicto::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = ValidPadding(),
)::HyperConv2dDepsIn

    params = HyperConv2dDepsIn(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
        param_dict2["$layer_name/$matrix_name"],
        dropdims(param_dict2["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
        param_dicto["$layer_name/$matrix_name"],
        dropdims(param_dicto["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,

    )

    check_size(params, expected_size)

    return params
end

function get_hyper_network_deps_params(
    param_dict::Dict{String},
    param_dict2::Dict{String},
    param_dicto::Dict{String},
    layer_name::String,
    expected_size::NTuple{2,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
)::HyperlinearDeps

    params = HyperlinearDeps(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        param_dict2["$layer_name/$matrix_name"],
        dropdims(param_dict2["$layer_name/$bias_name"], dims = 1),
        param_dicto["$layer_name/$matrix_name"],
        dropdims(param_dicto["$layer_name/$bias_name"], dims = 1),
    )

    #check_size(params, expected_size)

    return params
end

function get_hyper_network_deps_paramsIn(
    param_dict::Dict{String},
    param_dict2::Dict{String},
    param_dicto::Dict{String},
    layer_name::String,
    expected_size::NTuple{2,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
)::HyperlinearDepsIn

    params = HyperlinearDepsIn(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        param_dict2["$layer_name/$matrix_name"],
        dropdims(param_dict2["$layer_name/$bias_name"], dims = 1),
        param_dicto["$layer_name/$matrix_name"],
        dropdims(param_dicto["$layer_name/$bias_name"], dims = 1),
    )

    #check_size(params, expected_size)

    return params
end