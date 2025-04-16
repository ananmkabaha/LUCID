ENV["PYTHON"]="/usr/bin/python3.8"

using Gurobi
using PyCall
using PyPlot
using Gurobi
using Images
using Printf
using Dates
using Base.Cartesian
using JuMP
using MathOptInterface
using Memento
using DocStringExtensions
using ProgressMeter
using ArgParse



np = pyimport("numpy")

include("MIPVerify.jl/src/MIPVerify.jl")
const dependencies_path = joinpath(@__DIR__, "MIPVerify.jl/", "deps")
@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3
const DEFAULT_TIGHTENING_ALGORITHM = mip

include("MIPVerify.jl/src/vendor/ConditionalJuMP.jl")
include("MIPVerify.jl/src/net_components.jl")
include("MIPVerify.jl/src/logging.jl")
include("MIPVerify.jl/src/models.jl")
include("MIPVerify.jl/src/utils.jl")
include("perturbation_models.jl")
include("help_functions.jl")
include("datasets.jl")
include("models.jl")
include("mip.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "twitter, crypto, adult, or credit"
        arg_type = String
        required = false
        default = "crypto"
        "--model_name", "-n"
        help = "2x10, 2x50, 2x100, 4x30, or CNN"
        arg_type = String
        required = false
        default = "2x50"
        "--model_path", "-m"
        help = "model path"
        arg_type = String
        required = false
        default = "./model/crypto.p"
        "--hypers_dir_path"
        help = "hypers directory path"
        arg_type = String
        required = false
        default = "/tmp/"
        "--ctag", "-c"
        help = "source class"
        arg_type = Int
        required = false
        default = 1
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2"
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 2400
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "./results/"
        "--token"
        help = "token for unique runs"
        arg_type = String
        required = false
        default = "0000000"
        "--worker"
        help = "worker ID"
        arg_type = Int
        required = false
        default = 1
        "--deps"
        help = "is dependencies?"
        arg_type = Int
        required = false
        default = 1
        "--me"
        help = "relax if similar threshold"
        arg_type = Float64
        required = false
        default = 0.01
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true

    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = args["model_name"]
    model_path = args["model_path"]
    hypers_dir_path = args["hypers_dir_path"]
    c_tag = args["ctag"]
    c_targets = parse_numbers_to_Int64(args["ct"])
    results_path = args["output_dir"]
    timout = args["timout"]
    is_deps = args["deps"]
    worker_id = args["worker"]
    token = args["token"]
    global me_th

    me_th = args["me"]

    dim, c = get_dataset_params( dataset )
    nn,is_conv = get_nn(model_path, model_name, dim, c, dataset)

    global network_version
    global upper_bound_prev
    global lower_bound_prev
    global u_for_spread
    global l_for_spread
    global diff_

    total_time = @elapsed begin
    is_first_run = true
    while true
        check_if_to_start(worker_id, token)
        if check_if_to_finish(token)
            break
        end
        nn_hyper = get_nn_hyper(model_path, model_name, dim, c, dataset, hypers_dir_path, is_deps, worker_id, token)
        for c_target in c_targets
            optimizer = Gurobi.Optimizer
            d = Dict()
            d[:TargetIndex] = get_target_indexes(c_target, c)
            d[:SourceIndex] = get_target_indexes(c_tag, c)
            mip_reset()
            println("Run: computing bounds.")
            dummy_input = zeros(Float64, 1,1,1,dim)
            if is_conv
               dummy_input = zeros(Float64, 1,dim,1,1)
            end
            bounds_time = @elapsed begin
                merge!(d, get_model(nn, nn_hyper, dummy_input, optimizer,
                 get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]
            mip_set_delta_property(m, d)
            set_optimizer(m, optimizer)
            mip_set_attr(m, d, timout, (worker_id==1 && is_first_run))
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)
            println("Run: optimize.")
            optimize!(m)
            mip_log(m, d)
            results.str = update_results_str(results.str, c_tag, c_target, d)
            worker_log(worker_id, token, d)
            upper_bound_prev = []
            lower_bound_prev = []
            u_for_spread = []
            l_for_spread = []
            diff_  = []
            is_first_run = false
            println("--------- Results ---------")
            println("c:"*string(c_tag-1)*" ,t:"*string(c_target-1)*" ,bound:"*string(d[:best_bound])*" ,solve time:"*string(d[:solve_time]))
        end
    end
    end
    println("Total time:"*string(total_time))
    println("---------------------------")
end

main()
