global network_version
global upper_bound_prev
global lower_bound_prev
global u_for_spread
global l_for_spread
global me_th
global diff_ = []
global all_bounds_of_original = []
global I_z_prev_up = []
global I_z_prev_down = []

mutable struct ReuseBoundAndDepsConfig
    is_reuse_bounds_and_deps::Bool
    reusable_indexes::Int
    reusable_bounds::Vector{Float64}
    reusable_deps::Vector{Any}
end
reuse_bounds_conf = ReuseBoundAndDepsConfig(false, 1, [],Any[])

mutable struct NeuronsAssignNames
    neuron::Int
    layer::Int
end
neurons_names = NeuronsAssignNames(0, 0)

mutable struct FirstMIPSolution
    solution::Float64
    time::Float64
end
first_mip_solution = FirstMIPSolution(-1.0, 0.0)

layers_info_dict = Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}}()

mutable struct Results
    str::String
end
results = Results("")

@pyimport pickle
function mypickle(filename, obj)
    out = open(filename,"w")
    pickle.dump(obj, out)
    close(out)
 end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

function compute_acc(mnist, nn, is_conv,w_,h_,k_)
    num_correct = 0.0
    num_samples_ = 10000
    num_samples_ = min(num_samples_, num_samples(mnist.test))
    for sample_index in 1:num_samples_
        input = MIPVerify.get_image(mnist.test.images, sample_index)
        actual_label = MIPVerify.get_label(mnist.test.labels, sample_index)
        if is_conv
            predicted_label = (reshape(np.transpose(input),(1,w_,h_,k_))|> nn |> MIPVerify.get_max_index) - 1
        else
            predicted_label = ((input)|> nn |> MIPVerify.get_max_index) - 1
        end
        if actual_label == predicted_label
             num_correct += 1
        end
    end
    println("Model accuracy: " * string(num_correct / num_samples_))
end

function read_best_val_via_optimization(ss, tt, token_signature)
    file = open("/tmp/best_val_" * string(ss-1) * "_" * string(tt-1) * "_" * string(token_signature) * ".txt")
    line = readline(file)
    close(file)
    value = parse(Float64, line)
    return value
end

function save_results(results_path, model_name, results_str, d, nn, ss, tt, w_, h_, k_)

    file = open(results_path * model_name *".txt", "w")
    write(file, results_str)
    close(file)
    try
        sample = d[:v_in]

        if k_ == 1
            sample = reshape(sample, w_, h_)
        else
            sample = reshape(sample, w_, h_,k_)
        end
        sample_reshaped = reshape(sample, 1, w_, h_,k_)
        sample_reshaped_result = argmax(sample_reshaped |> nn)
        println("Clean sample classification: ", sample_reshaped_result)
        matshow(sample, vmin=0, vmax=1)
        mypickle(results_path*string(ss)*"_"*string(tt)*"_org.p", sample)
        savefig(results_path*string(ss)*"_"*string(tt)*"_org.png")

        perturbed_sample = (d[:v_in_p])
        if k_ == 1
            perturbed_sample = reshape(perturbed_sample, w_, h_)
        else
            perturbed_sample = reshape(perturbed_sample, w_, h_,k_)
        end
        perturbed_sample_reshaped = reshape(perturbed_sample, 1, w_, h_,k_)
        perturbed_sample_reshaped_result = argmax(perturbed_sample_reshaped |> nn)
        println("Perturbed sample classification: ",perturbed_sample_reshaped_result)
        matshow(perturbed_sample, vmin=0, vmax=1)
        mypickle(results_path*string(ss)*"_"*string(tt)*"_perturbed.p", perturbed_sample)
        savefig(results_path*string(ss)*"_"*string(tt)*"_perturbed.png")
   catch e
        println("no results")
   end
end

function create_perturbation_string(perturbation_size)
    perturbation_size_string = ""
    for i in eachindex(perturbation_size)
        perturbation_size_string *= string(perturbation_size[i])
        if i <length(perturbation_size)
           perturbation_size_string *=","
        end
    end
    return perturbation_size_string
end

function get_default_tightening_options(optimizer)::Dict
    optimizer_type_name = string(typeof(optimizer()))
    if optimizer_type_name == "Gurobi.Optimizer"
        return Dict("OutputFlag" => 0, "TimeLimit" => 0.5)
    elseif optimizer_type_name == "Cbc.Optimizer"
        return Dict("logLevel" => 0, "seconds" => 20)
    else
        return Dict()
    end
end

function my_callback(cb_data::Gurobi.CallbackData, where::Int32)
    if where == GRB_CB_MIPSOL
        resultP = Ref{Float64}()
        GRBcbget(cb_data, where, GRB_CB_MIPSOL_OBJ, resultP)
        run_time =Ref{Float64}()
        GRBcbget(cb_data, where, GRB_CB_RUNTIME, run_time)
        if first_mip_solution.solution == -1
            first_mip_solution.solution = resultP[]
            first_mip_solution.time = run_time[]
        end
    end
end

function parse_numbers_to_Float64(input_str::String)
    str_numbers = rsplit(input_str, ",")
    numbers = Float64[]
    for str_num in str_numbers
        push!(numbers, parse(Float64, str_num))
    end
    return numbers
end

function parse_numbers_to_Int64(input_str::String)
    str_numbers = rsplit(input_str, ",")
    numbers = Int64[]
    for str_num in str_numbers
        push!(numbers, parse(Float64, str_num))
    end
    return numbers
end

function update_results_str(results, c_tag, c_target, d)
    return results*string(c_tag-1)*","*string(c_target-1)*","*string(d[:incumbent_obj])*","*
        string(d[:best_bound])*","*string(d[:solve_time])*"\n"
end

function check_if_to_start(wi, token)
        println("Worker "*string(wi)*" of token "*string(token)*" is waiting for a task.")
        flush(stdout)
        while isfile("/tmp/start"*string(wi)*"_"*string(token)*".txt") == false
            sleep(0.01)
            if isfile("/tmp/exit_"*string(token)*".txt")
                break
            end
        end
        println("Worker "*string(wi)*" of token "*string(token)*" is processing a task.")
        flush(stdout)
end

function check_if_to_finish(token)
    if isfile("/tmp/exit_"*string(token)*".txt")
        return true
    end
    return false
end

function worker_log(wi, token, d)

    rm("/tmp/start"*string(wi)*"_"*string(token)*".txt")
    file = open("/tmp/results"*string(wi)*"_"*string(token)*".txt", "w")
    write(file, string(d[:best_bound]))
    close(file)
    file = open("/tmp/finished"*string(wi)*"_"*string(token)*".txt", "w")
    write(file, "")
    close(file)
end