function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
end

function mip_set_delta_property(m, d)
    set_max_indexes(m, d[:v_out_p], d[:TargetIndex])
    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(d[:v_out], d[:SourceIndex])
    maximum_nontarget_var = maximum_ge(nontarget_vars)
    delta = @variable(m)
    @constraint(m, delta == maximum_target_var - maximum_nontarget_var)
    @objective(m, Max, delta)
end

function mip_set_attr(m, d, timout, quick_anytime_bound)
    set_optimizer_attribute(m, "MIPFocus", 3)
    set_optimizer_attribute(m, "Cutoff", 0)
    set_optimizer_attribute(m, "Threads", 8)
    set_optimizer_attribute(m, "TimeLimit", timout)
    set_optimizer_attribute(m, "MIPGap", 0.1)
    if quick_anytime_bound
        set_optimizer_attribute(m, "BestBdStop", 10^9)
    end
end

function mip_log(m, d)
    d[:SolveStatus] = JuMP.termination_status(m)
    d[:SolveTime] = JuMP.solve_time(m)
    incumbent_obj = 0
    try
        incumbent_obj = JuMP.objective_value(m)
    catch e
        println("no incumbent_obj")
    end
    d[:incumbent_obj] = incumbent_obj
    d[:best_bound] = JuMP.objective_bound(m)
    d[:solve_time] = JuMP.solve_time(m)
    d[:first_mip_solution] = first_mip_solution.solution
    d[:time_for_first_mip_solution] = first_mip_solution.time
    println(string(incumbent_obj)*"  "*string(d[:best_bound])*"  "*string(d[:solve_time]))
    try
        d[:v_in_p] = (JuMP.value.(d[:v_in_p]))
        d[:v_in] = (JuMP.value.(d[:v_in]))
        if d[:Perturbation] != "None"
            d[:Perturbation] = (JuMP.value.(d[:Perturbation]))
        end
    catch e
        d[:v_in_p] = 0
        d[:v_in] = 0
        d[:Perturbation] = 0
    end
end

function mip_reuse_bounds()
    reuse_bounds_conf.is_reuse_bounds_and_deps = true
    reuse_bounds_conf.reusable_indexes = 1
end
