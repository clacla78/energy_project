using CSV
using DataFrames
using Statistics
using CairoMakie
using Agents
using Printf
using Dates

include("types.jl")

module BaselineModel
    using Agents, CSV, DataFrames, Statistics
    import ..ModelParams, ..Household
    include("model.jl")
    adoption_rate_fn(model) = (count(a -> a.pv_installed, allagents(model)) / nagents(model)) * 100
end

module PolicyModel
    using Agents, CSV, DataFrames, Statistics
    import ..ModelParams, ..Household
    include("model_policy.jl")
    adoption_rate_fn(model) = (count(a -> a.pv_installed, allagents(model)) / nagents(model)) * 100
end

function run_simulation_variant(row, arch_idx::Int, n_months::Int, use_policy::Bool)::Float64
    i_att_val = (hasproperty(row, :i_att) && !ismissing(row.i_att) && !isnan(row.i_att)) ? Float64(row.i_att) : 0.3
    p = ModelParams(
        decision_architecture_idx = arch_idx,
        w_eco              = Float64(row.w_eco),
        w_env              = Float64(row.w_env),
        w_soc              = Float64(row.w_soc),
        adoption_threshold = Float64(row.adoption_threshold),
        activation_prob    = Float64(row.activation_prob),
        i_att              = i_att_val,
    )
    
    if use_policy
        model = PolicyModel.initialize_model(p)
        _, mdf = PolicyModel.run!(model, n_months; mdata = [PolicyModel.adoption_rate_fn], when = 12, showprogress = false)
        return last(mdf.adoption_rate_fn)
    else
        model = BaselineModel.initialize_model(p)
        _, mdf = BaselineModel.run!(model, n_months; mdata = [BaselineModel.adoption_rate_fn], when = 12, showprogress = false)
        return last(mdf.adoption_rate_fn)
    end
end

function display_rr_comprehensive_table(df_rr)
    println("\n" * "═"^115)
    println("  RR 2050 PROJECTIONS & CALIBRATED PARAMETER PROFILES")
    println("═"^115)
    @printf("| %-8s | %-6s | %-6s | %-6s | %-9s | %-8s | %-13s | %-13s | %-8s |\n", 
            "Run ID", "RMSE", "w_eco", "w_env", "adopt_th", "act_prob", "Baseline (%)", "Policy (%)", "Change")
    println("|" * "-"^113 * "|")
    for r in eachrow(df_rr)
        delta = r.With_Policy - r.No_Policy
        @printf("| #%-6d | %-6.3f | %-6.2f | %-6.2f | %-9.2f | %-8.3f | %11.2f %% | %11.2f %% | %+7.2f %% |\n", 
                r.Run_ID, r.RMSE, r.w_eco, r.w_env, r.adoption_threshold, r.activation_prob, r.No_Policy, r.With_Policy, delta)
    end
    println("═"^115)
end

function display_se_comprehensive_table(df_se)
    println("\n" * "═"^128)
    println("          SE 2050 PROJECTIONS & CALIBRATED PARAMETER PROFILES")
    println("═"^128)
    @printf("| %-8s | %-6s | %-6s | %-6s | %-6s | %-6s | %-9s | %-8s | %-13s | %-13s | %-8s |\n", 
            "Run ID", "RMSE", "w_eco", "w_env", "w_soc", "i_att", "adopt_th", "act_prob", "Baseline (%)", "Policy (%)", "Change")
    println("|" * "-"^126 * "|")
    for r in eachrow(df_se)
        delta = r.With_Policy - r.No_Policy
        @printf("| #%-6d | %-6.3f | %-6.2f | %-6.2f | %-6.2f | %-6.2f | %-9.2f | %-8.3f | %11.2f %% | %11.2f %% | %+7.2f %% |\n", 
                r.Run_ID, r.RMSE, r.w_eco, r.w_env, r.w_soc, r.i_att, r.adoption_threshold, r.activation_prob, r.No_Policy, r.With_Policy, delta)
    end
    println("═"^128)
end

function main()
    CALIB_CSV    = "C:\\Users\\clari\\Documents\\MA3\\Project energy St Prex\\julia_clarisse\\calibration_results_total.csv"
    PROJ_MONTHS  = 492   

    df_res = CSV.read(CALIB_CSV, DataFrame)
    df_rr  = filter(:architecture => ==("RR"), df_res)
    df_se  = filter(:architecture => ==("SE"), df_res)

    df_rr_filtered = df_rr[df_rr.rmse .< 3.0, :]
    df_se_filtered = df_se[df_se.rmse .< 3.0, :]

    println("Processing $(nrow(df_rr_filtered)) valid RR configurations and $(nrow(df_se_filtered)) valid SE configurations...")

    rr_rows = []
    println("\n⏳ Executing twin trajectories for valid RR runs...")
    for row in eachrow(df_rr_filtered)
        rate_baseline = run_simulation_variant(row, 0, PROJ_MONTHS, false)
        rate_policy   = run_simulation_variant(row, 0, PROJ_MONTHS, true)
        push!(rr_rows, (
            Run_ID = row.run_id,
            RMSE = row.rmse,
            w_eco = row.w_eco,
            w_env = row.w_env,
            adoption_threshold = row.adoption_threshold,
            activation_prob = row.activation_prob,
            No_Policy = rate_baseline,
            With_Policy = rate_policy
        ))
    end
    df_rr_results = DataFrame(rr_rows)
    sort!(df_rr_results, :RMSE)

    se_rows = []
    println("Executing twin trajectories for valid SE runs...")
    for row in eachrow(df_se_filtered)
        rate_baseline = run_simulation_variant(row, 1, PROJ_MONTHS, false)
        rate_policy   = run_simulation_variant(row, 1, PROJ_MONTHS, true)
        i_att_val = hasproperty(row, :i_att) ? row.i_att : 0.3
        push!(se_rows, (
            Run_ID = row.run_id,
            RMSE = row.rmse,
            w_eco = row.w_eco,
            w_env = row.w_env,
            w_soc = row.w_soc,
            i_att = i_att_val,
            adoption_threshold = row.adoption_threshold,
            activation_prob = row.activation_prob,
            No_Policy = rate_baseline,
            With_Policy = rate_policy
        ))
    end
    df_se_results = DataFrame(se_rows)
    sort!(df_se_results, :RMSE)

    display_rr_and_policy_tables = true
    if display_rr_and_policy_tables
        display_rr_comprehensive_table(df_rr_results)
        display_se_comprehensive_table(df_se_results)
    end

    CSV.write("comprehensive_projections_rr.csv", df_rr_results)
    CSV.write("comprehensive_projections_se.csv", df_se_results)
    println("\nSaved separate detailed datasets to disk.")

    # Calculate unweighted means for runs tracking under an baseline RMSE < 3.0 threshold
    mean_with_policy_rr_30 = nrow(df_rr_results) > 0 ? mean(df_rr_results.With_Policy) : NaN
    mean_with_policy_se_30 = nrow(df_se_results) > 0 ? mean(df_se_results.With_Policy) : NaN

    println("\n" * "═"^75)
    println("          POLICY ENSEMBLE MEAN PERFORMANCE (2050)")
    println("═"^75)
    if !isnan(mean_with_policy_rr_30)
        @printf("  • RR Architecture (RMSE < 3.0) | Policy Scenario Mean: %6.2f %%\n", mean_with_policy_rr_30)
    else
        println("  • RR Architecture (RMSE < 3.0) | No valid calibration configurations available")
    end
    if !isnan(mean_with_policy_se_30)
        @printf("  • SE Architecture (RMSE < 3.0) | Policy Scenario Mean: %6.2f %%\n", mean_with_policy_se_30)
    else
        println("  • SE Architecture (RMSE < 3.0) | No valid calibration configurations available")
    end
    println("═"^75)
end

main()