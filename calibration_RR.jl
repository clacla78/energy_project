using BlackBoxOptim
using Agents
using CSV
using DataFrames
using Statistics
using Plots


df_real = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
years = 2009:2024

real_counts = [count(row -> !ismissing(row.BeginningOfOperation_Year) && 
                            row.BeginningOfOperation_Year <= y, eachrow(df_real)) for y in years]

#  (MSE - Mean Squared Error)
function cost_rr(X::Vector{Float64})::Float64
    w_e = X[1]
    w_n = X[2]
    w_s = 1.0 - (w_e + w_n)
    if w_s < 0; return 1e9; end 

    params = ModelParams(
        decision_architecture_idx = 0, # FORCE LE MODE RR
        w_eco = w_e,
        w_env = w_n,
        w_soc = w_s,
        adoption_threshold = X[3],
        activation_prob = X[4]
    )
    
    model = initialize_model(params)
    _, mdf = run!(model, 192; mdata = [adoption_rate], when = 12, showprogress = false)
    
    sim_counts = (mdf.adoption_rate ./ 100) .* nagents(model)

    len = min(length(sim_counts), length(real_counts))
    mse = sum((sim_counts[1:len] - real_counts[1:len]).^2) / len
    
    return mse
end

opt_res = bboptimize(
    cost_rr; 
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    SearchRange = [
        (0.2, 1.0),     # w_eco
        (0.2, 1.0),     # w_env
        (0.6, 0.9),     # adoption_threshold
        (0.001, 0.05)   # activation_prob
    ],
    NumDimensions = 4,
    MaxSteps = 100,     
    TraceMode = :verbose
)


best_params = best_candidate(opt_res)
println("\n--- Calibration RR done ---")
println("Best fit :")
println("w_eco : ", best_params[1])
println("w_env : ", best_params[2])
println("w_soc : ", 1.0 - (best_params[1] + best_params[2]))
println("Seuil  : ", best_params[3])
println("Prob   : ", best_params[4])