using BlackBoxOptim
using Agents
using CSV
using DataFrames
using Statistics

df_real = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
years = 2009:2024
real_counts = [count(row -> row.BeginningOfOperation_Year <= y, eachrow(df_real)) for y in years]

# 2. Fonction de coût spécifique au modèle RR
function cost_rr(X::Vector{Float64})::Float64
    w_e = X[1]
    w_n = X[2]
    w_s = 1.0 - (w_e + w_n)
    
    if w_s < 0; return 1e9; end 

    params = ModelParams(
        decision_architecture_idx = 0, # FORCÉ EN RR
        w_eco = w_e,
        w_env = w_n,
        w_soc = w_s,
        adoption_threshold = X[3]
    )
    
    model = initialize_model(params)
    
    mdf = run!(model, 192; mdata = [adoption_rate], when = 12)
    sim_counts = (mdf.adoption_rate ./ 100) .* nagents(model)
    return sum((sim_counts - real_counts).^2) / length(real_counts)
end

opt_res = bboptimize(
    cost_rr; 
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    SearchRange = [
        (0.0, 1.0), # w_eco
        (0.0, 1.0), # w_env
        (0.7, 0.95)  # adoption_threshold
    ],
    NumDimensions = 3,
    MaxSteps = 500
)

println("Calibration RR finished")
println("Best Parameters (w_eco, w_env, threshold): ", best_candidate(opt_res))