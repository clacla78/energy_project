using BlackBoxOptim
using Agents
using CSV
using DataFrames
using Statistics


include("types.jl") 
include("model.jl")

df_real = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
years = 2009:2024

real_counts = [count(row -> row.BeginningOfOperation_Year <= y, eachrow(df_real)) for y in years]

function cost_se(X::Vector{Float64})::Float64
    try
        params = Main.ModelParams(
            decision_architecture_idx = 1, # Mode SE
            w_eco = X[1],
            w_env = X[2],
            w_soc = X[3],           
            adoption_threshold = X[4],
            activation_prob = X[5],
            i_att = X[6]           
        )
        
        model = Main.initialize_model(params)
        _, mdf = run!(model, 192; mdata = [Main.adoption_rate], when = 12, showprogress = false) #192 mois = 16 ans (jusqu'à 2024)
        sim_counts = (mdf.adoption_rate ./ 100) .* nagents(model)
        return sum((sim_counts .- real_counts).^2) / length(real_counts)
    catch e
        return Inf
    end
end

function run_calibration()
    println("Calibration en cours (i_pbc calculé comme 1 - i_att)...")
    opt_res = bboptimize(
        cost_se; 
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        SearchRange = [
            (0.3,0.6), # w_eco
            (0.2, 0.8), # w_env
            (0.2, 0.8), # w_soc
            (0.1, 0.5), # adoption_threshold
            (0.01, 0.2), # activation prob
            (0.4, 0.6)  # i_att (PBC = 1 - i_att)
        ],
        NumDimensions = 6,
        MaxSteps = 1000 
    )
    return opt_res
end

res = run_calibration()
println("Best fit : ", best_candidate(res))


