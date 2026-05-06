using GLMakie
using Agents
using Statistics
using DataFrames

include("types.jl")
include("model.jl")

# Metric
adoption_rate(m) = (count(a -> a.pv_installed, allagents(m)) / nagents(m)) * 100

"""
Sweeps one parameter at a time. 
'baseline_params' ensures other parameters stay constant.
"""
function find_active_range_RR(param::Symbol)
    test_values = 0.0:0.05:1.0
    results = Float64[]
    
    # These are the fixed values for parameters NOT being tested
    baseline = Dict(
        :w_eco => 0.33, :w_env => 0.33, :w_soc => 0.34,
        :adoption_threshold => 0.4, :activation_prob => 0.1
    )
    
    for v in test_values
        # Create params using baseline values
        params = ModelParams(
            w_eco = baseline[:w_eco],
            w_env = baseline[:w_env],
            w_soc = baseline[:w_soc],
            adoption_threshold = baseline[:adoption_threshold],
            activation_prob = baseline[:activation_prob],
            decision_architecture_idx = 0 
        )
        
        # Override only the parameter currently being tested
        setproperty!(params, param, v)
        
        model = initialize_model(params)
        
        # Run for 360 months
        _, mdf = run!(model, 360; mdata = [adoption_rate], showprogress = false)
        push!(results, mdf.adoption_rate[end])
    end
    
    return test_values, results
end

# --- Visualization ---
fig = Figure(size = (1000, 400))
params_to_test = [:w_eco, :w_env, :w_soc]
titles = ["w_eco", "w_env", "w_soc"]

for (i, p) in enumerate(params_to_test)
    x_vals, y_vals = find_active_range_RR(p)
    
    ax = Axis(fig[1, i], title = titles[i], xlabel = "Value", ylabel = "Adoption %")
    scatterlines!(ax, x_vals, y_vals, color = :darkblue, markersize = 10)
end

display(fig)