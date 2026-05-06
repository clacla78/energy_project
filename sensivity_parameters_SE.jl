using Revise 
# Now, any changes you save in model.jl will be updated 
# automatically when you run your main script.
using GLMakie
using Agents
using Statistics
using DataFrames

include("types.jl")
include("model.jl")


#model = initialize_model(params) # This MUST happen after include
#run!(model, 360)

# Data Collection Metric
adoption_rate(m) = (count(a -> a.pv_installed, allagents(m)) / nagents(m)) * 100

"""
    run_sensitivity_with_baseline(param_to_vary::Symbol, values)
Runs the SE model by isolating one parameter (setting others 0) 
to observe its pure effect on adoption dynamics.
"""
function run_sensitivity_with_baseline(param_to_vary::Symbol, values)
    fig = Figure(size = (900, 600), fontsize = 18)
    ax = Axis(fig[1, 1], 
              title = "Sensitivity: $param_to_vary (Non-zero Baseline)",
              xlabel = "Months", 
              ylabel = "Adoption Rate (%)")

    colors = [:red, :blue, :green, :orange, :purple]

    for (i, val) in enumerate(values)
        # 1. Define a REALISTIC baseline instead of zeros
        # These should match your standard model assumptions
        test_params = ModelParams(
            w_eco = 0.5,             # Standard weight
            w_env = 0.33,            # Standard weight
            w_soc = 0.34,            # Standard weight
            i_att = 0.5,             # Standard importance
            adoption_threshold = 0.35, # Standard threshold
            activation_prob = 0.1,   #
            decision_architecture_idx = 1 # Force SE Mode
        )
        
        # 2. Overwrite only the parameter we are studying
        setproperty!(test_params, param_to_vary, val)
        
        # 3. Initialize and run
        model = initialize_model(test_params)
        _, mdf = run!(model, 360; mdata = [adoption_rate], when = true, showprogress = false)
        
        t_col = :step in propertynames(mdf) ? :step : :time
        lines!(ax, mdf[!, t_col], mdf.adoption_rate, 
               label = "$param_to_vary = $val", 
               color = colors[mod1(i, length(colors))],
               linewidth = 3)
    end
    
    axislegend(ax, position = :lt, framevisible = false)
    return fig
end

# --- ANALYSIS GENERATION ---

# 1. Pure Social Influence (i_soc)
# Agents decide BASED ONLY on their neighbors' behavior.

#ig3 = run_sensitivity_with_baseline(:w_eco, [0.2, 0.5, 0.8, 1.0])
#isplay(fig3)

fig4 = run_sensitivity_with_baseline(:w_env, [0.2, 0.5, 0.8, 1.0])
display(fig4)


# 5. Impact of Activation Probability
fig5 = run_sensitivity_with_baseline(:activation_prob, [0.01, 0.05, 0.1, 0.2])
display(fig5)

# 6. Impact of Adoption Threshold

#ig6 = run_sensitivity_with_baseline(:adoption_threshold, [0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

#isplay(fig6)