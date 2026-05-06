using Agents, DataFrames, Statistics, CairoMakie

# Define the range to explore
# We keep weights normalized (sum = 1 is often best, but we'll explore independently)
parameter_ranges = Dict(
    :decision_architecture_idx => 0:1,
    :w_eco => 0:0.1:1,
    :w_env => 0:0.1:1,
    :w_soc => 0:0.1:1
)

# Run the sweep (using your model functions)
# 'n' is the number of steps (360 months)
adf, mdf = paramscan(
    parameter_ranges,
    initialize_model;
    agent_step! = agent_step!,
    model_step! = dummystep,
    mdata = [adoption_rate],
    n = 360
)


function plot_sensitivity(mdf)
    # Filter for a specific architecture to compare
    df_rr = filter(row -> row.decision_architecture_idx == 0, mdf)
    df_se = filter(row -> row.decision_architecture_idx == 1, mdf)

    fig = Figure(size = (1000, 500))
    
    # Plot RR Sensitivity
    ax1 = Axis(fig[1, 1], title = "Rational (RR) Adoption", xlabel = "w_eco", ylabel = "Adoption Rate (%)")
    scatter!(ax1, df_rr.w_eco, df_rr.adoption_rate, color = df_rr.w_soc, colormap = :viridis)
    
    # Plot SE Sensitivity
    ax2 = Axis(fig[1, 2], title = "Social-Psych (SE) Adoption", xlabel = "w_eco", ylabel = "Adoption Rate (%)")
    scatter!(ax2, df_se.w_eco, df_se.adoption_rate, color = df_se.w_soc, colormap = :plasma)
    
    Colorbar(fig[1, 3], label = "Weight Social (w_soc)")
    return fig
end

display(plot_sensitivity(mdf))