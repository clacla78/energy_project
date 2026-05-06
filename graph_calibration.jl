using GLMakie
using CSV
using DataFrames


df_real_csv = CSV.read("pv_adoption_points.csv", DataFrame)

real_map = Dict(df_real_csv.Year .=> df_real_csv.Cumulative_Adoption_Rate .* 100)

real_percentages = Float64[]
last_val = 0.0
for y in 2009:2024
    global last_val = get(real_map, y, last_val)
    push!(real_percentages, last_val)
end

# replace with the results of the optimization (best_params_vec) from calibration_SE.jl
best_params_vec =[0.32829349966540405, 0.500451517590326, 0.6549373730764283, 0.32226302977055343, 0.0644281520945365, 0.35493325091891764]

best_model_params = ModelParams(
    decision_architecture_idx = 1, # Strictement SE
    w_eco = best_params_vec[1],
    w_env = best_params_vec[2],
    w_soc = best_params_vec[3],
    adoption_threshold = best_params_vec[4],
    activation_prob = best_params_vec[5],
    i_att = 0.5 
)


println("Simulation .")
model_360 = initialize_model(best_model_params)

_, mdf_360 = run!(model_360, 360; mdata = [adoption_rate], when = 12)

annees_sim = 2009:(2009 + 30) 
sim_to_plot = mdf_360.adoption_rate 

annees_real = 2009:2024

fig = Figure(size = (900, 600), fontsize = 18)
ax = Axis(fig[1, 1],
    title = "Diffusion PV St-Prex : Historique et Projection (SE)",
    xlabel = "Année",
    ylabel = "Taux d'Adoption (%)",
    xticks = 2009:5:2039)

scatterlines!(ax, annees_real, real_percentages, 
    label = "Historic data (2009-2024)",
    color = :blue, markersize = 10)


lines!(ax, annees_sim, sim_to_plot, 
    label = "Simulation SE (360 mounths)",
    color = :red, linewidth = 3, linestyle = :dash)


vlines!(ax, [2024], color = :gray, linestyle = :dot)
text!(ax, 2024.5, 5, text = "end of historic data", color = :gray, fontsize = 12)

axislegend(ax, position = :lt)
ax.limits[] = (2009, 2040, 0, 100) 

display(fig)