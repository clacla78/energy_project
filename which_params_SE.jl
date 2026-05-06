using GLMakie
using Agents
using Statistics
using DataFrames

include("types.jl")
include("model.jl")

# Métrique : Pourcentage d'adoption
adoption_rate(m) = (count(a -> a.pv_installed, allagents(m)) / nagents(m)) * 100

"""
    find_active_range_se(param::Symbol)
Balaye un paramètre de 0.0 à 1.0 pour identifier les points de bascule.
"""
function find_active_range_se(param::Symbol)
    test_values = 0.0:0.1:1.0
    results = Float64[]
    
    for v in test_values
        # Baseline standard pour l'architecture SE
        params = Main.ModelParams(
            w_eco = 0.5, 
            w_env = 0.3, 
            w_soc = 0.3,          
            i_att = 0.5,         
            adoption_threshold = 0.5, 
            activation_prob = 0.1,
            decision_architecture_idx = 1 # Force SE Mode
        )
        
        # Application de la valeur de test
        setproperty!(params, param, v)
        
        # Initialisation (Baseline 2009 : 1 seul adoptant)
        model = Main.initialize_model(params)
        
        # Simulation sur 360 mois (30 ans)
        _, mdf = run!(model, 360; mdata = [adoption_rate], showprogress = false)
        push!(results, mdf.adoption_rate[end])
    end
    
    # Calcul des points de bascule (Décollage > 5% et Saturation > 90%)
    idx_start = findfirst(r -> r > 5.0, results)
    idx_sat = findfirst(r -> r > 90.0, results)
    
    # Correction de la faute de frappe 'idx_stara' -> 'idx_start'
    start_v = test_values[idx_start === nothing ? 1 : idx_start]
    sat_v = test_values[idx_sat === nothing ? length(test_values) : idx_sat]
    
    return test_values, results, start_v, sat_v
end

# --- Logique de Visualisation ---

# Utilisation de Figure() de GLMakie
fig = Figure(size = (1000, 1200), fontsize = 20)

params_to_test = [:i_att, :w_soc, :w_env, :w_eco, :adoption_threshold]
titles = [
    "Attitude vs PBC Importance (i_att)", 
    "Poids de l'Influence Sociale (w_soc)", 
    "Poids des Préoccupations Environnementales (w_env)",
    "Poids des Préoccupations Économiques (w_eco)",
    "Seuil d'Adoption (Threshold)"
]

for (i, p) in enumerate(params_to_test)
    x_vals, y_vals, start_v, sat_v = find_active_range_se(p)
    
    ax = Axis(fig[i, 1], 
              title = titles[i], 
              xlabel = "Valeur de $p", 
              ylabel = "Adoption Finale %")
    
    # Ligne de tendance
    scatterlines!(ax, x_vals, y_vals, color = :royalblue, linewidth = 3, markersize = 12)
    
    # Marqueurs verticaux
    vlines!(ax, [start_v], color = :forestgreen, linestyle = :dash, linewidth = 2)
    vlines!(ax, [sat_v], color = :firebrick, linestyle = :dash, linewidth = 2)
    
    # Annotations textuelles
    text!(ax, start_v, 10, text = " Start: $start_v", color = :forestgreen, align = (:left, :bottom))
    text!(ax, sat_v, 90, text = " Saturation: $sat_v", color = :firebrick, align = (:right, :top))
    
    # Correction de l'erreur : Utilisation explicite de Makie.ylims!
    Makie.ylims!(ax, -5, 105) 
end

rowgap!(fig.layout, 30)
display(fig)