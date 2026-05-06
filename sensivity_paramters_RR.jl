using Revise 
using GLMakie
using Agents
using Statistics
using DataFrames

include("types.jl")
include("model.jl")

# --- SUPPRIME OU COMMENTE CES LIGNES ---
# model = initialize_model(params) # <--- ERREUR ICI : params n'existe pas
# run!(model, 360)
# ---------------------------------------

# Data Collection Metric
adoption_rate(m) = (nagents(m) == 0) ? 0.0 : (count(a -> a.pv_installed, allagents(m)) / nagents(m)) * 100

function run_pure_sensitivity_rr(param_to_vary::Symbol, values)
    fig = Figure(size = (900, 600), fontsize = 18)
    ax = Axis(fig[1, 1], 
              title = "RR Architecture: Influence of $param_to_vary",
              xlabel = "Months", 
              ylabel = "Adoption Rate (%)")

    colors = [:red, :blue, :green, :orange, :purple]

    for (i, val) in enumerate(values)
        # Création manuelle de ModelParams avec des valeurs par défaut sécurisées
        # Assure-toi que tous les arguments requis par ton constructeur ModelParams sont présents
        test_params = ModelParams(
            w_eco = 0.0, 
            w_env = 0.0, 
            w_soc = 0.0,
            adoption_threshold = 0.3, 
            activation_prob = 0.1,
            decision_architecture_idx = 0, # RR Mode
            # Ajoute ici les autres paramètres nécessaires s'ils manquent :
            # tpv = 10.0, learning_rate = 0.01, etc.
        )
        
        # Injection de la valeur de sensibilité
        setproperty!(test_params, param_to_vary, val)
        
        # Initialisation propre à chaque itération
        model = initialize_model(test_params)
        
        # On récupère les données (mdata)
        _, mdf = run!(model, 360; mdata = [adoption_rate], when = true, showprogress = false)
        
        t_col = :step in propertynames(mdf) ? :step : :time
        lines!(ax, mdf[!, t_col], mdf.adoption_rate, 
               label = "$param_to_vary = $val", 
               color = colors[mod1(i, length(colors))],
               linewidth = 3)
    end
    
    axislegend(ax, position = :lt)
    return fig
end

# --- ANALYSE ---
# On peut maintenant lancer les tests
fig_rr_soc = run_pure_sensitivity_rr(:w_soc, [0.1, 0.4, 0.7, 1.0])
display(fig_rr_soc)


fig_rr_eco = run_pure_sensitivity_rr(:w_eco, [0.1, 0.4, 0.7, 1.0])
display(fig_rr_eco)

fig_rr_env = run_pure_sensitivity_rr(:w_env, [0.1, 0.4, 0.7, 1.0])
display(fig_rr_env)