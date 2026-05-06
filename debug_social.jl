include("types.jl")
include("model.jl")

# Configuration du test
params = ModelParams(
    activation_prob = 0.5,     # On facilite l'adoption pour voir la contagion
    adoption_threshold = 0.2    # On baisse le seuil pour tester le moteur
)

model = initialize_model(params)

# --- État Initial ---
n_total = nagents(model)
adopters_init = count(a -> a.pv_installed, allagents(model))
println("--- Début de la Simulation ---")
println("Total agents : $n_total")
println("Adoptants initiaux (t=0) : $adopters_init")

if adopters_init == 0
    println("❌ ERREUR : Aucun adoptant initial. Le u_soc ne pourra jamais décoller.")
end

# Pour suivre l'évolution sans spammer la console
last_count_social = -1

println("\nTracking de l'influence sociale sur 360 mois...")
println("Mois | Agents avec u_soc > 0 | Total Adoptants")
println("----------------------------------------------")

for t in 1:360
    step!(model)
    
    # Compter les agents qui ressentent une influence sociale
    agents_with_social = count(a -> a.u_soc > 0, allagents(model))
    total_adopters = count(a -> a.pv_installed, allagents(model))
    
    # On affiche uniquement si le nombre d'agents influencés change
    if agents_with_social != last_count_social
        @info "Évolution t=$t" Agents_Influencés=agents_with_social Total_Adoptants=total_adopters
        global last_count_social = agents_with_social
    end

    # Petit rapport tous les 120 mois (10 ans) quoi qu'il arrive
    if t % 120 == 0
        println(">> Point d'étape Mois $t : $agents_with_social agents influencés.")
    end
end

final_adopters = count(a -> a.pv_installed, allagents(model))
println("\n--- Simulation Terminée ---")
println("Adoptants finaux : $final_adopters")
println("Agents ayant fini avec une influence sociale : $last_count_social")