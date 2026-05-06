using Plots
import Plots: plot, vline!, hline!, display, savefig

# --- 1. INCOME CAPABILITY (Financial Barrier) ---
function plot_income_logic(max_income_real=187843.48)
    max_barrier = 0.6
    steepness = 4.0
    mid_point = 0.8
    normal = 600.0 / 553.0
    
    # We plot up to 250k to see the behavior past the max
    incomes = range(0, 250000, length=500)
    
    caps = map(incomes) do inc
        norm_income = clamp(inc / max_income_real, 0.0, 1.0)
        sigmoid = 1.0 / (1.0 + exp(-((normal * norm_income) - mid_point) * steepness))
        return max_barrier * (1.0 - sigmoid)
    end

    p1 = Plots.plot(incomes, caps, 
        title="Financial Barrier (inc_cap) vs Income",
        xlabel="Household Income (CHF)", 
        ylabel="Barrier Level (Lower is better)",
        lw=3, color=:red, label="inc_cap",
        grid=true)
    
    # Marker for the 80% threshold where the barrier drops
    Plots.vline!(p1, [max_income_real * 0.8], label="Transition (80% Max)", ls=:dash, color=:black)
    return p1
end

# --- 2. ENVIRONMENTAL SIGMOID ---
function plot_env_logic(avg_co2_saving=17634.81)
    # Range of possible CO2 savings (from 0 to 40k to see the spread)
    co2_range = range(0, 40000, length=500)
    
    u_envs = map(co2_range) do c
        diff = c - avg_co2_saving
        scaled_diff = diff / 5000.0 # Scale factor from your code
        return exp(scaled_diff) / (1.0 + exp(scaled_diff))
    end

    p2 = Plots.plot(co2_range, u_envs, 
        title="Environmental Utility (u_env) vs CO2",
        xlabel="CO2 Saving (kg)", 
        ylabel="u_env Score (Higher is better)",
        lw=3, color=:green, label="u_env",
        grid=true)

    # Marker for the population mean
    Plots.vline!(p2, [avg_co2_saving], label="Pop. Mean ($avg_co2_saving)", ls=:dash, color=:black)
    return p2
end

# --- EXECUTION ---
# Using your exact values found in the debug
p1 = plot_income_logic(187843.48) 
p2 = plot_env_logic(17634.81)

final_plot = Plots.plot(p1, p2, layout=(2,1), size=(800, 900))
Plots.display(final_plot)

# Save the figure to your current folder
Plots.savefig("st_prex_model_sigmoids.png")