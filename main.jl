using InteractiveDynamics
using Agents
using GLMakie

include("types.jl")
include("model.jl")


params = ModelParams()
path_b = "StPrex_Buildings_with_PV_Potential_and_HS.csv"
path_i = "C:\\Users\\clari\\Documents\\MA3\\Project energy St Prex\\julia_clarisse\\st_prex_agents_536.csv"

model = initialize_model(params; path_buildings=path_b, path_income=path_i)

interaction_params = Dict(
    :decision_architecture_idx => 0:1, 
    :w_eco => 0:0.05:1,
    :w_env => 0:0.05:1,
    :w_soc => 0:0.05:1,
    :i_att => 0:0.05:1,
    :adoption_threshold => 0:0.05:1,
    :activation_prob => 0:0.001:1# Finer steps for the prob to help you slow it down
)


# The value at t=0 comes from the agents' state right after initialization
adoption_rate(m) = (count(a -> a.pv_installed, allagents(m)) / nagents(m)) * 100
m_data = [adoption_rate] 

agent_color(a) = a.pv_installed ? :green : :red
agent_size(a) = a.pv_installed ? 12 : 7

function get_dynamic_title(m)
    arch = m.decision_architecture_idx == 0 ? "RR (Rational)" : "SE (Social-Psych)"
    t = Int(abmtime(m))
    year = 2009 + floor(Int, t/12)
    return "Architecture: $arch | Month: $t | Year: $year"
end

fig, abmobs = abmexploration(
    model;
    params = interaction_params,
    agent_color,
    agent_size,
    mdata = m_data,
    figure = (size = (1300, 900),)
)

all_axes = [b for b in fig.content if b isa Axis]
ax_map = all_axes[1]
ax_plot = all_axes[2]

ax_plot.ylabel = "Adoption Rate (%)"
ax_plot.xlabel = "Year"
ax_plot.limits[] = (0, 360, 0, 100)



# Create ticks every 5 years (60 months)
tick_months = 0:60:360
tick_years = [string(2009 + floor(Int, m/12)) for m in tick_months]
ax_plot.xticks = (tick_months, tick_years)


on(abmobs.model) do m
    new_title = get_dynamic_title(m)
    ax_map.title[] = new_title
    
    # Auto-stop condition
    if abmtime(m) >= 360 && abmobs.run[]
        abmobs.run[] = false
        ax_map.title[] = "[FINISH] " * new_title
        println("Simulation reached 30 years.")
    end
end

Label(fig[2, 1], "Historical Target: ~14% by 2024 (Month 180)", 
      tellwidth = false, font = :italic, color = :blue)

display(fig)