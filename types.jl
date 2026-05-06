using Agents
using Random
using CSV
using DataFrames
using Parameters

# ==========================================
# 1. AGENT DEFINITION
# ==========================================
@agent struct Household(ContinuousAgent{2, Float64})
    roof_surface::Float64
    pv_potential::Float64
    household_income::Float64
    pv_installed::Bool
    u_eco::Float64
    u_env::Float64
    u_soc::Float64
    co2_saving::Float64
    installed_kWp::Float64   
    base_cost::Float64        
    inc_cap::Float64          

end

# ==========================================
# 2. MODEL PARAMETERS
# ==========================================
@with_kw mutable struct ModelParams
    tpv::Float64 = 21.0
    price_sell_series::Vector{Float64} = []
    price_buy_series::Vector{Float64} = []
    co2_factor::Float64 = 0.5
    current_avg_co2::Float64 = 0.0
    pv_efficiency::Float64 = 0.18
    learning_rate::Float64 = 0.0042
    max_income::Float64 = 1.0
    adoption_threshold::Float64 = 0.3
    activation_prob::Float64 = 0.2
    max_months::Int = 360
    decision_architecture_idx::Int = 1  # 0=RR, 1=SE
    w_eco::Float64 = 0.33
    w_env::Float64 = 0.33
    w_soc::Float64 = 0.34
    i_att::Float64 = 0.5
end



function initialize_model(params::ModelParams; 
    path_buildings="StPrex_Buildings_with_PV_Potential_and_HS.csv", 
    path_income="st_prex_agents_536.csv")
    
    path_rachat="prix_rachat.csv"
    path_elec="price_electricity.csv"

    # Load series and main data
    df_rachat = CSV.read(path_rachat, DataFrame; stripwhitespace = true)
    df_elec = CSV.read(path_elec, DataFrame; stripwhitespace = true)
    params.price_sell_series = df_rachat.Price_cts_kWh ./ 100.0
    params.price_buy_series = df_elec.Price_cts_kWh ./ 100.0

    df_buildings = CSV.read(path_buildings, DataFrame)
    df_income = CSV.read(path_income, DataFrame)
    df_buildings.income = df_income.salary_chf
    params.max_income = maximum(df_buildings.income)

    min_x, max_x = minimum(df_buildings.x), maximum(df_buildings.x)
    min_y, max_y = minimum(df_buildings.y), maximum(df_buildings.y)

    width, height = ceil(max_x - min_x + 2.0), ceil(max_y - min_y + 2.0)

    space = ContinuousSpace((width, height); spacing = 1.0)
    
    
    # agent_step! is now known because it was defined above
    model = StandardABM(
        Household, 
        space; 
        agent_step! = agent_step!, 
        model_step! = model_step!,
        properties = params,
        warn = false
    )

    for row in eachrow(df_buildings)
        pos = (row.x - min_x + 1.0, row.y - min_y + 1.0)
        
        val_year = row.BeginningOfOperation_Year
        is_pre_installed = !ismissing(val_year) && val_year != -99 && val_year <= 2009

        agent_co2 = Float64(row.PV_Pot * params.co2_factor)
        kwp = params.pv_efficiency * Float64(row.area_roof_solar_m2)
        b_cost = ((5523.0 / (kwp^0.4862)) + 578.4) * kwp
        
        norm_income = clamp(Float64(row.income) / params.max_income, 0.0, 1.0)
        sigmoid = 1.0 / (1.0 + exp(-(( (600.0/553.0) * norm_income) - 0.8) * 4.0))
        i_capability = 0.6 * (1.0 - sigmoid)

        add_agent!(pos, model, (0.0, 0.0), 
            Float64(row.area_roof_solar_m2), Float64(row.PV_Pot), Float64(row.income), 
            is_pre_installed, 0.0, 0.0, 0.0, agent_co2, 
            kwp, b_cost, i_capability)
    end

    model_step!(model) # Initialise la moyenne CO2
    return model
end




# ==========================================
# 5. MAIN EXECUTION
# ==========================================
params = ModelParams()
model = initialize_model(params) 

total_n = nagents(model) 
adopters_n = count(a -> a.pv_installed, allagents(model)) 

println("Total Agents: $total_n")
println("Adopters at t=0: $adopters_n")