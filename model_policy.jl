# model_policy.jl

if !isdefined(Main, :Household)
    include("types.jl")
end

function agent_step!(agent::Household, model::AgentBasedModel)
    t = abmtime(model)
    annee_actuelle = 2009 + floor(Int, t / 12)
    idx = clamp(annee_actuelle - 2009 + 1, 1, length(model.price_sell_series))


    current_investment_cost_gross = agent.base_cost * exp(-model.learning_rate * t)

    subsidy_base = 350.0
    subsidy_per_kwp = 300.0
    total_allocated_subsidy = subsidy_base + (subsidy_per_kwp * agent.installed_kWp)
       
    total_allocated_subsidy = min(total_allocated_subsidy, current_investment_cost_gross * 0.30)

    current_investment_cost_gross = agent.base_cost * exp(-model.learning_rate * t)
    if t >= 180
        subsidy_base = 350.0
        subsidy_per_kwp = 300.0
        total_allocated_subsidy = subsidy_base + (subsidy_per_kwp * agent.installed_kWp)
        total_allocated_subsidy = min(total_allocated_subsidy, current_investment_cost_gross * 0.30)
        
        current_investment_cost = current_investment_cost_gross - total_allocated_subsidy
    else
        current_investment_cost = current_investment_cost_gross
    end

    grid_price = model.price_buy_series[idx] 
    annual_savings = agent.pv_potential * grid_price 
    

    tpp = current_investment_cost / max(annual_savings, 0.01)
    
    agent.u_eco = clamp((model.tpv - tpp) / model.tpv, 0, 1)

    n_pv = 0
    n_total = 0
    for n_id in agent.neighbors
        n = model[n_id]
        n_total += 1
        if n.pv_installed
            n_pv += 1
        end
    end
    agent.u_soc = n_total > 0 ? n_pv / n_total : 0.0 

    agent.co2_saving = agent.pv_potential * 0.56 
    s_co2_bar = model.current_avg_co2 
    diff = agent.co2_saving - s_co2_bar
    scaled_diff = diff / 5000.0 

    agent.u_env = 1.0 / (1.0 + exp(-scaled_diff))

    inc_cap = get_inc_capability(agent.household_income, model.max_income)
    score = calculate_score(agent, model, inc_cap)
    if !agent.pv_installed && rand() < model.activation_prob
        if score > model.adoption_threshold
            agent.pv_installed = true
        end 
    end
end

function get_inc_capability(income::Float64, max_income::Float64)
    norm_income = clamp(income / max_income, 0.0, 1.0)
    max_barrier = 1.0
    steepness = 5.0
    mid_point = 0.5
    sigmoid = 1.0 / (1.0 + exp(-(norm_income - mid_point) * steepness)) 
    return max_barrier * (1.0 - sigmoid)
end

function model_step!(model)
    total_co2 = 0.0
    for a in allagents(model)
        total_co2 += a.co2_saving 
    end
    model.current_avg_co2 = total_co2 / nagents(model)
end

function calculate_score(agent::Household, model::AgentBasedModel, inc_cap::Float64)
    if model.decision_architecture_idx == 0 
        model.w_soc = 1.0 - (model.w_eco + model.w_env)
        model.w_soc = max(0.0, model.w_soc)
        if agent.u_eco < inc_cap return 0.0 end 
        if agent.u_eco > inc_cap   
            return (model.w_eco * agent.u_eco + model.w_env * agent.u_env + model.w_soc * agent.u_soc) 
        end
    else 
        u_att = (model.w_eco * agent.u_eco) + (model.w_env * agent.u_env) 
        u_pbc = (1.0 - inc_cap) 
        internal_factor = (model.i_att * u_att) + ((1.0 - model.i_att) * u_pbc)
        
        effective_u_soc = agent.u_soc^0.5  
        score = ((1.0 - model.w_soc) * internal_factor) + (model.w_soc * effective_u_soc)

        return score
    end
end

function initialize_neighbors!(model::AgentBasedModel, radius::Float64)
    for agent in allagents(model)
        agent.neighbors = Int[n.id for n in nearby_agents(agent, model, radius)] 
    end
end

function initialize_model(params::ModelParams; 
    path_buildings="StPrex_Buildings_with_PV_Potential_and_HS.csv", 
    path_income="st_prex_agents_536.csv")
    
    path_rachat="prix_rachat.csv"
    path_elec="price_electricity.csv"

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
        b_cost = ((5523.0 / (kwp^0.4862)) + 578.4  + (156.2 * exp(-0.2321 * kwp))) * kwp 
        i_capability = get_inc_capability(Float64(row.income), params.max_income)

        add_agent!(pos, model, (0.0, 0.0), 
            Float64(row.area_roof_solar_m2), Float64(row.PV_Pot), Float64(row.income), 
            is_pre_installed, 0.0, 0.0, 0.0, agent_co2, 
            kwp, b_cost, i_capability,
            Int[]
            ) 
    end
    
    initialize_neighbors!(model, params.neighbors_radius)
    model_step!(model) 
    return model
end