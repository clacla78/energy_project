using Plots

function agent_step!(agent::Household, model::AgentBasedModel)
    t = abmtime(model)
    annee_actuelle = 2009 + floor(Int, t / 12)
    idx = clamp(annee_actuelle - 2009 + 1, 1, length(model.price_sell_series))

    # Economic Logic with Avoided Costs
    # Installation capacity
  # installed_kWp = model.pv_efficiency * agent.roof_surface
    
   #base_cost = ((5523.0 / (installed_kWp^0.4862)) + 578.4) * installed_kWp
    current_investment_cost = agent.base_cost * exp(-model.learning_rate * t)

    # Avoided Cost: Savings = (Generation * Grid_Buy_Price) + (Excess_Sold * Sell_Price)
    # avoided cost per kWh = grid electricity price
    grid_price = model.price_buy_series[idx]
    annual_savings = agent.pv_potential * grid_price 
    
    # Payback Period (TPP)
    tpp = current_investment_cost / max(annual_savings, 0.01)
    
    # Update U_eco based on profitability (Target Payback Period tpv=21, like in filatova)
    agent.u_eco = clamp((model.tpv - tpp) / model.tpv, 0, 1)

    # Social Logic
    neighbors = nearby_agents(agent, model, 50.0) 
   #count_n = count(n -> n.pv_installed, neighbors)
    #otal_n = length(collect(neighbors))
  # total_n = count(x -> true, neighbors) #a faster version
    n_pv = 0
    n_total = 0
    for n in nearby_agents(agent, model, 50.0)
        n_total += 1
        if n.pv_installed
            n_pv += 1
        end
    end
    

    agent.u_soc = n_total > 0 ? n_pv / n_total : 0.0


    # Avoided CO2 for env calculation

    #s_co2_bar = sum(a.co2_saving for a in allagents(model)) / nagents(model)
    #agent.u_env = exp(scaled_diff) / (1.0 + exp(scaled_diff))
    #println("DEBUG Agent $(agent.id) | CO2 Saving: $(agent.co2_saving) | Mean: $s_co2_bar | Diff: $diff | u_env: $(agent.u_env)")
    
    agent.co2_saving = agent.pv_potential * 0.56 
    s_co2_bar = model.current_avg_co2 
    diff = agent.co2_saving - s_co2_bar
    scaled_diff = diff / 5000.0 # Adjusted scale factor

    agent.u_env = 1.0 / (1.0 + exp(-scaled_diff))

    inc_cap = get_inc_capability(agent, model)
    score = calculate_score(agent, model, inc_cap) 
    #if t % 50 == 0 println("SE Score: ", score) end
    #--- Debug Output ---
    
    #println("DEBUG [Month $t] Agent $(agent.id) | Income: $(agent.household_income)")
    #println("U_eco: $(agent.u_eco) | U_soc: $(agent.u_soc) | U_env: $(agent.u_env)")
    #println("Inc_Capability: $inc_cap | Final Score: $score")
    #println("--------------------------------------------------")
    

    #debug SE Score and its components
    #println("DEBUG [Month $t] Agent $(agent.id) | SE Score: $score | Attitude: $(model.w_eco * agent.u_eco + model.w_env * agent.u_env) | PBC: $inc_cap | Norm: $(agent.u_soc)")
    #=

    open("detailed_debug.csv", "a") do f
        # Write header if it's the first step (optional, can be done manually)
        if t == 0 && agent.id == 1
            #println(f, "t,agent_id,income,u_eco,inc_capability,u_soc,u_env,final_score")
        end
        #println(f, "$t,$(agent.id),$(agent.household_income),$(agent.u_eco),$inc_capability,$(agent.u_soc),$(agent.u_env),$score")
    end

    =#
    #println(model.adoption_threshold)
    #println(score)
    # Adoption Decision
    if !agent.pv_installed && rand() < model.activation_prob
        if score > model.adoption_threshold
            agent.pv_installed = true
            #println("Agent $(agent.id) adopted PV at month $t with score $score")
        end
    end
end

function get_inc_capability(agent::Household, model::AgentBasedModel)
    norm_income = clamp(agent.household_income / model.max_income, 0.0, 1.0)
    max_barrier = 0.6
    steepness = 4.0
    mid_point = 0.8
    normal = 600.0 / 553.0
    
    # if the income is very low, we want a high inc_capability (close to max_barrier), and if it's high, we want it close to 0
    sigmoid = 1.0 / (1.0 + exp(-((normal * norm_income) - mid_point) * steepness))
    #println("DEBUG Agent $(agent.id) | Income: $(agent.household_income) | Normalized: $norm_income | Sigmoid: $sigmoid | Inc_Capability: $(max_barrier * (1.0 - sigmoid))")
    return max_barrier * (1.0 - sigmoid)
end


function model_step!(model)
    # Calculate the average CO2 saving of all agents ONCE per step
    total_co2 = 0.0
    for a in allagents(model)
        total_co2 += a.co2_saving
    end
    model.current_avg_co2 = total_co2 / nagents(model)
end



function calculate_score(agent::Household, model::AgentBasedModel, inc_cap::Float64)
    if model.decision_architecture_idx == 0 # RR
        if agent.u_eco < inc_cap return 0.0 end
       
        if agent.u_eco > inc_cap   # so basically if income is higher, inc_cap is lower, so it is easier to get above it, the agent passes the step 
            return (model.w_eco * agent.u_eco + model.w_env * agent.u_env + model.w_soc * agent.u_soc) 
        end
 else # SE Logic 
        
        # 1. Calculate the Utility of Attitude (u_att)
        # Combination of economic and environmental utilities weighted by w
        u_att = (model.w_eco * agent.u_eco) + (model.w_env * agent.u_env) 
        
        # 2. Calculate the Utility of Perceived Behavioral Control (u_pbc)
        # Derived from the income barrier (inc_cap)
        u_pbc = (1.0 - inc_cap) * model.w_eco
        
        # 3. Calculate the Internal Factor
        # Balance between Attitude and PBC using importance factor i_att
        i_pbc = 1.0 - model.i_att
        internal_factor = (model.i_att * u_att) + (i_pbc * u_pbc)
        
        # 4. Final Decision Score
        # Weigh the Internal Factor against the Social Utility (u_soc) using i_soc
        score = ((1.0 - model.w_soc) * internal_factor) + (model.w_soc * agent.u_soc)
        
        # --- Debugging ---
        #=
        println("--- DEBUG Agent $(agent.id) (Month $(abmtime(model))) ---")
        println("Utilities:  [u_eco: $(round(agent.u_eco, digits=3)), u_env: $(round(agent.u_env, digits=3)), u_soc: $(round(agent.u_soc, digits=3))]")
        println("Weights:    [w_eco: $(model.w_eco), w_env: $(model.w_env), w_soc: $(model.w_soc)]")
        println("Factors:    [i_att: $(model.i_att), i_pbc: $(1.0 - model.i_att)]")
        println("Results:    [u_att (Weighted Eco+Env): $(round(u_att, digits=3)), u_pbc: $(round(u_pbc, digits=3))]")
        println("Final:      [Internal Factor: $(round(internal_factor, digits=3)) | Total Score: $(round(score, digits=3))]")
        println("Threshold:  [Score: $(round(score, digits=3)) vs Threshold: $(model.adoption_threshold)]")
        println("Decision:   $(score > model.adoption_threshold ? "PASS" : "FAIL")")
        println("--------------------------------------------------")
         =#   
        return score
        
    end
end


#function get_detailed_stats(model)
 #   agents = allagents(model)
  #  N = nagents(model)
    
   # avg_u_eco = sum(a.u_eco for a in agents) / N
    #avg_u_soc = sum(a.u_soc for a in agents) / N
    #avg_u_env = sum(a.u_env for a in agents) / N
    
    # how many are within 0.1 of crossing it
 
   # close_to_adoption = count(a -> !a.pv_installed && 
  #                             calculate_score(a, model) > (model.adoption_threshold - 0.1), agents)
    
 #   return (avg_u_eco, avg_u_soc, avg_u_env, close_to_adoption)
#end



#df = CSV.read("st_prex_agents_536.csv", DataFrame)
#println(maximum(df.salary_chf))

#real_avg = sum(a.co2_saving for a in allagents(model)) / nagents(model)
#println(real_avg)


t = @elapsed model = initialize_model(params)
println("Initialization : $t secondes")

t_sim = @elapsed run!(model, 360)
println("Simulation : $t_sim secondes")