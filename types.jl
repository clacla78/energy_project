using Agents
using Random
using CSV
using DataFrames
using Parameters

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
    neighbors::Vector{Int} # Store neighbor IDs
end

@with_kw mutable struct ModelParams
    tpv::Float64 = 21.0
    price_sell_series::Vector{Float64} = []
    price_buy_series::Vector{Float64} = []
    co2_factor::Float64 = 0.5
    current_avg_co2::Float64 = 0.0
    pv_efficiency::Float64 = 0.18
    learning_rate::Float64 = 0.0042
    max_income::Float64 = 1.0
    adoption_threshold::Float64 = 0.7
    activation_prob::Float64 = 0.01
    max_months::Int = 360



    decision_architecture_idx::Int = 1 # 0=RR, 1=SE
    w_eco::Float64 = 0.3
    w_env::Float64 = 0.6
    w_soc::Float64 = 0.15
    i_att::Float64 = 0.6

    neighbors_radius::Float64 = 50
end




# ==========================================
# 5. MAIN EXECUTION
# ==========================================

# params = ModelParams()
# model = initialize_model(params) 

# total_n = nagents(model) 
# adopters_n = count(a -> a.pv_installed, allagents(model)) 

# println("Total Agents: $total_n")
# println("Adopters at t=0: $adopters_n")
