using InteractiveDynamics
using Agents
using BenchmarkTools
using GLMakie

include("types.jl")
include("model.jl")


params = ModelParams()
path_b = "StPrex_Buildings_with_PV_Potential_and_HS.csv"
path_i = "st_prex_agents_536.csv"


# Simple run

input_data = load_input_data()

@btime model = initialize_model($params; path_buildings=path_b, path_income=path_i)

@btime model = initialize_model($params, $input_data)

@btime run!($model, 192; mdata = [adoption_rate], when = 12, showprogress = false)

@code_warntype agent_step!(first(allagents(model)), model)