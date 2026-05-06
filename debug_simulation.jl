# debug_full.jl
include("types.jl")
include("model.jl")

params = ModelParams()
model = initialize_model(params)



println("Running simulation and generating debug logs...")

for i in 1:10
    step!(model)
end

println("Simulation complete. Detailed data available in 'detailed_debug.csv'.")