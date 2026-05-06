# Replace your current CSV read block with this to debug:
path_rachat="prix_rachat.csv"
path_elec="price_electricity.csv"

df_rachat = CSV.read(path_rachat, DataFrame)
println("DEBUG: Columns in rachat: ", repr(names(df_rachat)))

df_elec = CSV.read(path_elec, DataFrame)
println("DEBUG: Columns in electricity: ", repr(names(df_elec)))

# Dans la fonction initialize_model de types.jl

# Charger les fichiers avec l'option stripwhitespace = true
df_rachat = CSV.read(path_rachat, DataFrame; stripwhitespace = true)
df_elec = CSV.read(path_elec, DataFrame; stripwhitespace = true)

rename!(df_rachat, strip.(names(df_rachat)))
rename!(df_elec, strip.(names(df_elec)))

params.price_sell_series = df_rachat.Price_cts_kWh ./ 100.0
params.price_buy_series = df_elec.Price_cts_kWh ./ 100.0
