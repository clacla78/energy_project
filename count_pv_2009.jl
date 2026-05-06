using CSV, DataFrames

df = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
col = "BeginningOfOperation_Year"
df[!,col] = df[col].replace(-99, pd.NA)

df[[!,col]].head()

# Count how many had PV in 2009 (excluding missing and -99)
initial_pv_count = count(year -> !ismissing(year) && year <= 2009, df.BeginningOfOperation_Year)
total_buildings = nrow(df)

println("Count at t=0 (2009): $initial_pv_count")
println("Starting Rate: $(round(initial_pv_count/total_buildings * 100, digits=2))%")