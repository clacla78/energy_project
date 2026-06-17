"""
meta_calibration_v2.jl

"""

using CSV
using DataFrames
using Statistics
using BlackBoxOptim
using Printf
using Dates
using Agents

include("types.jl")
include("model.jl")


const DF_REAL = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
const TOTAL_BLD = nrow(DF_REAL)

df_adopters = filter(row -> !ismissing(row.BeginningOfOperation_Year) && 
                            row.BeginningOfOperation_Year != -99 &&
                            row.BeginningOfOperation_Year < 2024, DF_REAL)

history_counts = combine(groupby(df_adopters, :BeginningOfOperation_Year), nrow => :new_adopters)
sort!(history_counts, :BeginningOfOperation_Year)
history_counts.cumulative_adopters = cumsum(history_counts.new_adopters)
history_counts.adoption_rate = (history_counts.cumulative_adopters ./ TOTAL_BLD) .* 100

const YEARS_HIST = 2009:2023
const REAL_RATES = map(YEARS_HIST) do y
    idx = findlast(history_counts.BeginningOfOperation_Year .<= y)
    isnothing(idx) ? 0.0 : history_counts.adoption_rate[idx]
end


adoption_rate_fn(model) = (count(a -> a.pv_installed, allagents(model)) / nagents(model)) * 100

function cost_rr(X::Vector{Float64})::Float64
    w_e, w_n, threshold, prob = X[1], X[2], X[3], X[4]
    w_s = max(0.0, 1.0 - (w_e + w_n)) 

    params = ModelParams(
        decision_architecture_idx = 0, # RR Mode
        w_eco = w_e, w_env = w_n, w_soc = w_s,
        adoption_threshold = threshold, activation_prob = prob
    )
    
    model = initialize_model(params)
    _, mdf = run!(model, 180; mdata = [adoption_rate_fn], when = 12, showprogress = false)
    
    len = min(length(mdf.adoption_rate_fn), length(REAL_RATES))
    return sum((mdf.adoption_rate_fn[1:len] .- REAL_RATES[1:len]).^2) / len
end

function cost_se(X::Vector{Float64})::Float64
    params = ModelParams(
        decision_architecture_idx = 1, # Mode SE
        w_eco = X[1], w_env = X[2], w_soc = X[3],
        adoption_threshold = X[4], activation_prob = X[5], i_att = X[6]
    )
    
    model = initialize_model(params)
    _, mdf = run!(model, 180; mdata = [adoption_rate_fn], when = 12, showprogress = false)
    
    len = min(length(mdf.adoption_rate_fn), length(REAL_RATES))
    return sum((mdf.adoption_rate_fn[1:len] .- REAL_RATES[1:len]).^2) / len
end


const N_CALIBRATIONS = 15
const MAX_STEPS      = 900

const TIMESTAMP  = replace(string(now()), r"[:\.]" => "-")
const CSV_FILE   = "calibration_results_$(TIMESTAMP).csv"

const CSV_COLS = [:architecture, :run_id, :fitness,
                  :w_eco, :w_env, :w_soc,
                  :adoption_threshold, :activation_prob, :i_att, :learning_rate]


const SEARCH_RR = [
    (0.1, 0.9),   # w_eco
    (0.1, 0.9),   # w_env
    (0.4, 0.9),   # adoption_threshold
    (0.001, 0.05) # activation_prob
]

const SEARCH_SE = [
    (0.1, 0.6),   # w_eco
    (0.1, 0.6),   # w_env
    (0.1, 0.6),  # w_soc
    (0.2, 0.7),   # adoption_threshold
    (0.001, 0.1), # activation_prob
    (0.3, 0.8)    # i_att
]


function init_csv()
    empty_df = DataFrame([col => [] for col in CSV_COLS])
    CSV.write(CSV_FILE, empty_df)
    println("Output CSV : $CSV_FILE\n")
end

function append_row(row::NamedTuple)
    df = DataFrame([col => [get(row, col, NaN)] for col in CSV_COLS])
    CSV.write(CSV_FILE, df; append=true)
end

function run_optim(cost_fn, search_range, n_dims)
  
    callback = function(oc)
        if best_fitness(oc) < 0.1
            return true   # stoppe l'optimisation par arrêt anticipé
        end
        return false
    end

    opt = bboptimize(cost_fn;
        SearchRange      = search_range,
        NumDimensions    = n_dims,
        MaxSteps         = MAX_STEPS,
        DisplayFlag      = :off,        # silence BlackBoxOptim
        #TraceMode        = :silent,     # silence le progress
        CallbackInterval = 10,
        CallbackFunction = callback
    )

    return best_candidate(opt), best_fitness(opt)
end

function calibrate_rr(run_id::Int)
    best_p, fitness = run_optim(cost_rr, SEARCH_RR, 4)
    w_eco = best_p[1]
    w_env = best_p[2]
    total_w = w_eco + w_env
    if total_w > 0.8
        w_eco = (w_eco / total_w) * 0.8
        w_env = (w_env / total_w) * 0.8
    end
    w_soc = 1.0 - (w_eco + w_env)

    return (
        architecture       = "RR", run_id = run_id, fitness = fitness,
        w_eco              = w_eco, w_env = w_env, w_soc = w_soc,
        adoption_threshold = best_p[3], activation_prob = best_p[4],
        i_att              = NaN, learning_rate = NaN
    )
end

function calibrate_se(run_id::Int)
    best_p, fitness = run_optim(cost_se, SEARCH_SE, 6)
    return (
        architecture       = "SE", run_id = run_id, fitness = fitness,
        w_eco              = best_p[1], w_env = best_p[2], w_soc = best_p[3],
        adoption_threshold = best_p[4], activation_prob = best_p[5],
        i_att              = best_p[6], learning_rate = NaN
    )
end


function run_meta_calibration(n_runs::Int = N_CALIBRATIONS)
    init_csv()

    for (arch_name, calibrate_fn) in [("RR", calibrate_rr), ("SE", calibrate_se)]
        println("── Architecture $arch_name ──────────────────")
        for i in 1:n_runs
            row = calibrate_fn(i)
            append_row(row)
            @printf("  [%s] Calibration %2d/%d  →  fitness = %.6f\n",
                    arch_name, i, n_runs, row.fitness)
        end
        println()
    end

    df = CSV.read(CSV_FILE, DataFrame)
    println("Done. $CSV_FILE  ($(nrow(df)) rows)")

    for arch in ["RR", "SE"]
        sub = filter(:architecture => ==(arch), df)
        best_idx = argmin(sub.fitness)
        println("  $arch  best fitness = $(round(sub.fitness[best_idx], digits=6))  " *
                "(mean = $(round(mean(sub.fitness), digits=4)))")
    end

    return df
end

run_meta_calibration()