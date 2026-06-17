
using CSV
using DataFrames
using Statistics
using CairoMakie
using Agents
using Printf
using Dates

include("types.jl")
include("model.jl")

adoption_rate_fn(model) = (count(a -> a.pv_installed, allagents(model)) / nagents(model)) * 100

function print_se_comprehensive_extremes(df_se_sorted::DataFrame)
    if nrow(df_se_sorted) == 0
        println("\n[Warning] SE dataframe is empty. Skipping table generation.")
        return
    end

    top_3 = first(df_se_sorted, 3)
    worst_3 = last(df_se_sorted, 3)
    
    top_3[!, :Tier] .= "BEST 3 (Top Fit)"
    worst_3[!, :Tier] .= "WORST 3 (Poor Fit)"
    df_extremes = vcat(top_3, worst_3)

    println("\n" * "═"^120)
    println("                             UNFILTERED SE TOP 3 BEST vs BOTTOM 3 WORST CONFIGURATIONS")
    println("═"^120)
    @printf("| %-18s | %-8s | %-7s | %-6s | %-6s | %-6s | %-6s | %-9s | %-8s |\n",
            "Performance Tier", "Run ID", "RMSE", "w_eco", "w_env", "w_soc", "i_att", "adopt_th", "act_prob")
    println("|" * "-"^118 * "|")
    
    for r in eachrow(df_extremes)
        i_att_val = hasproperty(r, :i_att) ? r.i_att : 0.3
        @printf("| %-18s | #%-6d | %-7.3f | %-6.2f | %-6.2f | %-6.2f | %-6.2f | %-9.2f | %-8.3f |\n",
                r.Tier, r.run_id, r.rmse, r.w_eco, r.w_env, r.w_soc, i_att_val, r.adoption_threshold, r.activation_prob)
    end
    println("═"^120)

    output_path = joinpath("plots_output", "se_unfiltered_extremes_profiles.csv")
    CSV.write(output_path, df_extremes[:, [:run_id, :rmse, :w_eco, :w_env, :w_soc, :i_att, :adoption_threshold, :activation_prob]])
    println("Saved unfiltered SE extremes structural matrix to: $output_path")
end

function run_params(row, arch_idx::Int, n_months::Int)::Vector{Float64}
    i_att_val = (hasproperty(row, :i_att) && !ismissing(row.i_att) && !isnan(row.i_att)) ? Float64(row.i_att) : 0.3
    p = ModelParams(
        decision_architecture_idx = arch_idx,
        w_eco              = Float64(row.w_eco),
        w_env              = Float64(row.w_env),
        w_soc              = Float64(row.w_soc),
        adoption_threshold = Float64(row.adoption_threshold),
        activation_prob    = Float64(row.activation_prob),
        i_att              = i_att_val,
    )
    model = initialize_model(p)
    _, mdf = run!(model, n_months; mdata = [adoption_rate_fn], when = 12, showprogress = false)
    return mdf.adoption_rate_fn
end

function envelope(traces)
    n = minimum(length.(traces))
    mat = hcat([t[1:n] for t in traces]...)
    return vec(mean(mat, dims=2)), vec(minimum(mat, dims=2)), vec(maximum(mat, dims=2))
end

function print_simple_table(model_name, df_sorted)
    top_3 = first(df_sorted, 3)
    worst_3 = last(df_sorted, 3)
    println("\n" * "═"^100)
    println(" PARAMETERS FOR $model_name")
    println("═"^100)
    @printf("| %-10s | %-8s | %-7s | %-7s | %-7s | %-7s | %-10s | %-10s | %-7s |\n",
            "Group", "Run ID", "rmse", "w_eco", "w_env", "w_soc", "adopt_th", "act_prob", "i_att")
    println("|" * "-"^98 * "|")
    for r in eachrow(top_3)
        i_att_val = hasproperty(r, :i_att) ? r.i_att : NaN
        @printf("| %-10s | #%-7d | %-7.3f | %-7.2f | %-7.2f | %-7.2f | %-10.2f | %-10.2f | %-7.2f |\n",
                "BEST 3", r.run_id, r.rmse, r.w_eco, r.w_env, r.w_soc, r.adoption_threshold, r.activation_prob, i_att_val)
    end
    println("|" * "-"^98 * "|")
    for r in eachrow(worst_3)
        i_att_val = hasproperty(r, :i_att) ? r.i_att : NaN
        @printf("| %-10s | #%-7d | %-7.3f | %-7.2f | %-7.2f | %-7.2f | %-10.2f | %-10.2f | %-7.2f |\n",
                "WORST 3", r.run_id, r.rmse, r.w_eco, r.w_env, r.w_soc, r.adoption_threshold, r.activation_prob, i_att_val)
    end
    println("═"^100)
end

function main()
    CALIB_CSV    = "C:\\Users\\clari\\Documents\\MA3\\Project energy St Prex\\julia_clarisse\\calibration_results_total.csv"
    TOTAL_MONTHS = 180  
    PROJ_MONTHS  = 492  
    scatter!     = CairoMakie.scatter!

    OUTPUT_DIR   = "plots_output"
    if !ispath(OUTPUT_DIR)
        mkpath(OUTPUT_DIR)
        println("Created output directory: $OUTPUT_DIR")
    end

    df_res = CSV.read(CALIB_CSV, DataFrame)
    df_rr  = filter(:architecture => ==("RR"), df_res)
    df_se  = filter(:architecture => ==("SE"), df_res)

    println("Loaded $(nrow(df_rr)) RR runs, $(nrow(df_se)) SE runs")

    DF_REAL   = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
    TOTAL_BLD = nrow(DF_REAL)

    df_adopters = filter(row -> !ismissing(row.BeginningOfOperation_Year) &&
                                 row.BeginningOfOperation_Year != -99 &&
                                 row.BeginningOfOperation_Year < 2024, DF_REAL)
    hc = combine(groupby(df_adopters, :BeginningOfOperation_Year), nrow => :new_adopters)
    sort!(hc, :BeginningOfOperation_Year)
    hc.cumulative_adopters = cumsum(hc.new_adopters)
    hc.adoption_rate       = (hc.cumulative_adopters ./ TOTAL_BLD) .* 100

    YEARS_HIST = collect(2009:2023)
    real_rates = map(YEARS_HIST) do y
        idx = findlast(hc.BeginningOfOperation_Year .<= y)
        isnothing(idx) ? 0.0 : hc.adoption_rate[idx]
    end

    YEARS_PROJ = collect(2009:2050)

    df_rr_sorted = sort(df_rr, :rmse)
    df_se_sorted = sort(df_se, :rmse)

    print_simple_table("RR", df_rr_sorted)
    print_simple_table("SE", df_se_sorted)

    row_rr_best  = first(df_rr_sorted, 1)[1, :]
    row_rr_worst = last(df_rr_sorted, 1)[1, :]
    row_se_best  = first(df_se_sorted, 1)[1, :]
    row_se_worst = last(df_se_sorted, 1)[1, :]
    print_se_comprehensive_extremes(df_se_sorted)
    
    println("\nRunning all calibration parameter sets (calibration window)…")
    traces_rr = Vector{Vector{Float64}}()
    traces_se = Vector{Vector{Float64}}()

    for row in eachrow(df_rr) push!(traces_rr, run_params(row, 0, TOTAL_MONTHS)) end
    for row in eachrow(df_se) push!(traces_se, run_params(row, 1, TOTAL_MONTHS)) end

    println("\nRunning projections for best/worst…")
    trace_rr_best  = run_params(row_rr_best,  0, PROJ_MONTHS)
    trace_rr_worst = run_params(row_rr_worst, 0, PROJ_MONTHS)
    trace_se_best  = run_params(row_se_best,  1, PROJ_MONTHS)
    trace_se_worst = run_params(row_se_worst, 1, PROJ_MONTHS)

    # Projecting all paths to 2050 for baseline unweighted average extraction
    traces_rr_proj = [run_params(row, 0, PROJ_MONTHS) for row in eachrow(df_rr)]
    traces_se_proj = [run_params(row, 1, PROJ_MONTHS) for row in eachrow(df_se)]

    mean_rr, lo_rr, hi_rr = envelope(traces_rr)
    mean_se, lo_se, hi_se = envelope(traces_se)
    n_hist = min(length(mean_rr), length(YEARS_HIST))
    xs = 1:n_hist

    final_mean_2050_rr_30 = NaN; final_lo_2050_rr_30 = NaN; final_hi_2050_rr_30 = NaN
    final_mean_2050_se_30 = NaN; final_lo_2050_se_30 = NaN; final_hi_2050_se_30 = NaN

   
    # FIG 1: Generate and format RMSE distribution histograms for both RR and SE configurations
    fig1 = Figure(size = (900, 450))
    ax_rr = Axis(fig1[1, 1]; title = "rmse distribution — RR", xlabel = "rmse", ylabel = "Count")
    ax_se = Axis(fig1[1, 2]; title = "rmse distribution — SE", xlabel = "rmse", ylabel = "Count")
    all_rmse = vcat(df_rr.rmse, df_se.rmse)
    bins = 0.0:0.2:(ceil(maximum(all_rmse) / 0.2) * 0.2)
    hist!(ax_rr, df_rr.rmse; bins = collect(bins), color = (:tomato, 0.75), strokecolor = :white)
    hist!(ax_se, df_se.rmse; bins = collect(bins), color = (:royalblue, 0.75), strokecolor = :white)
    vlines!(ax_rr, [row_rr_best.rmse]; color = :darkred, linewidth = 2, linestyle = :dash, label = "Best")
    vlines!(ax_se, [row_se_best.rmse]; color = :navy,    linewidth = 2, linestyle = :dash, label = "Best")
    axislegend(ax_rr; position = :rt); axislegend(ax_se; position = :rt)
    save(joinpath(OUTPUT_DIR, "fig1_rmse_distribution.png"), fig1, px_per_unit = 2)

    # FIG 2a: Plot every calibration trace generated by the RR architecture against historical data points
    n_rr = nrow(df_rr)
    rr_colors = cgrad(:tab20, n_rr; categorical = true)
    fig2a = Figure(size = (1100, 520))
    ax2a = Axis(fig2a[1, 1]; title = "RR calibration runs", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:length(YEARS_HIST), string.(YEARS_HIST)), xticklabelrotation = 0.5)
    for (i, row) in enumerate(eachrow(df_rr))
        lines!(ax2a, xs, traces_rr[i][1:n_hist]; color = rr_colors[i], linewidth = 1.8)
    end
    scatter!(ax2a, xs, real_rates[1:n_hist]; color = :black, markersize = 8, label = "Historical data")
    lines!(ax2a, xs, real_rates[1:n_hist]; color = :black, linewidth = 2.5)
    save(joinpath(OUTPUT_DIR, "fig2a_rr_all_runs.png"), fig2a, px_per_unit = 2)

    # FIG 2b: Plot every calibration trace generated by the SE architecture against historical data points
    n_se = nrow(df_se)
    se_colors = cgrad(:tab20, n_se; categorical = true)
    fig2b = Figure(size = (1100, 520))
    ax2b = Axis(fig2b[1, 1]; title = "SE calibration runs", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:length(YEARS_HIST), string.(YEARS_HIST)), xticklabelrotation = 0.5)
    for (i, row) in enumerate(eachrow(df_se))
        lines!(ax2b, xs, traces_se[i][1:n_hist]; color = se_colors[i], linewidth = 1.8)
    end
    scatter!(ax2b, xs, real_rates[1:n_hist]; color = :black, markersize = 8, label = "Historical data")
    lines!(ax2b, xs, real_rates[1:n_hist]; color = :black, linewidth = 2.5)
    save(joinpath(OUTPUT_DIR, "fig2b_se_all_runs.png"), fig2b, px_per_unit = 2)

    # FIG 2c: Filter out poorly fit runs (RMSE >= 3.0) and display the resulting envelope area for RR
    filtered_rr_indices = findall(df_rr.rmse .< 3.0)
    fig2c = Figure(size = (800, 480))
    ax2c = Axis(fig2c[1, 1]; title = "RR filtered uncertainty (RMSE < 3.0)", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:length(YEARS_HIST), string.(YEARS_HIST)), xticklabelrotation = 0.5)
    if length(filtered_rr_indices) > 0
        filtered_traces_rr = traces_rr[filtered_rr_indices]
        for tr in filtered_traces_rr lines!(ax2c, xs, tr[1:n_hist]; color = (:tomato, 0.25), linewidth = 1.2) end
        mean_filt_rr, lo_filt_rr, hi_filt_rr = envelope(filtered_traces_rr)
        band!(ax2c, xs, lo_filt_rr[1:n_hist], hi_filt_rr[1:n_hist]; color = (:tomato, 0.15))
        mean_line_rr = lines!(ax2c, xs, mean_filt_rr[1:n_hist]; color = :darkred, linewidth = 2.8)
        hist_scatter_rr = scatter!(ax2c, xs, real_rates[1:n_hist]; color = :black, markersize = 8)
        axislegend(ax2c, [hist_scatter_rr, mean_line_rr], ["Historical data", "Filtered Mean"], position = :lt)
    end
    save(joinpath(OUTPUT_DIR, "fig2c_rr_filtered_uncertainty.png"), fig2c, px_per_unit = 2)

    # FIG 2d: Filter out poorly fit runs (RMSE >= 3.0) and display the resulting envelope area for SE
    filtered_se_indices = findall(df_se.rmse .< 3.0)
    fig2d = Figure(size = (800, 480))
    ax2d = Axis(fig2d[1, 1]; title = "SE filtered uncertainty (RMSE < 3.0)", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:length(YEARS_HIST), string.(YEARS_HIST)), xticklabelrotation = 0.5)
    if length(filtered_se_indices) > 0
        filtered_traces_se = traces_se[filtered_se_indices]
        for tr in filtered_traces_se lines!(ax2d, xs, tr[1:n_hist]; color = (:royalblue, 0.25), linewidth = 1.2) end
        mean_filt_se, lo_filt_se, hi_filt_se = envelope(filtered_traces_se)
        band!(ax2d, xs, lo_filt_se[1:n_hist], hi_filt_se[1:n_hist]; color = (:royalblue, 0.15))
        mean_line_se = lines!(ax2d, xs, mean_filt_se[1:n_hist]; color = :navy, linewidth = 2.8)
        hist_scatter_se = scatter!(ax2d, xs, real_rates[1:n_hist]; color = :black, markersize = 8)
        axislegend(ax2d, [hist_scatter_se, mean_line_se], ["Historical data", "Filtered Mean"], position = :lt)
    else
        text!(ax2d, length(YEARS_HIST)/2, 50.0; text = "No SE runs with RMSE < 3.0", align = (:center, :center), fontsize = 14)
    end
    save(joinpath(OUTPUT_DIR, "fig2d_se_filtered_uncertainty.png"), fig2d, px_per_unit = 2)

    # FIG 3: Directly contrast the trajectories of absolute best and worst calibrations for both architectures
    fig3 = Figure(size = (950, 450))
    ax3_rr = Axis(fig3[1, 1]; title = "Best vs Worst calibrations— RR", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:n_hist, string.(YEARS_HIST[1:n_hist])), xticklabelrotation = 0.5)
    ax3_se = Axis(fig3[1, 2]; title = "Best vs Worst calibrations — SE", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:n_hist, string.(YEARS_HIST[1:n_hist])), xticklabelrotation = 0.5)
    
    lines!(ax3_rr, xs, trace_rr_best[1:n_hist]; color = :tomato, linewidth = 2.5, label = "Best (rmse=$(round(row_rr_best.rmse, digits=3)))")
    lines!(ax3_rr, xs, trace_rr_worst[1:n_hist]; color = (:tomato, 0.45), linewidth = 2.0, linestyle = :dash, label = "Worst (rmse=$(round(row_rr_worst.rmse, digits=3)))")
    scatter!(ax3_rr, xs, real_rates[1:n_hist]; color = :black, markersize = 7)
    lines!(ax3_rr, xs, real_rates[1:n_hist]; color = :black, linewidth = 2)
    axislegend(ax3_rr; position = :lt)

    lines!(ax3_se, xs, trace_se_best[1:n_hist]; color = :royalblue, linewidth = 2.5, label = "Best (rmse=$(round(row_se_best.rmse, digits=3)))")
    lines!(ax3_se, xs, trace_se_worst[1:n_hist]; color = (:royalblue, 0.45), linewidth = 2.0, linestyle = :dash, label = "Worst (rmse=$(round(row_se_worst.rmse, digits=3)))")
    scatter!(ax3_se, xs, real_rates[1:n_hist]; color = :black, markersize = 7)
    lines!(ax3_se, xs, real_rates[1:n_hist]; color = :black, linewidth = 2)
    axislegend(ax3_se; position = :lt)
    save(joinpath(OUTPUT_DIR, "fig3_best_vs_worst_calibration.png"), fig3, px_per_unit = 2)

    # FIG 3a: Isolated layout highlighting exclusively the top fits against historical data
    fig3a = Figure(size = (950, 450))
    ax3a_rr = Axis(fig3a[1, 1]; title = "Best Calibrations — RR", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:n_hist, string.(YEARS_HIST[1:n_hist])), xticklabelrotation = 0.5)
    ax3a_se = Axis(fig3a[1, 2]; title = "Best Calibrations — SE", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:n_hist, string.(YEARS_HIST[1:n_hist])), xticklabelrotation = 0.5)
    
    lines!(ax3a_rr, xs, trace_rr_best[1:n_hist]; color = :tomato, linewidth = 2.5, label = "Best (rmse=$(round(row_rr_best.rmse, digits=3)))")
    scatter!(ax3a_rr, xs, real_rates[1:n_hist]; color = :black, markersize = 7)
    lines!(ax3a_rr, xs, real_rates[1:n_hist]; color = :black, linewidth = 2)
    axislegend(ax3a_rr; position = :lt)

    lines!(ax3a_se, xs, trace_se_best[1:n_hist]; color = :royalblue, linewidth = 2.5, label = "Best (rmse=$(round(row_se_best.rmse, digits=3)))")
    scatter!(ax3a_se, xs, real_rates[1:n_hist]; color = :black, markersize = 7)
    lines!(ax3a_se, xs, real_rates[1:n_hist]; color = :black, linewidth = 2)
    axislegend(ax3a_se; position = :lt)
    save(joinpath(OUTPUT_DIR, "fig3a_best_calibration.png"), fig3a, px_per_unit = 2)

    # FIG 4: Plot long-term projection trajectories up to 2050 for the extremes (Best vs Worst)
    fig4 = Figure(size = (950, 480))
    ax4_rr = Axis(fig4[1, 1]; title = "Best vs Worst Proj — RR", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), limits = (nothing, nothing, 0, 100))
    ax4_se = Axis(fig4[1, 2]; title = "Best vs Worst Proj — SE", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), limits = (nothing, nothing, 0, 100))
    n_proj = length(YEARS_PROJ)
    
    lines!(ax4_rr, 1:min(n_proj, length(trace_rr_best)), trace_rr_best; color = :tomato, linewidth = 2.5, label = "Best")
    lines!(ax4_rr, 1:min(n_proj, length(trace_rr_worst)), trace_rr_worst; color = (:tomato, 0.45), linewidth = 2.0, linestyle = :dash, label = "Worst")
    scatter!(ax4_rr, 1:length(YEARS_HIST), real_rates; color = :black, markersize = 6)
    vlines!(ax4_rr, [length(YEARS_HIST)]; color = :gray50, linestyle = :dot)
    axislegend(ax4_rr; position = :lt)

    lines!(ax4_se, 1:min(n_proj, length(trace_se_best)), trace_se_best; color = :royalblue, linewidth = 2.5, label = "Best")
    lines!(ax4_se, 1:min(n_proj, length(trace_se_worst)), trace_se_worst; color = (:royalblue, 0.45), linewidth = 2.0, linestyle = :dash, label = "Worst")
    scatter!(ax4_se, 1:length(YEARS_HIST), real_rates; color = :black, markersize = 6)
    vlines!(ax4_se, [length(YEARS_HIST)]; color = :gray50, linestyle = :dot)
    axislegend(ax4_se; position = :lt)
    save(joinpath(OUTPUT_DIR, "fig4_best_vs_worst_projection.png"), fig4, px_per_unit = 2)

    # FIG 5b: Render bounded projection uncertainty spaces up to 2050 using only runs where RMSE < 3.0
    println("\nGenerating Figure 5b: Projection uncertainty for runs with RMSE < 3.0...")
    filtered_rr_proj_idx_3 = findall(df_rr.rmse .< 3.0)
    filtered_se_proj_idx_3 = findall(df_se.rmse .< 3.0)
    
    fig5b = Figure(size = (1000, 480))
    ax5b_rr = Axis(fig5b[1, 1]; title = "RR Proj Uncertainty (RMSE < 3.0)", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), limits = (nothing, nothing, 0, 100))
    ax5b_se = Axis(fig5b[1, 2]; title = "SE Proj Uncertainty (RMSE < 3.0)", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), limits = (nothing, nothing, 0, 100))

    for (ax, filtered_idx, arch_idx, col, name) in [(ax5b_rr, filtered_rr_proj_idx_3, 0, :tomato, "RR"), (ax5b_se, filtered_se_proj_idx_3, 1, :royalblue, "SE")]
        if length(filtered_idx) > 0
            filtered_proj_traces = Vector{Vector{Float64}}()
            df_curr = name == "RR" ? df_rr : df_se
            for idx in filtered_idx
                push!(filtered_proj_traces, run_params(df_curr[idx, :], arch_idx, PROJ_MONTHS))
            end
            xs_p = 1:length(YEARS_PROJ)
            for tr in filtered_proj_traces lines!(ax, xs_p, tr; color = (col, 0.15), linewidth = 1.0) end
            mean_p, lo_p, hi_p = envelope(filtered_proj_traces)
            band!(ax, xs_p, lo_p, hi_p; color = (col, 0.12))
            mean_line = lines!(ax, xs_p, mean_p; color = col, linewidth = 2.8)
            hist_scatter = scatter!(ax, 1:length(YEARS_HIST), real_rates; color = :black, markersize = 6)
            vlines!(ax, [length(YEARS_HIST)]; color = :gray50, linestyle = :dot)
            axislegend(ax, [hist_scatter, mean_line], ["Historical data", "Filtered Mean"], position = :lt)
            
            if name == "RR"
                final_mean_2050_rr_30 = last(mean_p); final_lo_2050_rr_30 = last(lo_p); final_hi_2050_rr_30 = last(hi_p)
            else
                final_mean_2050_se_30 = last(mean_p); final_lo_2050_se_30 = last(lo_p); final_hi_2050_se_30 = last(hi_p)
            end
        else
            text!(ax, length(YEARS_PROJ)/2, 50.0; text = "No $name runs with RMSE < 3.0", align = (:center, :center), fontsize = 14)
        end
    end
    save(joinpath(OUTPUT_DIR, "fig5b_projections_uncertainty_rmse_3.png"), fig5b, px_per_unit = 2)
    println("Saved: fig5b_projections_uncertainty_rmse_3.png")

    # FIG 5c: Display individualized 2050 projections for each member of the top 3 calibration configurations
    println("\nGenerating Figure 5c: Projections for the Top 3 Best Calibration runs...")
    fig5c = Figure(size = (1000, 480))
    ax5c_rr = Axis(fig5c[1, 1]; title = "Top 3 Projections — RR", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), limits = (nothing, nothing, 0, 100))
    ax5c_se = Axis(fig5c[1, 2]; title = "Top 3 Projections — SE", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), limits = (nothing, nothing, 0, 100))

    xs_p = 1:length(YEARS_PROJ)
    rr_top3_colors = [:darkred, :firebrick, :indianred]
    se_top3_colors = [:navy, :royalblue, :cornflowerblue]

    top3_rr_runs = []; top3_se_runs = []

    for i in 1:min(3, nrow(df_rr_sorted))
        row = df_rr_sorted[i, :]
        trace = run_params(row, 0, PROJ_MONTHS)
        push!(top3_rr_runs, (idx=i, rmse=row.rmse, run_id=row.run_id, val_2050=last(trace)))
        lines!(ax5c_rr, xs_p, trace; color = rr_top3_colors[i], linewidth = 2.5,
               label = "Rank $i (RMSE=$(round(row.rmse, digits=3)))")
    end
    scatter!(ax5c_rr, 1:length(YEARS_HIST), real_rates; color = :black, markersize = 6)
    vlines!(ax5c_rr, [length(YEARS_HIST)]; color = :gray50, linestyle = :dot)
    axislegend(ax5c_rr; position = :lt)

    for i in 1:min(3, nrow(df_se_sorted))
        row = df_se_sorted[i, :]
        trace = run_params(row, 1, PROJ_MONTHS)
        push!(top3_se_runs, (idx=i, rmse=row.rmse, run_id=row.run_id, val_2050=last(trace)))
        lines!(ax5c_se, xs_p, trace; color = se_top3_colors[i], linewidth = 2.5,
               label = "Rank $i (RMSE=$(round(row.rmse, digits=3)))")
    end
    scatter!(ax5c_se, 1:length(YEARS_HIST), real_rates; color = :black, markersize = 6)
    vlines!(ax5c_se, [length(YEARS_HIST)]; color = :gray50, linestyle = :dot)
    axislegend(ax5c_se; position = :lt)

    save(joinpath(OUTPUT_DIR, "fig5c_top3_projections.png"), fig5c, px_per_unit = 2)
    println("Saved: fig5c_top3_projections.png")

    # FIG 6: Isolated layout comparing the primary single best fitting execution profile for both groups
    println("\nGenerating Figure 6: Best projections 2050 only (Smallest RMSE)...")
    fig6 = Figure(size = (950, 480))
    ax6_rr = Axis(fig6[1, 1]; title = "Best projection 2050 — RR", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), xticklabelrotation = 0.5, limits = (nothing, nothing, 0, 100))
    ax6_se = Axis(fig6[1, 2]; title = "Best projection 2050 — SE", xlabel = "Year", ylabel = "Adoption rate (%)", xticks = (1:10:length(YEARS_PROJ), string.(YEARS_PROJ[1:10:end])), xticklabelrotation = 0.5, limits = (nothing, nothing, 0, 100))

    xs_hist_dots = 1:length(real_rates)

    xs_p_rr = 1:min(length(YEARS_PROJ), length(trace_rr_best))
    lines!(ax6_rr, xs_p_rr, trace_rr_best[xs_p_rr]; color = :tomato, linewidth = 3.0, label = "Best Fit (RMSE=$(round(row_rr_best.rmse, digits=3)))")
    scatter!(ax6_rr, xs_hist_dots, real_rates; color = :black, markersize = 6, label = "Historical data")
    vlines!(ax6_rr, [length(YEARS_HIST)]; color = :gray50, linewidth = 1.5, linestyle = :dot)
    axislegend(ax6_rr; position = :lt)

    xs_p_se = 1:min(length(YEARS_PROJ), length(trace_se_best))
    lines!(ax6_se, xs_p_se, trace_se_best[xs_p_se]; color = :royalblue, linewidth = 3.0, label = "Best Fit (RMSE=$(round(row_se_best.rmse, digits=3)))")
    scatter!(ax6_se, xs_hist_dots, real_rates; color = :black, markersize = 6, label = "Historical data")
    vlines!(ax6_se, [length(YEARS_HIST)]; color = :gray50, linewidth = 1.5, linestyle = :dot)
    axislegend(ax6_se; position = :lt)

    save(joinpath(OUTPUT_DIR, "fig6_best_projections_only.png"), fig6, px_per_unit = 2)
    println("Saved: fig6_best_projections_only.png")

 
    # Compute unweighted global averages at 2050 index
    unweighted_avg_2050_rr = mean([last(t) for t in traces_rr_proj])
    unweighted_avg_2050_se = mean([last(t) for t in traces_se_proj])

end

main()

