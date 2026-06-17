"""
qmc_sensitivity_analysis.jl

"""

using CSV
using DataFrames
using Statistics
using GlobalSensitivity
using QuasiMonteCarlo
using CairoMakie
using Printf

include("types.jl")
include("model.jl")


const DF_REAL   = CSV.read("StPrex_Buildings_with_PV_Potential_and_HS.csv", DataFrame)
const TOTAL_BLD = nrow(DF_REAL)

adoption_rate_fn(model) =
    (count(a -> a.pv_installed, allagents(model)) / nagents(model)) * 100

function gsa_interface_rr(X_matrix)
    n_sims  = size(X_matrix, 2)
    results = zeros(n_sims)
    for i in 1:n_sims
        w_e, w_n = X_matrix[1, i], X_matrix[2, i]
        total_w  = w_e + w_n
        if total_w > 0.8
            w_e = (w_e / total_w) * 0.8
            w_n = (w_n / total_w) * 0.8
        end
        w_s = 1.0 - (w_e + w_n)
        params = ModelParams(
            decision_architecture_idx = 0,
            w_eco = w_e, w_env = w_n, w_soc = w_s,
            adoption_threshold = X_matrix[3, i],
            activation_prob    = X_matrix[4, i]
        )
        model = initialize_model(params)
        _, mdf = run!(model, 180; mdata = [adoption_rate_fn], when = 12, showprogress = false)
        results[i] = mdf.adoption_rate_fn[end]
    end
    return results
end

function gsa_interface_se(X_matrix)
    n_sims  = size(X_matrix, 2)
    results = zeros(n_sims)
    for i in 1:n_sims
        params = ModelParams(
            decision_architecture_idx = 1,
            w_eco              = X_matrix[1, i],
            w_env              = X_matrix[2, i],
            w_soc              = X_matrix[3, i],
            adoption_threshold = X_matrix[4, i],
            activation_prob    = X_matrix[5, i],
            i_att              = X_matrix[6, i]
        )
        model = initialize_model(params)
        _, mdf = run!(model, 180; mdata = [adoption_rate_fn], when = 12, showprogress = false)
        results[i] = mdf.adoption_rate_fn[end]
    end
    return results
end


const N_SAMPLES  = 200

const LB_RR     = [0.1,  0.1,  0.4,  0.001]
const UB_RR     = [0.9,  0.9,  0.9,  0.05 ]
const LABELS_RR = ["w_eco", "w_env", "threshold", "activation_prob"]

const LB_SE     = [0.1,  0.1,  0.1,  0.2,  0.001, 0.3]
const UB_SE     = [0.6,  0.6,  0.6,  0.7,  0.010, 0.8]
const LABELS_SE = ["w_eco", "w_env", "w_soc", "threshold", "activation_prob", "i_att"]


println("=== PREFLIGHT CHECK ===")


RANDOMIZER = let
    lb_t = [0.0, 0.0]; ub_t = [1.0, 1.0]
    # Build candidates by probing the module at runtime — avoids import errors
    _qmc = QuasiMonteCarlo
    candidates = Tuple{String, Function}[]
    for (name, expr) in [
            ("QuasiMonteCarlo.MatousekScrambling()", "QuasiMonteCarlo.MatousekScrambling()"),
            ("QuasiMonteCarlo.Shift()",              "QuasiMonteCarlo.Shift()"),
        ]
        try
            r = eval(Meta.parse(expr))
            push!(candidates, (name, () -> eval(Meta.parse(expr))))
        catch
        end
    end
    push!(candidates, ("nothing (no randomizer)", () -> nothing))
    chosen = nothing
    for (name, builder) in candidates
        try
            r = builder()
            if isnothing(r)
                QuasiMonteCarlo.generate_design_matrices(4, lb_t, ub_t, SobolSample())
            else
                QuasiMonteCarlo.generate_design_matrices(4, lb_t, ub_t, SobolSample(), r)
            end
            println("  [OK] Randomizer: $name")
            chosen = (name, builder)
            break
        catch
            continue
        end
    end
    isnothing(chosen) && error("PREFLIGHT FAILED — no working randomizer found for generate_design_matrices")
    chosen
end

# helper to call generate_design_matrices with or without randomizer
function gen_matrices(n, lb, ub)
    rname, rbuild = RANDOMIZER
    r = rbuild()
    if isnothing(r)
        return QuasiMonteCarlo.generate_design_matrices(n, lb, ub, SobolSample())
    else
        return QuasiMonteCarlo.generate_design_matrices(n, lb, ub, SobolSample(), r)
    end
end

# 4b. Détecter les noms de champs du SobolResult
get_S1, get_ST = let
    dummy_fn(X) = vec(sum(X, dims=1))
    lb_t = [0.0, 0.0]; ub_t = [1.0, 1.0]
    A_t, B_t = gen_matrices(8, lb_t, ub_t)
    res_t = gsa(dummy_fn, Sobol(), A_t, B_t; batch = true)
    fields = fieldnames(typeof(res_t))
    println("  [INFO] SobolResult fields: $fields")

    if :S1 in fields && :ST in fields
        println("  [OK] Using .S1 and .ST")
        (r -> r.S1), (r -> r.ST)
    elseif :first_order in fields && :total_order in fields
        println("  [OK] Using .first_order and .total_order (older API)")
        (r -> r.first_order), (r -> r.total_order)
    else
        error("PREFLIGHT FAILED — unknown SobolResult field names: $fields")
    end
end

println("  [OK] All preflight checks passed\n")

println("=== Sobol QMC Sensitivity Analysis ===\n")
println("Generating design matrices…")

A_rr, B_rr = gen_matrices(N_SAMPLES, LB_RR, UB_RR)
A_se, B_se = gen_matrices(N_SAMPLES, LB_SE, UB_SE)

println("Running GSA — architecture RR  ($(2*N_SAMPLES) model evals)…")
res_rr = gsa(gsa_interface_rr, Sobol(order=[0,1,2]), A_rr, B_rr; batch = true)

println("Running GSA — architecture SE  ($(2*N_SAMPLES) model evals)…")
res_se = gsa(gsa_interface_se, Sobol(order=[0,1,2]), A_se, B_se; batch = true)

# ─── 7. Affichage terminal ────────────────────────────────────────────────────

function print_indices(labels, s1, st, arch)
    println("\nResults — $arch :")
    for i in eachindex(labels)
        @printf("  %-18s  S1 = %+.4f   ST = %+.4f\n", labels[i], s1[i], st[i])
    end
end

print_indices(LABELS_RR, get_S1(res_rr), get_ST(res_rr), "RR")
print_indices(LABELS_SE, get_S1(res_se), get_ST(res_se), "SE")


let
    function bar_panel(ax, labels, s1, st, title)
        n  = length(labels)
        xs = 1:n
        barplot!(ax, xs .- 0.2, s1; width = 0.35, color = (:lightblue, 0.9),
                 label = "S1 (first-order)")
        barplot!(ax, xs .+ 0.2, st; width = 0.35, color = (:steelblue4, 0.9),
                 label = "ST (total effect)")
        hlines!(ax, [0.0]; color = :black, linewidth = 0.8)
        ax.title  = title
        ax.xlabel = "Parameter"
        ax.ylabel = "Sensitivity index"
        ax.xticks = (collect(xs), labels)
        ylims!(ax, -0.05, 1.0)
        axislegend(ax; position = :rt)
    end

    fig = Figure(size = (950, 750))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[2, 1])
    bar_panel(ax1, LABELS_RR, get_S1(res_rr), get_ST(res_rr),
              "Sobol sensitivity — Architecture RR")
    bar_panel(ax2, LABELS_SE, get_S1(res_se), get_ST(res_se),
              "Sobol sensitivity — Architecture SE")
    save("qmc_sobol_sensitivity_analysis_200_runs.png", fig, px_per_unit = 2)
    println("\nSaved: qmc_sobol_sensitivity_analysis_200_runs.png")
end

# ─── 9. Graphique S2 (second-order) ──────────────────────────────────────────

let
    function s2_heatmap(fig_pos, labels, s2_mat, title)
        n   = length(labels)
        # S2 is upper-triangular; mirror it for display
        mat = zeros(n, n)
        for i in 1:n, j in 1:n
            if i < j && s2_mat !== nothing
                v = s2_mat[i, j]
                mat[i, j] = isnan(v) ? 0.0 : v
                mat[j, i] = mat[i, j]
            end
        end
        ax = Axis(fig_pos;
            title  = title,
            xticks = (1:n, labels),
            yticks = (1:n, labels),
            xticklabelrotation = 0.4)
        hm = heatmap!(ax, mat; colormap = :Blues, colorrange = (0, 0.3))
        Colorbar(fig_pos[1, 2], hm; label = "S2")
        return ax
    end

    s2_rr = get_S1(res_rr) !== nothing ? res_rr.S2 : nothing
    s2_se = get_S1(res_se) !== nothing ? res_se.S2 : nothing

    if s2_rr !== nothing && s2_se !== nothing
        fig2 = Figure(size = (1000, 500))
        s2_heatmap(fig2[1, 1][1, 1], LABELS_RR, s2_rr,
                   "S2 second-order — RR")
        s2_heatmap(fig2[1, 2][1, 1], LABELS_SE, s2_se,
                   "S2 second-order — SE")
        save("qmc_sobol_s2_heatmap_200_runs.png", fig2, px_per_unit = 2)
        println("Saved: qmc_sobol_s2_heatmap_200_runs.png")

        println("\n[S2 — RR]")
        n = length(LABELS_RR)
        for i in 1:n, j in i+1:n
            v = s2_rr[i,j]
            isnan(v) || @printf("  %-12s × %-12s  S2 = %.4f\n", LABELS_RR[i], LABELS_RR[j], v)
        end
        println("\n[S2 — SE]")
        n = length(LABELS_SE)
        for i in 1:n, j in i+1:n
            v = s2_se[i,j]
            isnan(v) || @printf("  %-12s × %-12s  S2 = %.4f\n", LABELS_SE[i], LABELS_SE[j], v)
        end
    else
        println("S2 not available (rerun with Sobol(order=[0,1,2]) to enable)")
    end
end