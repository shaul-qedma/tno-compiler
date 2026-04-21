"""Generate the 8-plot deck summarizing what we've learned about the compiler.

Each plot is self-contained: title, axes, and a plain-language note box
explaining what the reader should take away. No physics jargon in the
explanations.

Output: docs/plots/*.png
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("docs/data")
OUT = Path("docs/plots")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def load(path, int_cols=(), float_cols=(), str_cols=()):
    rows = []
    with open(DATA / path) as f:
        for r in csv.DictReader(f):
            out = {}
            for c in int_cols: out[c] = int(r[c])
            for c in float_cols: out[c] = float(r[c])
            for c in str_cols: out[c] = r[c]
            rows.append(out)
    return rows


def annotate(ax, text, loc="upper left"):
    """Standard explanation box: bordered, readable, in corner."""
    positions = {
        "upper left": (0.02, 0.98, "top", "left"),
        "upper right": (0.98, 0.98, "top", "right"),
        "lower left": (0.02, 0.02, "bottom", "left"),
        "lower right": (0.98, 0.02, "bottom", "right"),
    }
    x, y, va, ha = positions[loc]
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=9, verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", edgecolor="gray", alpha=0.95))


# ============================================================================
# Load + merge the compression grid
# ============================================================================

g1 = load("compression_grid.csv",
          int_cols=("n", "steps", "target_depth", "ansatz_depth", "seeds_agree"),
          float_cols=("g", "max_td", "mean_td", "compile_err_min",
                       "compile_err_max", "compile_err_spread"))
g2 = load("compression_grid2.csv",
          int_cols=("n", "steps", "target_depth", "ansatz_depth", "seeds_agree"),
          float_cols=("g", "max_td", "mean_td", "compile_err_min",
                       "compile_err_max", "compile_err_spread"))
seen = set()
grid = []
for r in g2 + g1:
    key = (r["n"], r["g"], r["steps"], r["ansatz_depth"])
    if key in seen: continue
    seen.add(key)
    grid.append(r)

print(f"Loaded {len(grid)} unique grid rows")


# ============================================================================
# Plot 1 — Minimum compiled-circuit depth vs target evolution length
# ============================================================================

def plot1_min_depth():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    steps_vals = sorted({r["steps"] for r in grid})
    for g, color, marker in [(0.3, "C0", "o"), (1.0, "C1", "s"), (1.5, "C2", "^")]:
        min_d_per_s = []
        s_used = []
        for s in steps_vals:
            candidates = [r["ansatz_depth"] for r in grid
                          if r["g"] == g and r["steps"] == s and r["max_td"] < 0.05]
            if candidates:
                min_d_per_s.append(min(candidates))
                s_used.append(s)
        ax.plot(s_used, min_d_per_s, "-"+marker, color=color, ms=8,
                label=f"Target coupling g={g}", linewidth=2)

    # Rule-of-thumb reference
    s_ref = np.array(steps_vals)
    rule = np.maximum(2, np.ceil(s_ref / 2))
    ax.plot(s_ref, rule, "--", color="gray", label="Naive rule: ⌈steps/2⌉", linewidth=1.5)

    ax.set_xlabel("Number of Trotter steps in target circuit\n"
                  "(roughly, how long an evolution we're simulating)")
    ax.set_ylabel("Minimum compiled circuit depth needed\n"
                  "(for < 5% worst-case error)")
    ax.set_title("How shallow can we make the compiled circuit?")
    ax.legend(loc="upper left")
    ax.set_xticks(steps_vals)

    annotate(ax,
             "Plain reading:\n"
             "• Each dot = smallest depth we tried that got the error\n"
             "  under 5%. Target circuit has 4× as many layers as the\n"
             "  number on the x-axis.\n"
             "• For easy ('gapped') targets, we compress much more\n"
             "  than the naive rule predicts — a depth-3 circuit suffices\n"
             "  even for 16-step targets (about 64 layers).\n"
             "• For harder targets (critical or paramagnetic), the rule\n"
             "  is closer to right but still pessimistic.",
             loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "01_min_depth_vs_steps.png")
    plt.close(fig)


plot1_min_depth()
print("  [1/8] 01_min_depth_vs_steps.png")


# ============================================================================
# Plot 2 — How accuracy scales with the number of qubits
# ============================================================================

def plot2_n_scaling():
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for s, color in zip([2, 4, 6, 8, 12, 16], ["C0", "C1", "C2", "C3", "C4", "C5"]):
        ad = max(2, math.ceil(s / 2))
        pts = [(r["n"], r["max_td"]) for r in grid
               if r["g"] == 0.3 and r["steps"] == s and r["ansatz_depth"] == ad]
        if len(pts) < 2: continue
        pts.sort()
        ns = [p[0] for p in pts]
        tds = [p[1] for p in pts]
        ax.plot(ns, tds, "-o", color=color, ms=8,
                label=f"{s} Trotter steps (compiled depth {ad})")

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Worst-case output error\n(max trace distance over 10 random inputs)")
    ax.set_title("Accuracy stays bounded as the system grows")
    ax.set_yscale("log")
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1.01, 1))

    fig.subplots_adjust(bottom=0.25)
    fig.text(0.02, 0.02,
             "Plain reading:  Each line fixes a target (easy 'gapped' regime, specific evolution length) and varies the number of qubits from 4 to 12.\n"
             "Error grows only mildly with system size — typically 1.5–2× from 4 → 12 qubits. This is consistent with the idea that short-time quantum\n"
             "simulations on gapped models do NOT get exponentially harder as we add qubits — a nontrivial property of the compiler.",
             fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.savefig(OUT / "02_n_scaling.png", bbox_inches="tight")
    plt.close(fig)


plot2_n_scaling()
print("  [2/8] 02_n_scaling.png")


# ============================================================================
# Plot 3 — Compile landscape heatmap over (steps, ansatz_d)
# ============================================================================

def plot3_landscape():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    gs_used = sorted({r["g"] for r in grid})
    steps_vals = sorted({r["steps"] for r in grid})
    ad_vals = sorted({r["ansatz_depth"] for r in grid if r["ansatz_depth"] <= 12})

    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=1e-3, vmax=1.0)

    for ax, g in zip(axes, gs_used):
        Z = np.full((len(ad_vals), len(steps_vals)), np.nan)
        for i, ad in enumerate(ad_vals):
            for j, s in enumerate(steps_vals):
                tds = [r["max_td"] for r in grid
                       if r["g"] == g and r["steps"] == s
                       and r["ansatz_depth"] == ad]
                if tds:
                    Z[i, j] = np.median(tds)
        im = ax.pcolormesh(
            np.arange(len(steps_vals)+1) - 0.5,
            np.arange(len(ad_vals)+1) - 0.5,
            Z, cmap="viridis", norm=norm, shading="auto")
        ax.set_xticks(range(len(steps_vals)))
        ax.set_xticklabels(steps_vals)
        ax.set_yticks(range(len(ad_vals)))
        ax.set_yticklabels(ad_vals)
        ax.set_xlabel("Target evolution length (Trotter steps)")
        label = {0.3: "Gapped (easy)", 1.0: "Critical (harder)",
                 1.5: "Paramagnetic"}[g]
        ax.set_title(f"{label}  —  g={g}")
        # Annotate the rule line
        s_arr = np.arange(len(steps_vals))
        rule_d = [max(2, math.ceil(s/2)) for s in steps_vals]
        rule_y = [ad_vals.index(d) if d in ad_vals else None for d in rule_d]
        s_plot = [s for s, y in zip(s_arr, rule_y) if y is not None]
        y_plot = [y for y in rule_y if y is not None]
        ax.plot(s_plot, y_plot, "w--", linewidth=2,
                label="Naive rule ⌈s/2⌉")
        ax.legend(loc="upper left", fontsize=8, facecolor="white", framealpha=0.8)

    axes[0].set_ylabel("Compiled circuit depth")
    fig.suptitle("Where does the compilation succeed? "
                 "(color: typical worst-case error, lower is better)",
                 y=1.02, fontsize=13)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                        extend="both")
    cbar.set_label("Worst-case output error\n(log scale)")

    # Explanation beneath the figure
    fig.text(0.01, -0.12,
             "Plain reading:\n"
             "  • Each cell = median accuracy over all qubit counts we tested (4–12 qubits).\n"
             "  • Dark = accurate. Bright yellow = error near 100%.\n"
             "  • Three panels, three difficulties of target. The easier the target (left),\n"
             "    the shorter the compiled circuit can be. The dashed line is a naive\n"
             "    rule-of-thumb prediction; the compiler consistently does better than this line.",
             fontsize=9, wrap=True,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.tight_layout()
    fig.savefig(OUT / "03_landscape.png", bbox_inches="tight")
    plt.close(fig)


plot3_landscape()
print("  [3/8] 03_landscape.png")


# ============================================================================
# Plot 4 — Frobenius vs max_td (Frobenius as channel-distance proxy)
# ============================================================================

def plot4_frobenius_vs_td():
    wva = load("worst_vs_avg.csv",
               int_cols=("n", "steps", "ansatz_d", "circuit_idx"),
               float_cols=("g", "frobenius_norm", "mean_td", "max_td"))
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ns = sorted({r["n"] for r in wva})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ns)))
    for n, color in zip(ns, colors):
        xs = [r["frobenius_norm"] for r in wva if r["n"] == n]
        ys = [r["max_td"] for r in wva if r["n"] == n]
        ax.scatter(xs, ys, color=color, s=60, alpha=0.85, edgecolor="black",
                   label=f"{n} qubits")

    # y=x reference
    xspan = np.logspace(-3.5, 0.1, 50)
    ax.plot(xspan, xspan, "k--", alpha=0.5, label="y = x (exact match)")
    ax.plot(xspan, 1.2*xspan, "k:", alpha=0.4, label="y = 1.2·x")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Cheap matrix-distance estimate\n"
                  "(normalized Frobenius distance ‖U − V‖ / √dim)")
    ax.set_ylabel("Worst-case channel error\n"
                  "(max trace distance over 30 random inputs)")
    ax.set_title("The cheap estimate tracks the real worst-case error tightly")
    ax.legend(loc="upper left", fontsize=9)

    annotate(ax,
             "Plain reading:\n"
             "• Points span six orders of magnitude in compile quality\n"
             "  (from very accurate to nearly-failed).\n"
             "• Across the range, the 'worst-case channel error' is\n"
             "  within ~20% of a simple matrix-distance calculation.\n"
             "• Implication: we can monitor the compiler's accuracy\n"
             "  using its own internal metric — no need for expensive\n"
             "  channel-level sampling during tuning.",
             loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "04_frobenius_vs_td.png")
    plt.close(fig)


plot4_frobenius_vs_td()
print("  [4/8] 04_frobenius_vs_td.png")


# ============================================================================
# Plot 5 — max_td / mean_td ratio distribution
# ============================================================================

def plot5_max_mean_ratio():
    wva = load("worst_vs_avg.csv",
               int_cols=("n", "steps", "ansatz_d", "circuit_idx"),
               float_cols=("g", "frobenius_norm", "mean_td", "max_td",
                            "max_over_mean"))
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = [("Very accurate\n(error < 1%)", 0, 0.01),
            ("Accurate\n(1-5% error)", 0.01, 0.05),
            ("Moderate\n(5-20% error)", 0.05, 0.2),
            ("Bad\n(>20% error)", 0.2, 2.0)]
    parts = []
    labels = []
    for label, lo, hi in bins:
        vals = [r["max_over_mean"] for r in wva
                if lo <= r["frobenius_norm"] < hi]
        if vals:
            parts.append(vals)
            labels.append(f"{label}\n(n={len(vals)})")
    vp = ax.violinplot(parts, showmedians=True, widths=0.7)
    for body in vp["bodies"]:
        body.set_facecolor("C0")
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("(worst-case error) / (typical error)\n"
                  "random input-state samples")
    ax.set_title("Worst-case error is always close to typical error,\n"
                 "regardless of compile quality")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.4)

    annotate(ax,
             "Plain reading:\n"
             "• For each compiled circuit we measured both the\n"
             "  worst-case error (max over 30 random inputs) and\n"
             "  the typical error (mean over the same 30).\n"
             "• Their ratio is almost always between 1.0 and 1.2,\n"
             "  even when the circuit is near-useless (right-most box).\n"
             "• Interpretation: the compile doesn't have 'weak spots'\n"
             "  that fail catastrophically on specific inputs.",
             loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "05_max_mean_ratio.png")
    plt.close(fig)


plot5_max_mean_ratio()
print("  [5/8] 05_max_mean_ratio.png")


# ============================================================================
# Plot 6 — Ensemble vs single-circuit comparison
# ============================================================================

def plot6_ensemble_vs_single():
    evs = load("ensemble_vs_single.csv",
               int_cols=("n", "steps", "ansatz_d"),
               float_cols=("g", "ensemble_max_td", "single0_max_td",
                            "single1_max_td", "single2_max_td",
                            "best_single_max_td"),
               str_cols=("label",))
    fig, ax = plt.subplots(figsize=(12, 6.5))

    x = np.arange(len(evs))
    width = 0.25
    worst = [max(r["single0_max_td"], r["single1_max_td"], r["single2_max_td"])
             for r in evs]
    best = [r["best_single_max_td"] for r in evs]
    ens = [r["ensemble_max_td"] for r in evs]

    ax.bar(x - width, worst, width, label="Worst of 3 candidates",
           color="C3", alpha=0.85)
    ax.bar(x, best, width, label="Best of 3 candidates",
           color="C2", alpha=0.85)
    ax.bar(x + width, ens, width, label="Weighted ensemble (QP)",
           color="C0", alpha=0.85, edgecolor="black", linewidth=1.2)

    # Short two-line labels to avoid rotation and clipping
    def short_label(r):
        return f"n={r['n']}\ng={r['g']:.1f}\nsteps={r['steps']}"
    labels = [short_label(r) for r in evs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2.0)
    ax.set_ylabel("Worst-case output error\n(log scale)")
    ax.set_title("The 'smart combination' of candidates (QP) just picks the winner —\n"
                 "it never does meaningfully better than the best single candidate")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    fig.subplots_adjust(bottom=0.32)
    fig.text(0.02, 0.02,
             "Plain reading:  For each target (x-axis), we compiled 3 candidates from different random starts.\n"
             "• RED bar — the worst of the 3; typically 10–250× worse than the best. Some random starts land in bad basins.\n"
             "• GREEN bar — the best of the 3.\n"
             "• BLUE bar — our 'optimal weighted combination' of all three. It tracks green almost exactly, meaning the\n"
             "   weighting mechanism's only real job is picking the best single candidate. No cooperative interference benefit.",
             fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.savefig(OUT / "06_ensemble_vs_single.png", bbox_inches="tight")
    plt.close(fig)


plot6_ensemble_vs_single()
print("  [6/8] 06_ensemble_vs_single.png")


# ============================================================================
# Plot 7 — Seed-spread shoulder vs ansatz_d
# ============================================================================

def plot7_seed_shoulder():
    by_ad = defaultdict(list)
    for r in grid:
        lr = math.log10(r["compile_err_max"] / max(r["compile_err_min"], 1e-30))
        by_ad[r["ansatz_depth"]].append(lr)

    ads = sorted(by_ad)
    med = [np.median(by_ad[ad]) for ad in ads]
    frac10 = [100 * np.mean([v > 1 for v in by_ad[ad]]) for ad in ads]

    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    ax2 = ax1.twinx()
    ax2.grid(False)

    color1 = "C0"; color2 = "C1"
    bar = ax1.bar(ads, frac10, width=0.6, color=color2, alpha=0.6,
                   label="% of cases with >10× seed-to-seed variation",
                   edgecolor="C3", linewidth=0.5)
    line, = ax2.plot(ads, med, "o-", color=color1, ms=10, linewidth=2,
                      label="Typical log₁₀(worst seed / best seed)")

    ax1.set_xlabel("Compiled circuit depth")
    ax1.set_ylabel("Percent of test cases where random starts disagree 10× or more",
                   color=color2)
    ax2.set_ylabel("Typical disagreement between random starts\n"
                   "(log₁₀ ratio of their errors)", color=color1)
    ax1.set_xticks(ads)
    ax1.tick_params(axis="y", labelcolor=color2)
    ax2.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(frac10) * 1.25)

    fig.legend(handles=[bar, line], loc="upper right",
               bbox_to_anchor=(0.82, 0.93), fontsize=10)
    ax1.set_title("Random initialization matters most at one specific depth")

    fig.subplots_adjust(bottom=0.32)
    fig.text(0.02, 0.02,
             "Plain reading:  Each column represents one 'compiled circuit depth' value we tested.\n"
             "• ORANGE BARS (left axis): how often do three random starts give wildly different answers (10× or more apart)?\n"
             "• BLUE DOTS (right axis): the typical size of that disagreement, measured in log-units.\n"
             "• The shoulder at depth 4–5 is clear: at exactly the minimum depth the target needs, random starts often find\n"
             "   different solutions (up to 30% of cases). Below depth 3: only one basin exists — all starts agree (boring).\n"
             "   Above depth 6: plenty of good basins — all starts find one (also boring).\n"
             "• The 'interesting' regime for any ensemble method is exactly this narrow depth window.",
             fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.savefig(OUT / "07_seed_shoulder.png", bbox_inches="tight")
    plt.close(fig)


plot7_seed_shoulder()
print("  [7/8] 07_seed_shoulder.png")


# ============================================================================
# Plot 8 — Mechanism benefits strip chart
# ============================================================================

def plot8_mechanisms():
    dr = load("drop_rate_sweep.csv",
              int_cols=("init_seed",),
              float_cols=("drop_rate", "final_cost"),
              str_cols=("config_label",))
    pb = load("perturbation_benefit.csv",
              float_cols=("baseline_max_td", "expanded_max_td", "perturb_scale"),
              str_cols=("label",))
    ot = load("overtight_perturb.csv",
              float_cols=("baseline_ens_max_td", "expanded_ens_max_td",
                           "perturb_scale"),
              str_cols=("label",))

    def best_improvement(config_data, baseline_key, expanded_key):
        base = config_data[0][baseline_key]
        best = min(config_data, key=lambda r: r[expanded_key])[expanded_key]
        return 100 * (base - best) / base if base > 0 else 0

    # Dropout: per config, improvement = (baseline_mean - best_drop_mean) / baseline_mean
    drop_by_cfg = defaultdict(lambda: defaultdict(list))
    for r in dr:
        drop_by_cfg[r["config_label"]][r["drop_rate"]].append(r["final_cost"])
    drop_improv = []
    for cfg, rates in drop_by_cfg.items():
        base = np.mean(rates[0.0])
        if base == 0: continue
        best = min(np.mean(v) for v in rates.values())
        drop_improv.append((cfg, 100 * (base - best) / base))

    # Paired perturbation
    pb_by_cfg = defaultdict(list)
    for r in pb: pb_by_cfg[r["label"]].append(r)
    pb_improv = [(cfg, best_improvement(rows, "baseline_max_td", "expanded_max_td"))
                 for cfg, rows in pb_by_cfg.items()]

    # Over-tight perturbation
    ot_by_cfg = defaultdict(list)
    for r in ot: ot_by_cfg[r["label"]].append(r)
    ot_improv = [(cfg, best_improvement(rows, "baseline_ens_max_td",
                                          "expanded_ens_max_td"))
                 for cfg, rows in ot_by_cfg.items()]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    all_improvs = []
    colors = []
    labels = []
    positions = []
    for i, (name, data) in enumerate([
            ("Dropout\n(skip gate updates)", drop_improv),
            ("Paired perturbation\n(Kalloor's recipe)", pb_improv),
            ("Paired perturbation\nwith deeper circuit", ot_improv)]):
        for cfg, val in data:
            positions.append(i)
            all_improvs.append(val)
            if val > 5:
                colors.append("C2")
            elif val > -5:
                colors.append("gray")
            else:
                colors.append("C3")
        labels.append(name)

    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(positions))
    ax.scatter(np.array(positions) + jitter, all_improvs, c=colors,
               s=100, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="k", linewidth=1)
    ax.axhline(5, color="C2", linestyle=":", alpha=0.5, label="±5% threshold")
    ax.axhline(-5, color="C3", linestyle=":", alpha=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Improvement over baseline\n(% reduction in worst-case error)")
    ax.set_title("Each intervention we tried gives modest benefits at best —\n"
                 "often nothing, sometimes makes things worse")
    ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    fig.subplots_adjust(bottom=0.35)
    fig.text(0.02, 0.02,
             "Plain reading:  Each dot is one test configuration. Y-axis shows how much the intervention reduced the worst-case error,\n"
             "compared to just running the compiler with 3 random starts.\n"
             "• GREEN (> +5%) — meaningful improvement. Rare.   • GRAY (±5%) — no real change.   • RED (< −5%) — intervention hurt.\n"
             "• Conclusion: none of the add-on mechanisms we tried (dropout, Kalloor's paired-opposite perturbation with various settings)\n"
             "   deliver reliable, substantial gains. The naive 3-candidate weighted ensemble already extracts the available benefit.",
             fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.savefig(OUT / "08_mechanism_benefits.png", bbox_inches="tight")
    plt.close(fig)


plot8_mechanisms()
print("  [8/8] 08_mechanism_benefits.png")

print(f"\nAll plots saved to {OUT}/")
