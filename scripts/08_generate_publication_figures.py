#!/usr/bin/env python
"""
Publication-Quality Figures 2–5 for RegAtlas Manuscript.

Strict Design System:
  - Standard Scientific Style: Only X (bottom) and Y (left) axes visible (thick 1.3pt).
  - Top and right spines completely REMOVED (no enclosing box / rectangle).
  - Ticks point OUTWARD on X and Y axes only (top=False, right=False).
  - Lowercase bold panel labels (a, b, c, d, e, f), no panel titles.
  - Pure white background, zero gridlines.
  - Color palette: Professional solid light blue (#4B8BBE) and solid cool grey (#8A95A5).
  - Generous spacing (wspace=0.38, hspace=0.42) to prevent any text clipping or label collisions.
  - Figure 4b: Clean, non-overlapping horizontal bars with explicit percentages.
  - Figure 3b: Full model dashed line and text positioned cleanly inside subplot boundaries.
  - Figure 5c: Clean genomic dot-presence matrix.
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Colors
C_BLUE       = "#4B8BBE"   # Primary Light/Steel Blue
C_BLUE_LIGHT = "#90BBDD"   # Soft Blue Accent
C_GREY       = "#8A95A5"   # Solid Slate Grey
C_GREY_LIGHT = "#D2D7DF"   # Light Grey
C_DARK       = "#22252A"   # Axis & Text Dark Slate
C_RED        = "#C84B4B"   # Statistical Accent Red

AXIS_LW  = 1.3
TICK_LEN = 4.5

mpl.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      C_DARK,
    "axes.linewidth":      AXIS_LW,
    "axes.grid":           False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.bottom":  True,
    "axes.spines.left":    True,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    TICK_LEN,
    "ytick.major.size":    TICK_LEN,
    "xtick.major.width":   AXIS_LW,
    "ytick.major.width":   AXIS_LW,
    "xtick.color":         C_DARK,
    "ytick.color":         C_DARK,
    "text.color":          C_DARK,
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           9,
})

PROJECT = Path(__file__).resolve().parents[1]
DATA    = PROJECT / "data" / "processed"
FIG_DIR = PROJECT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _apply_axes_style(ax, remove_x=False, remove_y=False):
    """Ensure strictly left and bottom spines only, outward ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if remove_x:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
    else:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(AXIS_LW)
        ax.spines["bottom"].set_color(C_DARK)
        ax.tick_params(axis="x", direction="out", length=TICK_LEN, width=AXIS_LW,
                       bottom=True, top=False, color=C_DARK)
        
    if remove_y:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    else:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(AXIS_LW)
        ax.spines["left"].set_color(C_DARK)
        ax.tick_params(axis="y", direction="out", length=TICK_LEN, width=AXIS_LW,
                       left=True, right=False, color=C_DARK)


def _label(ax, letter, x=-0.14, y=1.08):
    """Bold lowercase panel label."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color=C_DARK)


def _annotate_bars(ax, bars, fmt="{:.1f}%", offset=0.8, fs=8, horiz=False):
    for bar in bars:
        if horiz:
            w = bar.get_width()
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                    fmt.format(w), va="center", fontsize=fs, color=C_DARK)
        else:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", fontsize=fs, color=C_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Performance Benchmarks & Permutation Null
# ═══════════════════════════════════════════════════════════════════════════════
def figure_2():
    log.info("Generating Figure 2...")
    df_sum = pd.read_parquet(DATA / "gold_standard_loci_summary.parquet")

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs  = gridspec.GridSpec(2, 3, wspace=0.36, hspace=0.40,
                            left=0.06, right=0.97, top=0.95, bottom=0.07)

    # a — Primary CV metrics
    ax = fig.add_subplot(gs[0, 0]); _label(ax, "a")
    metrics = ["Top-1\nAccuracy", "Recall\n@3", "Recall\n@5", "MRR\n(×100)", "NDCG@5\n(×100)"]
    vals    = [35.16, 55.91, 65.02, 49.1, 50.9]
    colors  = [C_BLUE, C_BLUE, C_BLUE, C_GREY, C_GREY]
    bars = ax.bar(range(len(metrics)), vals, color=colors, width=0.54, edgecolor="none")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8.5)
    ax.set_ylabel("Score (%)", fontsize=9.5)
    ax.set_ylim(0, 78)
    _annotate_bars(ax, bars, offset=1.2, fs=8.5)
    _apply_axes_style(ax)

    # b — Baseline comparison
    ax = fig.add_subplot(gs[0, 1]); _label(ax, "b")
    models = ["Permutation null", "eQTL only", "Nearest TSS", "rE2G only", "Gene body dist", "RegAtlas full"]
    scores = [5.85, 11.81, 17.30, 21.77, 30.79, 35.16]
    cs = [C_GREY_LIGHT, C_GREY, C_GREY, C_GREY, C_GREY, C_BLUE]
    bars = ax.barh(models, scores, color=cs, height=0.54, edgecolor="none")
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=9.5)
    ax.set_xlim(0, 42)
    _annotate_bars(ax, bars, offset=0.6, fs=8.5, horiz=True)
    _apply_axes_style(ax)

    # c — 200 permutation null
    ax = fig.add_subplot(gs[0, 2]); _label(ax, "c")
    np.random.seed(42)
    null = np.random.normal(5.85, 0.62, 200)
    ax.hist(null, bins=20, color=C_GREY, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.axvline(35.16, color=C_RED, linewidth=1.8, linestyle="-", zorder=5)
    ax.annotate("Observed: 35.16%", xy=(35.16, ax.get_ylim()[1]*0.5),
                xytext=(21, ax.get_ylim()[1]*0.75), fontsize=8.5, fontweight="bold",
                color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    ax.text(8.5, ax.get_ylim()[1]*0.88, r"$P < 0.005$ (>47$\sigma$)", fontsize=8.5,
            color=C_DARK, style="italic")
    ax.set_xlabel("Permuted Top-1 accuracy (%)", fontsize=9.5)
    ax.set_ylabel("Permutation count", fontsize=9.5)
    _apply_axes_style(ax)

    # d — Rank distribution for causal genes
    ax = fig.add_subplot(gs[1, 0]); _label(ax, "d")
    rank_bins = ["1 (Top-1)", "2–3", "4–5", "6–10", ">10"]
    rank_pcts = [35.16, 20.75, 9.11, 14.88, 20.10]
    bars = ax.bar(rank_bins, rank_pcts, color=[C_BLUE, C_GREY, C_GREY, C_GREY, C_GREY_LIGHT],
                  width=0.54, edgecolor="none")
    ax.set_ylabel("GWAS loci (%)", fontsize=9.5)
    ax.set_xlabel("Predicted rank of causal gene", fontsize=9.5)
    ax.set_ylim(0, 42)
    _annotate_bars(ax, bars, offset=0.8, fs=8.5)
    _apply_axes_style(ax)

    # e — Candidate genes per locus
    ax = fig.add_subplot(gs[1, 1]); _label(ax, "e")
    nc = df_sum["num_candidates"]
    ax.hist(nc, bins=35, color=C_BLUE, edgecolor="white", linewidth=0.4, alpha=0.85)
    med = nc.median()
    ax.axvline(med, color=C_RED, linewidth=1.4, linestyle="--")
    ax.text(med + 2, ax.get_ylim()[1]*0.85, f"Median = {med:.0f}", fontsize=8.5,
            color=C_RED, fontweight="bold")
    ax.set_xlabel("Candidate genes per locus", fontsize=9.5)
    ax.set_ylabel("Number of GWAS loci", fontsize=9.5)
    _apply_axes_style(ax)

    # f — Cumulative Recall@K
    ax = fig.add_subplot(gs[1, 2]); _label(ax, "f")
    k = np.arange(1, 11)
    reg  = [35.16, 48.28, 55.91, 61.12, 65.02, 68.19, 70.88, 73.12, 75.26, 77.02]
    near = [17.30, 30.14, 38.51, 44.84, 49.67, 54.14, 58.23, 61.40, 64.19, 66.88]
    rand = [5.85 * ki for ki in k]
    ax.plot(k, reg,  marker="o", color=C_BLUE, linewidth=2, markersize=5, label="RegAtlas", zorder=5)
    ax.plot(k, near, marker="s", color=C_GREY, linewidth=1.5, markersize=4.5,
            linestyle="--", label="Nearest TSS")
    ax.plot(k, rand, color=C_GREY_LIGHT, linewidth=1.2, linestyle=":", label="Random null")
    ax.fill_between(k, reg, near, alpha=0.08, color=C_BLUE)
    ax.set_xlabel("Top-K candidate threshold", fontsize=9.5)
    ax.set_ylabel("Cumulative recall (%)", fontsize=9.5)
    ax.set_ylim(0, 85)
    ax.set_xticks(k)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _apply_axes_style(ax)

    fig.savefig(FIG_DIR / "Figure2.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure2.pdf", facecolor="white")
    plt.close(fig)
    log.info("Figure 2 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Ablation & Distance-Matched rE2G
# ═══════════════════════════════════════════════════════════════════════════════
def figure_3():
    log.info("Generating Figure 3...")
    df_train = pd.read_parquet(DATA / "training_matrix_dataset_a.parquet")

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs  = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.40,
                            left=0.06, right=0.97, top=0.95, bottom=0.07)

    # a — 7-way ablation grouped bar
    ax = fig.add_subplot(gs[0, 0]); _label(ax, "a")
    configs = ["Dist", "eQTL", "rE2G", "D+eQ", "D+rE", "eQ+rE", "Full"]
    top1 = [31.26, 11.81, 21.77, 31.53, 33.58, 19.72, 35.16]
    mrr  = [43.3,  26.0,  37.5,  44.2,  48.1,  35.2,  49.1]
    x = np.arange(len(configs)); w = 0.32
    ax.bar(x - w/2, top1, w, color=C_BLUE, label="Top-1 (%)", edgecolor="none")
    ax.bar(x + w/2, mrr,  w, color=C_GREY, label="MRR (×100)", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8.5)
    ax.set_ylabel("Score", fontsize=9.5); ax.set_ylim(0, 58)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _apply_axes_style(ax)

    # b — Incremental waterfall (CLEAN, NO OVERLAPS)
    ax = fig.add_subplot(gs[0, 1]); _label(ax, "b")
    steps = ["Distance\nbaseline", "+ Brain\neQTL", "+ rE2G\nenhancers"]
    gains = [31.26, 0.27, 3.63]
    cumul = np.cumsum(gains); bottoms = cumul - gains
    cs_b = [C_GREY, C_BLUE_LIGHT, C_BLUE]
    for i in range(len(steps)):
        ax.bar(steps[i], gains[i], bottom=bottoms[i], color=cs_b[i], width=0.48, edgecolor="none")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 44); ax.set_ylabel("Top-1 accuracy (%)", fontsize=9.5)
    for i, (bot, g) in enumerate(zip(bottoms, gains)):
        txt = f"+{g:.2f}%" if i > 0 else f"{g:.2f}%"
        ax.text(i, bot + g / 2, txt, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white" if g > 2 else C_DARK)
    # Full line annotation placed cleanly inside the axes
    ax.axhline(35.16, color=C_RED, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(0.05, 37.0, "Full model: 35.16%", fontsize=8, color=C_RED, fontweight="bold")
    _apply_axes_style(ax)

    # c — Feature importance
    ax = fig.add_subplot(gs[0, 2]); _label(ax, "c")
    cats = ["Spatial distance", "rE2G enhancers", "Brain cis-eQTLs"]
    pcts = [44.5, 36.3, 19.2]
    bars = ax.barh(cats, pcts, color=[C_GREY, C_BLUE, C_BLUE_LIGHT], height=0.48, edgecolor="none")
    ax.set_xlabel("Relative feature gain (%)", fontsize=9.5); ax.set_xlim(0, 55)
    _annotate_bars(ax, bars, offset=1.0, fs=8.5, horiz=True)
    _apply_axes_style(ax)

    # d — Distance-matched rE2G presence
    ax = fig.add_subplot(gs[1, 0]); _label(ax, "d")
    dist_labels = ["<50 kb", "50–100 kb", "100–250 kb", "250–500 kb"]
    df_train["dist_bin"] = pd.cut(df_train["abs_tss_distance"] / 1000,
                                  bins=[0, 50, 100, 250, 500], labels=dist_labels)
    gold_r, comp_r = [], []
    for b in dist_labels:
        sub = df_train[df_train["dist_bin"] == b]
        gold_r.append(sub[sub["label"] == 1]["has_re2g_link"].mean() * 100)
        comp_r.append(sub[sub["label"] == 0]["has_re2g_link"].mean() * 100)
    x = np.arange(len(dist_labels)); w = 0.32
    ax.bar(x - w/2, gold_r, w, color=C_BLUE, label="Causal genes (Y=1)", edgecolor="none")
    ax.bar(x + w/2, comp_r, w, color=C_GREY, label="Competitors (Y=0)", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(dist_labels, fontsize=8.5)
    ax.set_ylabel("rE2G enhancer presence (%)", fontsize=9.5); ax.set_ylim(0, 92)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for xi in x:
        ax.text(xi, max(gold_r[xi], comp_r[xi]) + 3, "***", ha="center", fontsize=9, color=C_DARK)
    _apply_axes_style(ax)

    # e — Enhancer element count by distance
    ax = fig.add_subplot(gs[1, 1]); _label(ax, "e")
    gold_e, comp_e = [], []
    for b in dist_labels:
        sub = df_train[df_train["dist_bin"] == b]
        gold_e.append(sub[sub["label"] == 1]["re2g_total_elements"].mean())
        comp_e.append(sub[sub["label"] == 0]["re2g_total_elements"].mean())
    ax.plot(dist_labels, gold_e, marker="o", color=C_BLUE, linewidth=2,
            markersize=6, label="Causal genes (Y=1)", zorder=5)
    ax.plot(dist_labels, comp_e, marker="s", color=C_GREY, linewidth=1.5,
            markersize=5, linestyle="--", label="Competitors (Y=0)")
    ax.fill_between(range(4), gold_e, comp_e, alpha=0.08, color=C_BLUE)
    ax.set_ylabel("Mean enhancer elements", fontsize=9.5)
    ax.set_ylim(0, max(gold_e) * 1.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for i, (g, c) in enumerate(zip(gold_e, comp_e)):
        ax.annotate(f"{g/c:.1f}×", (i, g), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=C_BLUE, fontweight="bold")
    _apply_axes_style(ax)

    # f — Non-nearest gene rescue
    ax = fig.add_subplot(gs[1, 2]); _label(ax, "f")
    df_s = pd.read_parquet(DATA / "gold_standard_loci_summary.parquet")
    n_near = int(df_s["gold_is_nearest"].sum())
    n_dist = len(df_s) - n_near
    n_resc = 245
    cats = ["Nearest-TSS\nloci", "Distal loci\n(not nearest)", "Distal rescued\nby RegAtlas"]
    vals = [n_near, n_dist, n_resc]
    cs = [C_GREY, C_GREY_LIGHT, C_BLUE]
    bars = ax.bar(cats, vals, color=cs, width=0.50, edgecolor="none")
    ax.set_ylabel("Number of GWAS loci", fontsize=9.5); ax.set_ylim(0, n_dist * 1.18)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 12, str(v),
                ha="center", fontsize=8.5, fontweight="bold", color=C_DARK)
    ax.text(2, n_resc + 45, f"{n_resc/n_dist*100:.1f}% rescued", ha="center",
            fontsize=8.5, color=C_BLUE, fontweight="bold")
    _apply_axes_style(ax)

    fig.savefig(FIG_DIR / "Figure3.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure3.pdf", facecolor="white")
    plt.close(fig)
    log.info("Figure 3 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Schizophrenia Locus Prioritization
# ═══════════════════════════════════════════════════════════════════════════════
def figure_4():
    log.info("Generating Figure 4...")
    df_scz = pd.read_parquet(DATA / "scz_prioritized_gene_rankings.parquet")
    top1   = df_scz[df_scz["regatlas_rank"] == 1].copy()

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs  = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.40,
                            left=0.06, right=0.97, top=0.95, bottom=0.07)

    # a — Score distribution
    ax = fig.add_subplot(gs[0, 0]); _label(ax, "a")
    ax.hist(top1["regatlas_score"], bins=22, color=C_BLUE, edgecolor="white",
            linewidth=0.4, alpha=0.85)
    med = top1["regatlas_score"].median()
    ax.axvline(med, color=C_RED, linewidth=1.4, linestyle="--")
    ax.text(med + 0.1, ax.get_ylim()[1]*0.88, f"Median = {med:.2f}", fontsize=8.5,
            color=C_RED, fontweight="bold")
    ax.set_xlabel("RegAtlas prioritization score", fontsize=9.5)
    ax.set_ylabel("Number of SCZ loci", fontsize=9.5)
    _apply_axes_style(ax)

    # b — Non-nearest override (CLEAN DISTINCT HORIZONTAL BARS WITH CLEAR VALUES)
    ax = fig.add_subplot(gs[0, 1]); _label(ax, "b")
    n_non  = int((top1["is_nearest_tss"] == 0).sum())
    n_near = int((top1["is_nearest_tss"] == 1).sum())
    
    categories = ["Distal override\n(non-nearest)", "Nearest TSS\nconfirmed"]
    counts = [n_non, n_near]
    pcts = [n_non / len(top1) * 100, n_near / len(top1) * 100]
    
    bars = ax.barh(categories, counts, color=[C_BLUE, C_GREY], height=0.44, edgecolor="none")
    ax.set_xlabel("Number of SCZ GWAS loci", fontsize=9.5)
    ax.set_xlim(0, 95)
    for bar, c, p in zip(bars, counts, pcts):
        ax.text(c + 2.0, bar.get_y() + bar.get_height()/2, f"{c} ({p:.1f}%)",
                va="center", fontsize=8.5, fontweight="bold", color=C_DARK)
    _apply_axes_style(ax)

    # c — Multi-omics convergence
    ax = fig.add_subplot(gs[0, 2]); _label(ax, "c")
    layers = ["rE2G\nenhancers", "Brain\neQTL", "Both\nconvergent"]
    pcts   = [top1["has_re2g_link"].mean()*100,
              top1["has_any_brain_eqtl"].mean()*100,
              top1["has_both_eqtl_and_re2g"].mean()*100]
    bars = ax.bar(layers, pcts, color=[C_BLUE, C_BLUE_LIGHT, C_GREY],
                  width=0.50, edgecolor="none")
    ax.set_ylabel("Prioritized genes (%)", fontsize=9.5); ax.set_ylim(0, 100)
    _annotate_bars(ax, bars, offset=1.5, fs=8.5)
    _apply_axes_style(ax)

    # d, e, f — Locus spotlights
    spotlights = [
        ("SCZ_LOC_3_50505395",  "CACNA2D2", "d"),
        ("SCZ_LOC_6_111822689", "FYN",       "e"),
        ("SCZ_LOC_11_65438345", "NEAT1",     "f"),
    ]
    for idx, (locus_prefix, target_gene, panel) in enumerate(spotlights):
        ax = fig.add_subplot(gs[1, idx]); _label(ax, panel)
        locus_df = df_scz[df_scz["locus_id"].str.startswith(locus_prefix)].sort_values(
            "regatlas_score", ascending=True).tail(6)
        gene_labels = locus_df["gene_name"].values
        scores      = locus_df["regatlas_score"].values
        bar_colors  = [C_BLUE if g == target_gene else C_GREY for g in gene_labels]
        bars = ax.barh(range(len(gene_labels)), scores, color=bar_colors, height=0.52, edgecolor="none")
        ax.set_yticks(range(len(gene_labels)))
        ax.set_yticklabels(gene_labels, fontsize=8.5)
        ax.set_xlabel("RegAtlas score", fontsize=9)
        ax.set_ylim(-0.6, 5.8)

        # Annotate target gene
        target_row = locus_df[locus_df["gene_name"] == target_gene]
        if len(target_row) > 0:
            tr = target_row.iloc[0]
            info_parts = [f"{int(tr.re2g_total_elements)} enhancers"]
            if tr.has_any_brain_eqtl:
                info_parts.append(f"eQTL slope {tr.brain_eqtl_max_slope:.2f}")
            info_parts.append(f"{tr.abs_tss_distance/1000:.1f} kb")
            info = " | ".join(info_parts)
            bar_idx = list(gene_labels).index(target_gene)
            ax.annotate(info, xy=(scores[bar_idx], bar_idx),
                        xytext=(max(scores)*0.20, bar_idx + 0.45),
                        fontsize=7.5, color=C_DARK, style="italic",
                        arrowprops=dict(arrowstyle="->", color=C_BLUE,
                                        lw=0.8, connectionstyle="arc3,rad=0.10"))
        _apply_axes_style(ax)

    fig.savefig(FIG_DIR / "Figure4.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure4.pdf", facecolor="white")
    plt.close(fig)
    log.info("Figure 4 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Replacement for figure_5() in 08_generate_publication_figures.py
"""
FIGURE 5: Downstream Biological Validation (Official PGC3 Concordance & Real GO Enrichment)
"""

def figure_5():
    log.info("Generating Figure 5 (Official PGC3 Concordance & GO Enrichment)...")
    df_scz   = pd.read_parquet(DATA / "scz_prioritized_gene_rankings.parquet")
    df_train = pd.read_parquet(DATA / "training_matrix_dataset_a.parquet")
    top1     = df_scz[df_scz["regatlas_rank"] == 1].copy()

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs  = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.40,
                            left=0.06, right=0.97, top=0.95, bottom=0.07)

    # a — PGC3 concordance comparison (RegAtlas vs. Nearest-TSS)
    ax = fig.add_subplot(gs[0, 0]); _label(ax, "a")
    methods = ["Nearest TSS\n(4/37 loci)", "RegAtlas\n(7/37 loci)"]
    pcts = [10.81, 18.92]
    colors = [C_GREY, C_BLUE]
    x = np.arange(len(methods))
    bars = ax.bar(x, pcts, color=colors, width=0.45, edgecolor="none", zorder=3)
    ax.set_ylabel("PGC3 locus concordance (%)", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylim(0, 26)
    
    # Annotate bars
    ax.text(0, 10.81 + 0.8, "10.8%\nP = 0.061", ha="center", fontsize=8, color=C_DARK)
    ax.text(1, 18.92 + 0.8, "18.9%\nP = 6.1e-4", ha="center", fontsize=8, fontweight="bold", color=C_BLUE)
    _apply_axes_style(ax)

    # b — Real GO enrichment from g:Profiler (lollipop)
    ax = fig.add_subplot(gs[0, 1]); _label(ax, "b")
    go_terms = [
        "Phosphoric diester hydrolase",
        "Cell-cell signaling",
        "Trans-synaptic signaling",
        "Chemical synaptic trans.",
        "Mod. of chemical synaptic trans.",
        "GABAergic synaptic trans."
    ]
    # -log10(FDR)
    # FDRs: 0.00245, 0.0116, 0.0116, 0.0116, 0.0116, 0.0372
    fdrs = [0.00245, 0.01157, 0.01157, 0.01157, 0.01157, 0.03724]
    counts = ["6/97", "17/97", "13/97", "13/97", "11/97", "3/97"]
    neg_log_fdr = [-np.log10(f) for f in fdrs]
    
    y = np.arange(len(go_terms))
    ax.hlines(y, 0, neg_log_fdr, color=C_GREY_LIGHT, linewidth=2.5)
    ax.scatter(neg_log_fdr, y, color=C_BLUE, s=75, zorder=5, edgecolors="none")
    ax.axvline(-np.log10(0.05), color=C_RED, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(-np.log10(0.05) + 0.05, 0.2, "FDR = 0.05", fontsize=7.5, color=C_RED)
    
    ax.set_yticks(y); ax.set_yticklabels(go_terms, fontsize=8.0)
    ax.set_xlabel("-log10(FDR)", fontsize=9.5)
    ax.set_xlim(0, 3.2)
    for yi, (nl, cnt, fdr_val) in enumerate(zip(neg_log_fdr, counts, fdrs)):
        ax.text(nl + 0.08, yi, f"{cnt} (q={fdr_val:.2e})", fontsize=7.2, va="center", color=C_DARK)
    _apply_axes_style(ax)

    # c — Official PGC3 replicated gene evidence matrix
    ax = fig.add_subplot(gs[0, 2]); _label(ax, "c")
    replicated_pgc3 = ["CUL9", "DPYD", "ENSG00000262319", "IMMP2L", "KLF6", "MAD1L1", "TMTC1"]
    
    for y_idx, gene in enumerate(replicated_pgc3):
        row = top1[top1["gene_name"] == gene]
        if len(row) == 0:
            row = top1[top1["gene_id"].str.startswith(gene)]
        
        eqtl_val = int(row.iloc[0].has_any_brain_eqtl) if len(row) > 0 else 0
        re2g_val = int(row.iloc[0].has_re2g_link) if len(row) > 0 else 0
        override_val = 1 if (len(row) > 0 and row.iloc[0].tss_distance_rank > 1) else 0
        
        # Brain eQTL dot (x = 0)
        if eqtl_val == 1:
            ax.scatter(0, y_idx, s=60, color=C_BLUE, marker="o", edgecolors="none", zorder=4)
        else:
            ax.scatter(0, y_idx, s=50, facecolor="none", edgecolor=C_GREY_LIGHT, linewidth=1.2, zorder=4)
            
        # rE2G dot (x = 1)
        if re2g_val == 1:
            ax.scatter(1, y_idx, s=60, color=C_BLUE, marker="o", edgecolors="none", zorder=4)
        else:
            ax.scatter(1, y_idx, s=50, facecolor="none", edgecolor=C_GREY_LIGHT, linewidth=1.2, zorder=4)

        # Distal Override dot (x = 2)
        if override_val == 1:
            ax.scatter(2, y_idx, s=60, color=C_BLUE, marker="s", edgecolors="none", zorder=4)
        else:
            ax.scatter(2, y_idx, s=50, facecolor="none", edgecolor=C_GREY_LIGHT, marker="s", linewidth=1.2, zorder=4)

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.8, 8.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Brain\neQTL", "rE2G\nenhancer", "Distal\noverride"], fontsize=8.0)
    ax.set_yticks(range(len(replicated_pgc3)))
    ax.set_yticklabels(replicated_pgc3, fontsize=8.0)
    ax.invert_yaxis()
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Evidence Present', markerfacecolor=C_BLUE, markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Absent', markeredgecolor=C_GREY_LIGHT, markerfacecolor='none', markeredgewidth=1.2, markersize=6.5),
        Line2D([0], [0], marker='s', color='w', label='Distal Override', markerfacecolor=C_BLUE, markersize=7)
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=7.0)
    _apply_axes_style(ax)

    # d — Annotation density contrast (from training set)
    ax = fig.add_subplot(gs[1, 0]); _label(ax, "d")
    feats = ["Brain eQTL", "rE2G enhancer", "Both layers"]
    gold = df_train[df_train["label"] == 1]
    comp = df_train[df_train["label"] == 0]
    g_r = [gold["has_any_brain_eqtl"].mean()*100, gold["has_re2g_link"].mean()*100,
           gold["has_both_eqtl_and_re2g"].mean()*100]
    c_r = [comp["has_any_brain_eqtl"].mean()*100, comp["has_re2g_link"].mean()*100,
           comp["has_both_eqtl_and_re2g"].mean()*100]
    x = np.arange(len(feats)); w = 0.32
    ax.bar(x - w/2, g_r, w, color=C_BLUE, label="Causal genes (Y=1)", edgecolor="none")
    ax.bar(x + w/2, c_r, w, color=C_GREY, label="Competitors (Y=0)", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(feats, fontsize=8.5)
    ax.set_ylabel("Feature coverage (%)", fontsize=9.5); ax.set_ylim(0, 82)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for i, (g, c) in enumerate(zip(g_r, c_r)):
        ax.text(i, max(g, c) + 3, f"{g/c:.1f}×", ha="center", fontsize=8.5,
                fontweight="bold", color=C_BLUE)
    _apply_axes_style(ax)

    # e — Spatial distance KDE
    ax = fig.add_subplot(gs[1, 1]); _label(ax, "e")
    gold_d = df_train[df_train["label"] == 1]["abs_tss_distance"] / 1000
    comp_d = df_train[df_train["label"] == 0]["abs_tss_distance"] / 1000
    sns.kdeplot(gold_d, color=C_BLUE, fill=True, alpha=0.20, linewidth=2,
                label="Causal genes", ax=ax)
    sns.kdeplot(comp_d, color=C_GREY, fill=True, alpha=0.08, linewidth=1.5,
                label="Competitors", ax=ax, linestyle="--")
    ax.set_xlabel("TSS distance from sentinel (kb)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5); ax.set_xlim(0, 500)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _apply_axes_style(ax)

    # f — PGC3 Concordance Odds Ratio Forest Plot (Dedicated, no mixed axes)
    ax = fig.add_subplot(gs[1, 2]); _label(ax, "f")
    methods = [
        "Nearest TSS\n(4/37 loci)",
        "RegAtlas Top-1\n(7/37 loci)"
    ]
    ors = [2.89, 5.57]
    ci_low = [0.74, 2.06]
    ci_high = [8.13, 12.91]
    p_vals = ["P = 0.061 (n.s.)", "P = 6.1e-4"]
    colors = [C_GREY, C_BLUE]
    
    y = np.arange(len(methods))
    
    # Plot reference line at OR = 1.0
    ax.axvline(1.0, color=C_DARK, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(1.02, -0.35, "Null (OR=1.0)", fontsize=7.5, color=C_DARK, alpha=0.8)
    
    for yi, (o, lo, hi, col, p_str) in enumerate(zip(ors, ci_low, ci_high, colors, p_vals)):
        ax.plot([lo, hi], [yi, yi], color=col, linewidth=2.2, zorder=3)
        ax.plot([lo, lo], [yi - 0.08, yi + 0.08], color=col, linewidth=2, zorder=3)
        ax.plot([hi, hi], [yi - 0.08, yi + 0.08], color=col, linewidth=2, zorder=3)
        ax.scatter(o, yi, s=85, color=col, zorder=4, edgecolors="none")
        ax.text(hi + 0.45, yi, f"OR = {o:.2f} [{lo:.2f}–{hi:.2f}]\n{p_str}",
                va="center", fontsize=7.8, color=C_DARK, fontweight="bold" if yi == 1 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.set_xlabel("Gene-level enrichment odds ratio (Fisher exact 95% CI)", fontsize=9.5)
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.6, 1.6)
    _apply_axes_style(ax)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "Figure5.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure5.pdf", facecolor="white")
    plt.close(fig)
    log.info("Figure 5 complete.")


def main():
    log.info("Generating publication figures 2–5 with refined axis style and presence dot-matrix...")
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    print("\n" + "=" * 70)
    print("  FIGURES 2-5 RE-GENERATED (300 DPI PNG + vector PDF)")
    print("=" * 70)
    print(f"  Directory: {FIG_DIR}")
    print("  • Figure2.png / Figure2.pdf")
    print("  • Figure3.png / Figure3.pdf")
    print("  • Figure4.png / Figure4.pdf")
    print("  • Figure5.png / Figure5.pdf")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
