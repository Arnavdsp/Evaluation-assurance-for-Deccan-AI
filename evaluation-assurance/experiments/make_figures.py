"""Regenerates figures/ from fresh runs (curves) and results/*.csv."""
import _setup
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import run_one_seed
from metrics import curve

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GRAY, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": LIGHT,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
F = _setup.FIGURES

# 1. operating curve ---------------------------------------------------------
budgets = np.linspace(0, 1, 51)
methods = ["random", "surface_rf", "rules", "structural_lr"]
C = {m: [] for m in methods}
for seed in range(1, _setup.N_SEEDS + 1):
    ps, scores, _ = run_one_seed(seed, methods=methods)
    sf = (ps.ground_truth_pass == 0).to_numpy(); rng = np.random.default_rng(seed)
    for m in methods:
        C[m].append(curve(scores[m], sf, budgets, rng))
style = {"random": (LIGHT, "--", "Random spot-check"),
         "surface_rf": (GRAY, "-", "Model on evaluator's own info"),
         "rules": (ORANGE, "-", "Rule: count fired checks"),
         "structural_lr": (BLUE, "-", "Independent signals + logistic reg.")}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for m in methods:
    a = np.array(C[m]) * 100; mu, sd = a.mean(0), a.std(0)
    col, ls, lab = style[m]
    ax.fill_between(budgets * 100, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    ax.plot(budgets * 100, mu, color=col, ls=ls, lw=2.2 if m == "structural_lr" else 1.8, label=lab)
ax.axvline(20, color=GRID, lw=1, zorder=0)
i20 = 10
ax.annotate(f"{np.mean(C['structural_lr'], 0)[i20]*100:.0f}%", (20, np.mean(C['structural_lr'], 0)[i20]*100),
            xytext=(8, -4), textcoords="offset points", color=INK, fontweight="bold")
ax.annotate(f"{np.mean(C['random'], 0)[i20]*100:.0f}%", (20, np.mean(C['random'], 0)[i20]*100),
            xytext=(8, -12), textcoords="offset points", color=INK2)
ax.set_xlim(0, 60); ax.set_ylim(0, 85)
ax.set_xlabel("% of evaluator PASS verdicts sent to human review")
ax.set_ylabel("% of silent failures caught")
ax.grid(axis="y", color=GRID, lw=0.8); ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title("Same review budget, ~2.4x more silent failures caught",
             loc="left", color=INK, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(F / "operating_curve.png"); plt.close(fig)

# 2. precision at tight budget ---------------------------------------------
s = pd.read_csv(_setup.RESULTS / "headline_summary.csv")
s5 = s[s.budget == 0.05].set_index("method")
order = ["random", "surface_rf", "rules", "structural_lr"]
labels = ["Random spot-check", "Model on evaluator's own info", "Rule: count fired checks", "Independent signals + LR"]
vals = [s5.loc[m, "precision"] * 100 for m in order]
fig, ax = plt.subplots(figsize=(7.6, 3.2))
bars = ax.barh(labels, vals, color=[LIGHT, GRAY, ORANGE, BLUE], height=0.6)
for b, v in zip(bars, vals):
    ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", color=INK, fontsize=11)
ax.set_xlim(0, 105); ax.set_xlabel("% of escalations that really were failures (review budget: 5%)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Where the model beats the rule: tight budgets", loc="left", color=INK, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(F / "precision_tight_budget.png"); plt.close(fig)

# 3. per-mode vs ceiling ----------------------------------------------------
pm = pd.read_csv(_setup.RESULTS / "per_mode.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = np.arange(len(pm))
ax.bar(x, pm["catch"] * 100, color=BLUE, width=0.6, label="Caught at 20% review")
ax.scatter(x, pm["ceiling"] * 100, color=INK, s=60, zorder=3, marker="_", linewidths=3,
           label="Instrumentation ceiling (check actually ran)")
for xi, v in zip(x, pm["catch"] * 100):
    ax.text(xi, v / 2, f"{v:.0f}%", ha="center", color="white", fontweight="bold")
ax.set_xticks(x, [f"{r.mode}\n" + r.name.replace(" ", "\n", 1) for r in pm.itertuples()], fontsize=9.5)
ax.set_ylim(0, 118); ax.set_ylabel("% of that mode's silent failures")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9.5)
ax.set_title("Modes C and E are capped by logging, not by the model", loc="left", color=INK, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(F / "per_mode_vs_ceiling.png"); plt.close(fig)

# 4. robustness + shift -----------------------------------------------------
rs = pd.read_csv(_setup.RESULTS / "robustness_shift.csv")
h = s[(s.budget == 0.2)].set_index("method")
lab = ["As modeled", "Mild noise", "Moderate", "Severe", "Harder task mix"]
vals = list(rs["mean"] * 100)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
cols = [BLUE] * 4 + ["#4a3aa7"]
b = ax.bar(lab, vals, color=cols, width=0.6, zorder=2)
for bi, v in zip(b, vals):
    ax.text(bi.get_x() + bi.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", color=INK)
for m, name, c in [("surface_rf", "model on evaluator's own info", GRAY), ("random", "random", LIGHT)]:
    ax.axhline(h.loc[m, "catch"] * 100, color=c, ls="--", lw=1.3)
    ax.text(-0.45, h.loc[m, "catch"] * 100 + 0.8, name, ha="left", color=INK2, fontsize=9, bbox=dict(fc="white", ec="none", pad=1))
ax.set_ylim(0, 60); ax.set_ylabel("% silent failures caught @ 20% review")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Degrades gracefully when signals get worse", loc="left", color=INK, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(F / "robustness_and_shift.png"); plt.close(fig)
print("figures written to", F)
