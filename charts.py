"""Generate charts for the Bias Dissociation Benchmark results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- Data ---

models = [
    ("Nova Lite\n(Amazon)", "Amazon", "small"),
    ("Gemini 2.0\nFlash Lite", "Gemini", "small"),
    ("Gemini 3\nFlash", "Gemini", "medium"),
    ("MiniMax\nM2.7", "MiniMax", "medium"),
    ("Kimi K2.5\n(Moonshot)", "Moonshot", "medium"),
]

magnitude_ibi = [0.136, 0.229, 0.346, 0.230, 0.283]
stereotype_ibi = [-0.004, 0.000, 0.000, -0.026, 0.020]
framing_ibi = [-0.000, -0.000, -0.000, -0.000, -0.000]

magnitude_ebr = [1.000, 0.967, 1.000, 0.900, 1.000]
stereotype_ebr = [1.000, 1.000, 1.000, 1.000, 1.000]

magnitude_ds = [0.136, 0.196, 0.346, 0.130, 0.283]

model_labels = [m[0] for m in models]
families = [m[1] for m in models]
tiers = [m[2] for m in models]

# Colors
C_MAG = "#e74c3c"    # red for magnitude/anchoring
C_STEREO = "#3498db"  # blue for stereotype
C_FRAME = "#95a5a6"   # gray for framing
C_EBR = "#2ecc71"     # green for explicit rejection

OUTDIR = "results/charts"
os.makedirs(OUTDIR, exist_ok=True)


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)


# ---------------------------------------------------------------
# Chart 1: The core dissociation — IBI by family across models
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
style_ax(ax)

x = np.arange(len(models))
w = 0.25

bars_mag = ax.bar(x - w, magnitude_ibi, w, label="Magnitude (anchoring)", color=C_MAG, zorder=3)
bars_ste = ax.bar(x, stereotype_ibi, w, label="Stereotype", color=C_STEREO, zorder=3)
bars_fra = ax.bar(x + w, framing_ibi, w, label="Framing", color=C_FRAME, zorder=3)

ax.set_ylabel("Implicit Bias Index (IBI)", fontsize=13)
ax.set_title("The Dissociation: Anchoring Persists, Stereotypes Don't", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=10)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylim(-0.08, 0.42)
ax.legend(fontsize=11, loc="upper left")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/01_dissociation.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved 01_dissociation.png")


# ---------------------------------------------------------------
# Chart 2: The hallway chart — EBR vs IBI for magnitude
# Shows models "know" anchoring is wrong but do it anyway
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
style_ax(ax)

x = np.arange(len(models))
w = 0.32

bars_ebr = ax.bar(x - w/2, magnitude_ebr, w, label="Explicit rejection (EBR)", color=C_EBR, zorder=3)
bars_ibi = ax.bar(x + w/2, magnitude_ibi, w, label="Implicit bias (IBI)", color=C_MAG, zorder=3)

ax.set_ylabel("Score", fontsize=13)
ax.set_title("They Know It's Wrong. They Do It Anyway.", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/02_know_vs_do.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved 02_know_vs_do.png")


# ---------------------------------------------------------------
# Chart 3: Magnitude IBI vs capability — the scaling story
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
style_ax(ax)

# Sort by IBI for visual clarity, but label by model
order = np.argsort(magnitude_ibi)
colors_by_family = {
    "Amazon": "#ff9800",
    "Gemini": "#4285f4",
    "MiniMax": "#9c27b0",
    "Moonshot": "#00bcd4",
}

for i in range(len(models)):
    c = colors_by_family[families[i]]
    marker = "o" if tiers[i] == "small" else "s"
    ax.scatter(i, magnitude_ibi[i], c=c, s=120, zorder=5, marker=marker, edgecolors="white", linewidth=1.5)
    ax.annotate(
        f"{model_labels[i]}\n({families[i]})",
        (i, magnitude_ibi[i]),
        textcoords="offset points",
        xytext=(0, 14),
        ha="center",
        fontsize=9,
    )

# Trend line
z = np.polyfit(range(len(models)), magnitude_ibi, 1)
p = np.poly1d(z)
ax.plot(range(len(models)), p(range(len(models))), "--", color=C_MAG, alpha=0.5, linewidth=1.5, zorder=2)

ax.set_ylabel("Anchoring IBI", fontsize=13)
ax.set_title("Anchoring Across Architectures", fontsize=14, fontweight="bold", pad=12)
ax.set_ylim(0.05, 0.45)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(["" for _ in models])
ax.grid(axis="y", alpha=0.3, zorder=0)

# Legend for markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Small tier"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Medium tier"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

fig.tight_layout()
fig.savefig(f"{OUTDIR}/03_across_architectures.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved 03_across_architectures.png")


# ---------------------------------------------------------------
# Chart 4: Summary "scorecard" — predictions vs observations
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

categories = ["IBI > 0", "Low CAS\n(cross-arch\nstable)", "SG > 0\n(scales with\ncapability)", "High DS\n(dissociation)", "High EBR\n(explicit\nrejection)"]
# Predictions: 1 = yes expected, 0 = no/opposite expected
magnitude_pred = [1, 1, 1, 1, 1]
stereotype_pred = [0, 0, 0, 0, 1]

# Observations (simplified to yes/no)
magnitude_obs = [1, 1, 1, 1, 1]  # all confirmed
stereotype_obs = [0, None, 0, 0, 1]  # CAS uninformative
framing_obs = [0, None, 0, 0, 1]

data = [
    ("Magnitude\n(anchoring)", magnitude_obs, C_MAG),
    ("Stereotype", stereotype_obs, C_STEREO),
    ("Framing", framing_obs, C_FRAME),
]

for ax_i, (title, obs, color) in zip(axes, data):
    style_ax(ax_i)
    ax_i.set_title(title, fontsize=13, fontweight="bold", color=color, pad=10)

    y_pos = np.arange(len(categories))

    for j, (cat, val) in enumerate(zip(categories, obs)):
        if val == 1:
            ax_i.barh(j, 1, color=color, alpha=0.8, height=0.6)
            ax_i.text(0.5, j, "YES", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        elif val == 0:
            ax_i.barh(j, 1, color="#ecf0f1", height=0.6)
            ax_i.text(0.5, j, "NO", ha="center", va="center", fontsize=10, color="#999")
        else:
            ax_i.barh(j, 1, color="#ecf0f1", height=0.6)
            ax_i.text(0.5, j, "N/A", ha="center", va="center", fontsize=10, color="#bbb", style="italic")

    ax_i.set_yticks(y_pos)
    ax_i.set_xlim(0, 1)
    ax_i.set_xticks([])
    ax_i.spines["bottom"].set_visible(False)
    ax_i.spines["left"].set_visible(False)
    ax_i.tick_params(left=False)

for ax_j in axes:
    ax_j.set_yticks(y_pos)
    ax_j.set_yticklabels(categories, fontsize=10)
    ax_j.invert_yaxis()

fig.suptitle("Predictions vs. Observations", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/04_predictions_scorecard.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved 04_predictions_scorecard.png")

print(f"\nAll charts saved to {OUTDIR}/")
