"""
Titanic Dataset - Data Visualization Assignment
================================================
Part 1: Visualization for Numerical Data   (Age, Fare)
Part 2: Visualization for Categorical Data (Survived, Sex, Pclass, Embarked)
Part 3: Design quality - consistent palette, labelled axes with units,
        titles, legends, annotations, zero-based axes (no truncation).

Requirements:  pip install pandas matplotlib numpy
Usage:         put Titanic-Dataset.csv next to this file and run
               python titanic_visualization.py
Output:        PNG charts are saved in the "charts" folder and also shown.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# ------------------------------------------------------------------
# 0. Setup: load data + one shared style for every chart
# ------------------------------------------------------------------
CSV_PATH = "Titanic-Dataset.csv"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Readable labels for coded columns
df["Survival"] = df["Survived"].map({0: "Did not survive", 1: "Survived"})
df["Class"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
df["Port"] = df["Embarked"].map(
    {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
)
df["Sex"] = df["Sex"].str.capitalize()

# Colour-blind-safe palette, used consistently everywhere
SURVIVED = "#2b7bba"      # blue  -> survived
DIED = "#e07b39"          # orange -> did not survive
NEUTRAL = "#5b6770"       # grey  -> single-series charts
FEMALE = "#8e5ea2"        # purple -> female
MALE = "#2a9d8f"          # teal   -> male
CLASS_COLORS = ["#1b4f72", "#5499c7", "#a9cce3"]   # 1st -> 3rd (dark -> light)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    print(f"saved -> {OUT_DIR}/{name}")


# Quick overview printed to console (useful for the write-up)
print("Rows, columns:", df.shape)
print("\nMissing values:\n", df.isna().sum()[lambda s: s > 0])
print("\nNumerical summary:\n", df[["Age", "Fare"]].describe().round(2))
print(f"\nOverall survival rate: {df['Survived'].mean():.1%}")


# ==================================================================
# PART 1 - NUMERICAL DATA
# ==================================================================

# ---- Chart 1: Histogram - Age distribution by survival -------------
ages = df.dropna(subset=["Age"])
bins = np.arange(0, 85, 5)   # 5-year bins from 0 to 80

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    [ages.loc[ages.Survived == 0, "Age"], ages.loc[ages.Survived == 1, "Age"]],
    bins=bins, stacked=True, color=[DIED, SURVIVED],
    edgecolor="white", label=["Did not survive", "Survived"],
)
median_age = ages["Age"].median()
ax.axvline(median_age, color="black", linestyle="--", linewidth=1.2)
ax.text(median_age + 1, ax.get_ylim()[1] * 0.92,
        f"Median age = {median_age:.0f} yrs", fontsize=10)
ax.annotate("Children under 10 had the\nbest survival odds",
            xy=(5, 30), xytext=(12, 75), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_title("Age Distribution of Titanic Passengers by Survival")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of passengers")
ax.set_xticks(bins)
ax.set_ylim(bottom=0)
ax.legend(title="Outcome")
fig.text(0.01, -0.02, f"Note: {df['Age'].isna().sum()} passengers with "
         "unknown age are excluded. Bins are 5 years wide.",
         fontsize=9, color="dimgray")
save(fig, "1_age_histogram.png")

# ---- Chart 2: Box plot - Fare by passenger class -------------------
fig, ax = plt.subplots(figsize=(10, 6))
fare_groups = [df.loc[df.Pclass == c, "Fare"] for c in (1, 2, 3)]
bp = ax.boxplot(
    fare_groups, patch_artist=True, widths=0.55,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
ax.set_xticks([1, 2, 3], ["1st Class", "2nd Class", "3rd Class"])
for patch, color in zip(bp["boxes"], CLASS_COLORS):
    patch.set_facecolor(color)
for i, g in enumerate(fare_groups, start=1):
    ax.text(i + 0.3, g.median(), f"median\n£{g.median():.1f}",
            va="center", fontsize=9)
top = df["Fare"].max()
ax.annotate(f"Outliers: £{top:.0f} tickets\n(highest fare paid)",
            xy=(1, top), xytext=(1.6, top - 60), fontsize=10,
            arrowprops=dict(arrowstyle="->"))
ax.set_title("Ticket Fare Distribution by Passenger Class")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Fare (British pounds, £)")
ax.set_ylim(bottom=0)
save(fig, "2_fare_boxplot.png")

# ---- Chart 3: Scatter plot - Age vs Fare, coloured by survival -----
fig, ax = plt.subplots(figsize=(10, 6))
for val, color, label in [(0, DIED, "Did not survive"), (1, SURVIVED, "Survived")]:
    sub = ages[ages.Survived == val]
    ax.scatter(sub["Age"], sub["Fare"], c=color, label=label,
               alpha=0.6, s=28, edgecolors="white", linewidths=0.4)
ax.set_title("Age vs Ticket Fare: Higher Fares Linked to Higher Survival")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Fare (British pounds, £)")
ax.set_xlim(0, 82)
ax.set_ylim(bottom=0)
ax.legend(title="Outcome", loc="upper right")
corr = ages["Age"].corr(ages["Fare"])
ax.text(0.02, 0.95, f"Pearson r (Age, Fare) = {corr:.2f}  -> very weak link",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="lightgray"))
save(fig, "3_age_fare_scatter.png")


# ==================================================================
# PART 2 - CATEGORICAL DATA
# ==================================================================

# ---- Chart 4: Bar chart - Survival rate by sex ---------------------
rate_sex = df.groupby("Sex")["Survived"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(rate_sex.index, rate_sex["mean"], color=[FEMALE, MALE],
              width=0.55, edgecolor="black", linewidth=0.6)
for bar, (_, row) in zip(bars, rate_sex.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, row["mean"] + 0.02,
            f"{row['mean']:.1%}\n(n = {int(row['count'])})",
            ha="center", fontsize=11, fontweight="bold")
overall = df["Survived"].mean()
ax.axhline(overall, color="black", linestyle="--", linewidth=1,
           label=f"Overall survival rate: {overall:.1%}")
ax.legend(loc="upper right")
ax.set_title("Survival Rate by Sex: Women Were ~4x More Likely to Survive")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival rate (% of passengers)")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
save(fig, "4_survival_by_sex_bar.png")

# ---- Chart 5: Grouped bar chart - Survival rate by class and sex ---
rate_cs = df.pivot_table(index="Class", columns="Sex",
                         values="Survived", aggfunc="mean")
x = np.arange(len(rate_cs))
w = 0.38
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w / 2, rate_cs["Female"], w, label="Female",
            color=FEMALE, edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + w / 2, rate_cs["Male"], w, label="Male",
            color=MALE, edgecolor="black", linewidth=0.6)
for bars_ in (b1, b2):
    ax.bar_label(bars_, labels=[f"{v:.0%}" for v in bars_.datavalues],
                 padding=3, fontsize=10)
ax.set_xticks(x, rate_cs.index)
ax.set_title("Survival Rate by Passenger Class and Sex")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Survival rate (% of passengers)")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(title="Sex", loc="upper right")
save(fig, "5_survival_by_class_sex_grouped_bar.png")

# ---- Chart 6: Stacked 100% bar - outcome share within each class ----
counts = pd.crosstab(df["Class"], df["Survival"])
share = counts.div(counts.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(10, 5))
left = np.zeros(len(share))
for col, color in [("Survived", SURVIVED), ("Did not survive", DIED)]:
    bars = ax.barh(share.index, share[col], left=left, color=color,
                   edgecolor="white", label=col)
    for bar, pct, n in zip(bars, share[col], counts[col]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.0%} ({n})", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
    left += share[col].values
ax.invert_yaxis()   # 1st class on top
ax.set_title("Share of Passengers Who Survived, Within Each Class")
ax.set_xlabel("Share of passengers in class (%)  -  labels show % (count)")
ax.set_ylabel("Passenger class")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.grid(axis="y", visible=False)
ax.legend(title="Outcome", ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.18))
save(fig, "6_outcome_by_class_stacked_bar.png")

# ---- Chart 7: Pie chart - Port of embarkation ----------------------
port_counts = df["Port"].value_counts()   # 2 missing values dropped
fig, ax = plt.subplots(figsize=(8, 7))
wedges, _, autotexts = ax.pie(
    port_counts, labels=None, startangle=90, counterclock=False,
    colors=["#2b7bba", "#e07b39", "#7fb069"],
    autopct=lambda p: f"{p:.1f}%\n({round(p * port_counts.sum() / 100)})",
    pctdistance=0.72, wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(color="white", fontsize=11, fontweight="bold"),
)
ax.legend(wedges, port_counts.index, title="Port of embarkation",
          loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_title("Passengers by Port of Embarkation\n(share of passengers, count in brackets)")
ax.axis("equal")
fig.text(0.01, 0.01, f"Note: {df['Embarked'].isna().sum()} passengers with "
         "unknown port excluded.", fontsize=9, color="dimgray")
save(fig, "7_embarkation_pie.png")


# ==================================================================
# PART 3 - SUMMARY DASHBOARD (all key insights on one page)
# ==================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Titanic Passenger Survival - Key Insights (n = 891)",
             fontsize=18, fontweight="bold")

# (a) Age histogram
ax = axes[0, 0]
ax.hist([ages.loc[ages.Survived == 0, "Age"], ages.loc[ages.Survived == 1, "Age"]],
        bins=bins, stacked=True, color=[DIED, SURVIVED], edgecolor="white",
        label=["Did not survive", "Survived"])
ax.set_title("Age distribution (numerical)")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of passengers")
ax.legend()

# (b) Fare box plot
ax = axes[0, 1]
bp = ax.boxplot(fare_groups, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5))
ax.set_xticks([1, 2, 3], ["1st", "2nd", "3rd"])
for patch, color in zip(bp["boxes"], CLASS_COLORS):
    patch.set_facecolor(color)
ax.set_title("Fare by class (numerical)")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Fare (£)")
ax.set_ylim(bottom=0)

# (c) Survival by sex
ax = axes[1, 0]
bars = ax.bar(rate_sex.index, rate_sex["mean"], color=[FEMALE, MALE],
              width=0.55, edgecolor="black", linewidth=0.6)
ax.bar_label(bars, labels=[f"{v:.0%}" for v in rate_sex["mean"]],
             padding=3, fontweight="bold")
ax.set_title("Survival rate by sex (categorical)")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

# (d) Survival by class & sex
ax = axes[1, 1]
b1 = ax.bar(x - w / 2, rate_cs["Female"], w, label="Female",
            color=FEMALE, edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + w / 2, rate_cs["Male"], w, label="Male",
            color=MALE, edgecolor="black", linewidth=0.6)
for bars_ in (b1, b2):
    ax.bar_label(bars_, labels=[f"{v:.0%}" for v in bars_.datavalues],
                 padding=3, fontsize=9)
ax.set_xticks(x, ["1st", "2nd", "3rd"])
ax.set_title("Survival rate by class & sex (categorical)")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(title="Sex")

fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "8_summary_dashboard.png")

# ------------------------------------------------------------------
# Key insights (printed for the report)
# ------------------------------------------------------------------
print("\nKEY INSIGHTS")
print(f"- Only {overall:.1%} of passengers survived.")
print(f"- Women survived at {rate_sex.loc['Female', 'mean']:.1%} vs men at "
      f"{rate_sex.loc['Male', 'mean']:.1%}.")
print(f"- 1st-class survival {share.loc['1st Class', 'Survived']:.0%} vs "
      f"3rd-class {share.loc['3rd Class', 'Survived']:.0%}.")
print(f"- Median fare: 1st £{fare_groups[0].median():.1f}, "
      f"2nd £{fare_groups[1].median():.1f}, 3rd £{fare_groups[2].median():.1f}; "
      f"max fare £{top:.0f} is an extreme outlier.")
print(f"- Most passengers boarded at Southampton "
      f"({port_counts.iloc[0] / port_counts.sum():.0%}).")

plt.show()