"""
Titanic Interactive Dashboard (Streamlit + Plotly)
==================================================
Requirements:  pip install streamlit plotly pandas
Usage:         put Titanic-Dataset.csv next to this file and run
               streamlit run titanic_dashboard.py
It opens in the browser at http://localhost:8501
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Titanic Dashboard", page_icon="🚢", layout="wide")

# ------------------------------------------------------------------
# Colours (same meaning on every chart)
# ------------------------------------------------------------------
OUTCOME_COLORS = {"Survived": "#2b7bba", "Did not survive": "#e07b39"}
SEX_COLORS = {"Female": "#8e5ea2", "Male": "#2a9d8f"}
CLASS_COLORS = {"1st Class": "#1b4f72", "2nd Class": "#5499c7", "3rd Class": "#a9cce3"}
PORT_COLORS = {"Southampton": "#2b7bba", "Cherbourg": "#e07b39", "Queenstown": "#7fb069"}
CLASS_ORDER = ["1st Class", "2nd Class", "3rd Class"]
OUTCOME_ORDER = ["Survived", "Did not survive"]


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
@st.cache_data
def load_data(path="Titanic-Dataset.csv"):
    df = pd.read_csv(path)
    df["Outcome"] = df["Survived"].map({0: "Did not survive", 1: "Survived"})
    df["Class"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    df["Port"] = df["Embarked"].map(
        {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    )
    df["Sex"] = df["Sex"].str.capitalize()
    return df


df = load_data()


def style(fig, height=420):
    """Shared clean layout for every chart."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        title_font_size=17,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        legend_title_text="",
    )
    return fig


# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
classes = st.sidebar.multiselect("Passenger class", CLASS_ORDER, default=CLASS_ORDER)
sexes = st.sidebar.multiselect("Sex", ["Female", "Male"], default=["Female", "Male"])
ports = st.sidebar.multiselect(
    "Port of embarkation", list(PORT_COLORS), default=list(PORT_COLORS)
)
age_range = st.sidebar.slider("Age range (years)", 0, 80, (0, 80))
include_unknown_age = st.sidebar.checkbox("Include passengers with unknown age", True)

age_ok = df["Age"].between(*age_range)
if include_unknown_age:
    age_ok |= df["Age"].isna()
port_ok = df["Port"].isin(ports) | df["Port"].isna()

f = df[df["Class"].isin(classes) & df["Sex"].isin(sexes) & port_ok & age_ok]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(f)}** of {len(df)} passengers")

if f.empty:
    st.warning("No passengers match these filters. Widen the selection in the sidebar.")
    st.stop()

# ------------------------------------------------------------------
# Header + KPI tiles
# ------------------------------------------------------------------
st.title("🚢 Titanic Passenger Survival Dashboard")
st.caption("Dataset: 891 passengers of RMS Titanic (1912). "
           "Use the sidebar to filter; every chart updates.")

overall = df["Survived"].mean()
rate = f["Survived"].mean()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Passengers", f"{len(f):,}")
k2.metric("Survival rate", f"{rate:.1%}",
          delta=f"{(rate - overall) * 100:+.1f} pts vs all passengers")
k3.metric("Median age", f"{f['Age'].median():.0f} yrs" if f["Age"].notna().any() else "n/a")
k4.metric("Median fare", f"£{f['Fare'].median():.2f}")

st.markdown("---")

tab_num, tab_cat, tab_data = st.tabs(
    ["📈 Numerical data", "📊 Categorical data", "🗂️ Data table"]
)

# ==================================================================
# TAB 1 - NUMERICAL
# ==================================================================
with tab_num:
    c1, c2 = st.columns(2)

    # Histogram - Age
    with c1:
        ages = f.dropna(subset=["Age"])
        fig = px.histogram(
            ages, x="Age", color="Outcome", barmode="stack",
            color_discrete_map=OUTCOME_COLORS,
            category_orders={"Outcome": OUTCOME_ORDER},
            title="Age Distribution by Survival",
            labels={"Age": "Age (years)"},
        )
        fig.update_traces(xbins=dict(start=0, end=80, size=5),
                          marker_line_color="white", marker_line_width=1)
        if not ages.empty:
            fig.add_vline(x=ages["Age"].median(), line_dash="dash",
                          annotation_text=f"Median {ages['Age'].median():.0f} yrs")
        fig.update_yaxes(title="Number of passengers", rangemode="tozero")
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption(f"5-year bins. {f['Age'].isna().sum()} passengers with "
                   "unknown age not shown.")

    # Box plot - Fare by class
    with c2:
        fig = px.box(
            f, x="Class", y="Fare", color="Class",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Class": CLASS_ORDER},
            title="Ticket Fare by Passenger Class",
            labels={"Class": "Passenger class", "Fare": "Fare (£)"},
            points="outliers",
        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption("Box = middle 50% of fares, line = median, dots = outliers.")

    # Scatter - Age vs Fare
    fig = px.scatter(
        ages, x="Age", y="Fare", color="Outcome", symbol="Sex",
        color_discrete_map=OUTCOME_COLORS,
        category_orders={"Outcome": OUTCOME_ORDER},
        hover_data=["Name", "Class"], opacity=0.7,
        title="Age vs Fare (hover for passenger details)",
        labels={"Age": "Age (years)", "Fare": "Fare (£)"},
    )
    fig.update_xaxes(range=[0, 82])
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(style(fig, 460), use_container_width=True)
    if len(ages) > 2:
        st.caption(f"Pearson correlation (Age, Fare) = "
                   f"{ages['Age'].corr(ages['Fare']):.2f}: almost no linear link.")

# ==================================================================
# TAB 2 - CATEGORICAL
# ==================================================================
with tab_cat:
    c1, c2 = st.columns(2)

    # Bar - survival rate by sex
    with c1:
        by_sex = f.groupby("Sex", as_index=False).agg(
            rate=("Survived", "mean"), n=("Survived", "size"))
        fig = px.bar(
            by_sex, x="Sex", y="rate", color="Sex", color_discrete_map=SEX_COLORS,
            text=by_sex.apply(lambda r: f"{r.rate:.0%} (n={r.n})", axis=1),
            title="Survival Rate by Sex",
            labels={"rate": "Survival rate (%)"},
        )
        fig.add_hline(y=overall, line_dash="dash",
                      annotation_text=f"All passengers {overall:.0%}")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig), use_container_width=True)

    # Grouped bar - class x sex
    with c2:
        by_cs = f.groupby(["Class", "Sex"], as_index=False)["Survived"].mean()
        fig = px.bar(
            by_cs, x="Class", y="Survived", color="Sex", barmode="group",
            color_discrete_map=SEX_COLORS, text_auto=".0%",
            category_orders={"Class": CLASS_ORDER},
            title="Survival Rate by Class and Sex",
            labels={"Class": "Passenger class", "Survived": "Survival rate (%)"},
        )
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        st.plotly_chart(style(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    # 100% stacked bar - outcome within class
    with c3:
        counts = f.groupby(["Class", "Outcome"], as_index=False).size()
        counts["share"] = counts["size"] / counts.groupby("Class")["size"].transform("sum")
        fig = px.bar(
            counts, y="Class", x="share", color="Outcome", orientation="h",
            color_discrete_map=OUTCOME_COLORS,
            category_orders={"Class": CLASS_ORDER, "Outcome": OUTCOME_ORDER},
            text=counts.apply(lambda r: f"{r.share:.0%} ({r['size']})", axis=1),
            title="Outcome Share Within Each Class",
            labels={"share": "Share of passengers in class (%)",
                    "Class": "Passenger class"},
        )
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(style(fig), use_container_width=True)

    # Pie - port of embarkation
    with c4:
        port = f["Port"].value_counts().reset_index()
        port.columns = ["Port", "Passengers"]
        fig = px.pie(
            port, names="Port", values="Passengers", color="Port",
            color_discrete_map=PORT_COLORS, hole=0.35,
            title="Passengers by Port of Embarkation",
        )
        fig.update_traces(textinfo="percent+value", textfont_size=13)
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption(f"{f['Port'].isna().sum()} passengers with unknown port not shown.")

# ==================================================================
# TAB 3 - DATA
# ==================================================================
with tab_data:
    cols = ["PassengerId", "Name", "Sex", "Age", "Class", "Fare", "Port", "Outcome"]
    st.dataframe(f[cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)",
                       f[cols].to_csv(index=False), "titanic_filtered.csv", "text/csv")

# ------------------------------------------------------------------
# Key insights (computed from the full dataset)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("💡 Key insights")
sex_rate = df.groupby("Sex")["Survived"].mean()
cls_rate = df.groupby("Class")["Survived"].mean()
st.markdown(
    f"""
- Only **{overall:.0%}** of all passengers survived.
- **Women survived at {sex_rate['Female']:.0%}** compared with **{sex_rate['Male']:.0%} for men**
  ("women and children first").
- **Class mattered:** {cls_rate['1st Class']:.0%} of 1st-class passengers survived
  vs {cls_rate['3rd Class']:.0%} in 3rd class.
- Fares were highly skewed: 1st-class median £{df.loc[df.Pclass == 1, 'Fare'].median():.0f}
  vs 3rd-class £{df.loc[df.Pclass == 3, 'Fare'].median():.0f}; a few tickets cost £{df['Fare'].max():.0f}.
- Most passengers (**{(df['Port'] == 'Southampton').mean():.0%}**) boarded at Southampton.
"""
)