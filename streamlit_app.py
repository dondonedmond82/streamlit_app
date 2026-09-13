"""
World Cup Standings Explorer — a multi-page Streamlit dashboard demonstrating
a wide range of visualizations and machine-learning techniques on the
team_standings.csv dataset (Standing, Team).

Run with:  streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_utils import load_raw, enrich, SIMULATED_COLUMNS

sns.set_theme(style="whitegrid")

st.set_page_config(page_title="World Cup Standings Explorer", layout="wide", page_icon="⚽")


def chart_with_note(note: str, chart_fn, ratio=(3, 1)):
    """Render a chart (via chart_fn, which takes the chart column as arg) with a
    short descriptive note placed beside it."""
    col_chart, col_note = st.columns(ratio)
    with col_chart:
        chart_fn()
    with col_note:
        st.markdown(f"**What this shows**\n\n{note}")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def get_data():
    raw = load_raw("team_standings.csv")
    return enrich(raw)


df = get_data()
numeric_cols = ["Standing"] + SIMULATED_COLUMNS

PAGES = [
    ("🏠 Overview", "🏠"),
    ("📊 Distributions (Pie / Bar / Barh)", "📊"),
    ("📈 Line & Scatter", "📈"),
    ("🔥 Correlation & Heatmap", "🔥"),
    ("🗺️ Geolocation Map", "🗺️"),
    ("📦 Boxplots", "📦"),
    ("📐 Linear Regression", "📐"),
    ("🧠 Classification (Perceptron / Logistic)", "🧠"),
    ("🌀 Clustering (DBSCAN / Hierarchical)", "🌀"),
    ("🔮 Prediction Playground", "🔮"),
]

if "page" not in st.session_state:
    st.session_state.page = PAGES[0][0]

st.sidebar.title("⚽ Navigation")
for label, _icon in PAGES:
    is_active = st.session_state.page == label
    if st.sidebar.button(
        label,
        key=f"nav_{label}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.page = label

page = st.session_state.page

st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ The source file only has **Standing** and **Team**. Confederation and "
    "capital coordinates below are real. All performance metrics ending in "
    "`_sim` (points, goals, possession, ...) are **simulated** (seeded random, "
    "correlated with final standing) purely to demonstrate the analyses — they "
    "are not the actual 2010 World Cup statistics."
)

with st.sidebar.expander("Show enriched dataset"):
    st.dataframe(df, use_container_width=True, height=300)

# ============================================================================
# PAGE: Overview
# ============================================================================
if page == "🏠 Overview":
    st.title("⚽ World Cup Standings Explorer")
    st.markdown(
        "An interactive Streamlit dashboard showcasing many chart types and "
        "machine-learning techniques (regression, classification, clustering) "
        "applied to a final-standings dataset."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams", len(df))
    c2.metric("Confederations", df["Confederation"].nunique())
    c3.metric("Continents", df["Continent"].nunique())
    c4.metric("Champion", df.loc[df["Standing"] == 1, "Team"].values[0])

    col_data, col_note = st.columns((3, 1))
    with col_data:
        st.subheader("Raw data (as uploaded)")
        st.dataframe(load_raw("team_standings.csv"), use_container_width=True)
    with col_note:
        st.markdown(
            "**What this shows**\n\n"
            "The original file exactly as uploaded: just the final `Standing` "
            "and `Team` for each of the 32 teams. Everything else in this app "
            "is derived from these two columns."
        )

    col_data2, col_note2 = st.columns((3, 1))
    with col_data2:
        st.subheader("Enriched data (used throughout this app)")
        st.dataframe(df, use_container_width=True)
    with col_note2:
        st.markdown(
            "**What this shows**\n\n"
            "The working dataset: real confederation/continent/coordinates plus "
            "the simulated `_sim` performance columns and two binary labels "
            "(`Podium`, `Top10`) used later for classification."
        )

    st.download_button(
        "Download enriched dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="team_standings_enriched.csv",
        mime="text/csv",
    )

# ============================================================================
# PAGE: Distributions
# ============================================================================
elif page == "📊 Distributions (Pie / Bar / Barh)":
    st.title("📊 Distributions: Pie, Bar & Horizontal Bar")

    st.subheader("Pie chart — Teams per confederation")
    def _pie():
        pie_data = df["Confederation"].value_counts().reset_index()
        pie_data.columns = ["Confederation", "Count"]
        fig = px.pie(pie_data, names="Confederation", values="Count", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)
    chart_with_note(
        "Share of the 32 qualified teams coming from each football confederation. "
        "UEFA (Europe) sends by far the largest contingent, which is why it "
        "dominates this pie.",
        _pie,
    )

    st.subheader("Bar chart — Final standing per team")
    def _bar():
        fig2 = px.bar(
            df.sort_values("Standing"), x="Team", y="Standing", color="Continent",
            title="Lower is better (1st place = winner)",
        )
        fig2.update_layout(xaxis_tickangle=-60)
        st.plotly_chart(fig2, use_container_width=True)
    chart_with_note(
        "Every team ordered by where it finished. Bars are colored by continent "
        "so you can spot which regions clustered near the top vs. the bottom "
        "of the standings.",
        _bar,
    )

    st.subheader("Horizontal bar chart — Top 15 by simulated points")
    def _barh():
        top15 = df.sort_values("Points_sim", ascending=False).head(15)
        fig3, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top15, y="Team", x="Points_sim", hue="Continent", dodge=False, ax=ax)
        ax.set_xlabel("Simulated points")
        st.pyplot(fig3)
    chart_with_note(
        "The 15 highest scorers on the simulated points metric, drawn as a "
        "horizontal bar so long team names stay readable. Confirms the top of "
        "the points ranking broadly agrees with the real final standing.",
        _barh,
    )

# ============================================================================
# PAGE: Line & Scatter
# ============================================================================
elif page == "📈 Line & Scatter":
    st.title("📈 Line Chart & Scatter Plot")

    st.subheader("Line chart — Simulated points vs. final standing")
    def _line():
        line_df = df.sort_values("Standing")
        fig = px.line(line_df, x="Standing", y="Points_sim", markers=True,
                       hover_name="Team", title="Simulated points as standing worsens")
        st.plotly_chart(fig, use_container_width=True)
    chart_with_note(
        "Traces simulated points as standing goes from 1st to 32nd. The "
        "downward trend (with noise) is expected by construction: points were "
        "simulated to correlate with final standing.",
        _line,
    )

    st.subheader("Scatter plot — Goals For vs. Goals Against")
    def _scatter():
        fig2 = px.scatter(
            df, x="GoalsFor_sim", y="GoalsAgainst_sim", color="Continent",
            size="Points_sim", hover_name="Team", trendline="ols",
            title="Bubble size = simulated points",
        )
        st.plotly_chart(fig2, use_container_width=True)
    chart_with_note(
        "Each bubble is a team. Teams in the bottom-right (many goals for, few "
        "against) scored the most simulated points — shown by bubble size — "
        "and tended to finish higher.",
        _scatter,
    )

    st.subheader("Scatter matrix")
    def _splom():
        fig3 = px.scatter_matrix(
            df, dimensions=["Standing", "Points_sim", "GoalsFor_sim", "Possession_sim"],
            color="Continent",
        )
        st.plotly_chart(fig3, use_container_width=True)
    chart_with_note(
        "A grid of pairwise scatter plots across four numeric columns at once, "
        "useful for spotting relationships (or the lack of one) between any "
        "two variables without building each chart individually.",
        _splom,
    )

# ============================================================================
# PAGE: Correlation & Heatmap
# ============================================================================
elif page == "🔥 Correlation & Heatmap":
    st.title("🔥 Correlation Matrix & Heatmap")

    def _heatmap_static():
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        st.pyplot(fig)
    chart_with_note(
        "Pearson correlation between every numeric column. Values near +1/-1 "
        "(deep red/blue) mean two columns move strongly together or in "
        "opposite directions; values near 0 mean little linear relationship.",
        _heatmap_static,
    )

    st.subheader("Interactive correlation heatmap")
    def _heatmap_plotly():
        corr = df[numeric_cols].corr()
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig2, use_container_width=True)
    chart_with_note(
        "Same correlation matrix as above, but interactive — hover any cell "
        "to read the exact coefficient between that pair of columns.",
        _heatmap_plotly,
    )

    st.subheader("Strongest correlations with Standing")
    corr = df[numeric_cols].corr()
    col_tbl, col_note = st.columns((3, 1))
    with col_tbl:
        st.dataframe(
            corr["Standing"].drop("Standing").sort_values().to_frame("Correlation with Standing")
        )
    with col_note:
        st.markdown(
            "**What this shows**\n\n"
            "Ranks every metric by how strongly it relates to final standing. "
            "Since standing is 1 = best, a negative correlation (e.g. Points) "
            "means higher values go with a better (lower-numbered) finish."
        )

# ============================================================================
# PAGE: Geolocation
# ============================================================================
elif page == "🗺️ Geolocation Map":
    st.title("🗺️ Geolocation — Team capitals")
    st.caption("Marker size = simulated points, color = final standing (darker = better).")

    def _geo():
        fig = px.scatter_geo(
            df, lat="Latitude", lon="Longitude", color="Standing",
            size="Points_sim", hover_name="Team",
            hover_data={"Standing": True, "Confederation": True, "Latitude": False, "Longitude": False},
            color_continuous_scale="Viridis_r", projection="natural earth",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    chart_with_note(
        "Each team is placed on its capital city (real coordinates). Bubble "
        "size shows simulated points and color shows final standing, so you "
        "can see at a glance which regions of the world performed best.",
        _geo,
    )

    st.subheader("Simple map view")
    def _map():
        st.map(df.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon"]])
    chart_with_note(
        "A lightweight built-in Streamlit map showing the same 32 capital "
        "city locations, without any styling by performance — useful as a "
        "quick sanity check on the geolocation data.",
        _map,
    )

# ============================================================================
# PAGE: Boxplots
# ============================================================================
elif page == "📦 Boxplots":
    st.title("📦 Boxplots")

    metric = st.selectbox("Metric to inspect", SIMULATED_COLUMNS, index=0)
    groupby = st.radio("Group by", ["Continent", "Confederation"], horizontal=True)

    def _box_plotly():
        fig = px.box(df, x=groupby, y=metric, color=groupby, points="all", hover_name="Team")
        st.plotly_chart(fig, use_container_width=True)
    chart_with_note(
        f"Distribution of **{metric}** within each {groupby.lower()}. The box "
        "spans the middle 50% of values, the line inside is the median, and "
        "individual dots are the actual teams — handy for spotting outliers.",
        _box_plotly,
    )

    def _box_static():
        fig2, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=groupby, y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig2)
    chart_with_note(
        "A static (Matplotlib/Seaborn) version of the same comparison — "
        "useful if you want a clean image to export or print.",
        _box_static,
    )

# ============================================================================
# PAGE: Linear Regression
# ============================================================================
elif page == "📐 Linear Regression":
    st.title("📐 Linear Regression")
    st.markdown("Predict **simulated points** from other simulated match statistics.")

    features = st.multiselect(
        "Feature(s) (X)", [c for c in SIMULATED_COLUMNS if c != "Points_sim"],
        default=["GoalDiff_sim", "Possession_sim"],
    )
    target = "Points_sim"

    if len(features) == 0:
        st.warning("Select at least one feature.")
    else:
        X = df[features].values
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² (test)", f"{r2_score(y_test, y_pred):.3f}")
        c2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.caption(
            "R² close to 1 means the model explains most of the variance in points; "
            "MAE/RMSE are the average prediction error in points."
        )

        coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        col_tbl, col_note = st.columns((3, 1))
        with col_tbl:
            st.dataframe(coef_df)
            st.write(f"Intercept: {model.intercept_:.3f}")
        with col_note:
            st.markdown(
                "**What this shows**\n\n"
                "Each coefficient is how many points change for a +1 unit "
                "change in that feature, holding the others fixed. Larger "
                "magnitude = bigger influence on the prediction."
            )

        if len(features) == 1:
            def _reg_plot():
                fig = px.scatter(df, x=features[0], y=target, hover_name="Team", trendline="ols",
                                  title=f"{target} vs {features[0]}")
                st.plotly_chart(fig, use_container_width=True)
            chart_with_note(
                f"Raw relationship between {features[0]} and {target}, with the "
                "fitted OLS trendline overlaid.",
                _reg_plot,
            )
        else:
            def _actual_vs_pred():
                pred_full = model.predict(X)
                fig = px.scatter(x=y, y=pred_full, hover_name=df["Team"],
                                  labels={"x": "Actual", "y": "Predicted"},
                                  title="Actual vs. Predicted (full dataset)")
                fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                              line=dict(dash="dash", color="red"))
                st.plotly_chart(fig, use_container_width=True)
            chart_with_note(
                "Each point compares a team's actual simulated points to what "
                "the model predicted. Points sitting exactly on the red "
                "dashed line would be perfect predictions.",
                _actual_vs_pred,
            )

# ============================================================================
# PAGE: Classification
# ============================================================================
elif page == "🧠 Classification (Perceptron / Logistic)":
    st.title("🧠 Classification: Perceptron & Logistic Regression")
    st.markdown("Classify whether a team finished in the **Top 10** using simulated stats.")

    features = st.multiselect(
        "Feature(s)", SIMULATED_COLUMNS,
        default=["Points_sim", "GoalDiff_sim", "Possession_sim"],
    )
    algo = st.radio("Algorithm", ["Perceptron", "Logistic Regression"], horizontal=True)

    if len(features) < 1:
        st.warning("Select at least one feature.")
    else:
        X = df[features].values
        y = df["Top10"].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.3, random_state=42, stratify=y
        )

        if algo == "Perceptron":
            clf = Perceptron(random_state=42, max_iter=1000)
        else:
            clf = LogisticRegression(random_state=42, max_iter=1000)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy (test)", f"{accuracy_score(y_test, y_pred):.2%}")
        c2.write(f"Train size: {len(X_train)} · Test size: {len(X_test)}")

        def _cm():
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Not Top10", "Top10"], yticklabels=["Not Top10", "Top10"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
        chart_with_note(
            "Rows are actual classes, columns are predicted classes. The "
            "diagonal cells are correct predictions; anything off-diagonal "
            "is a mistake (false positive/negative).",
            _cm,
            ratio=(2, 1),
        )

        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred, target_names=["Not Top10", "Top10"], zero_division=0))
        st.caption(
            "Precision = of teams predicted Top10, how many really were. "
            "Recall = of teams that really were Top10, how many the model caught."
        )

        if len(features) >= 2:
            st.subheader("Decision boundary (first 2 selected features, full-data fit)")
            def _boundary():
                clf2 = clf.__class__(**clf.get_params()).fit(Xs[:, :2], y)
                x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
                y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
                Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")
                sns.scatterplot(x=Xs[:, 0], y=Xs[:, 1], hue=y, palette="RdYlGn", ax=ax2, edgecolor="k")
                ax2.set_xlabel(features[0] + " (scaled)")
                ax2.set_ylabel(features[1] + " (scaled)")
                st.pyplot(fig2)
            chart_with_note(
                "Shows the boundary the model draws between the two classes "
                "using only the first two selected features (scaled). Dots "
                "are real teams colored by their true class.",
                _boundary,
            )

# ============================================================================
# PAGE: Clustering
# ============================================================================
elif page == "🌀 Clustering (DBSCAN / Hierarchical)":
    st.title("🌀 Clustering: DBSCAN & Hierarchical")

    features = st.multiselect(
        "Feature(s) for clustering", SIMULATED_COLUMNS,
        default=["Points_sim", "GoalDiff_sim", "Possession_sim", "PassAccuracy_sim"],
    )

    if len(features) < 2:
        st.warning("Select at least two features.")
    else:
        X = df[features].values
        Xs = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(Xs)

        tab1, tab2 = st.tabs(["DBSCAN", "Hierarchical (dendrogram)"])

        with tab1:
            eps = st.slider("eps", 0.2, 3.0, 1.0, 0.1)
            min_samples = st.slider("min_samples", 2, 10, 3)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            st.write(f"Clusters found: **{n_clusters}** · Noise points: **{(labels == -1).sum()}**")

            def _dbscan_plot():
                plot_df = pd.DataFrame({
                    "PC1": X2[:, 0], "PC2": X2[:, 1],
                    "Cluster": labels.astype(str), "Team": df["Team"],
                })
                fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", hover_name="Team",
                                  title="DBSCAN clusters (PCA-reduced, 2D)")
                st.plotly_chart(fig, use_container_width=True)
            chart_with_note(
                "Teams projected onto 2 principal components (PCA) and colored "
                "by DBSCAN cluster. Cluster '-1' means DBSCAN flagged that team "
                "as noise (didn't fit densely into any group). Adjust eps/"
                "min_samples above to make clusters looser or tighter.",
                _dbscan_plot,
            )

        with tab2:
            method = st.selectbox("Linkage method", ["ward", "average", "complete", "single"])
            def _dendro():
                Z = linkage(Xs, method=method)
                fig2, ax = plt.subplots(figsize=(10, 6))
                dendrogram(Z, labels=df["Team"].values, ax=ax, leaf_rotation=90)
                ax.set_title(f"Hierarchical clustering dendrogram ({method} linkage)")
                st.pyplot(fig2)
            chart_with_note(
                "A tree showing how teams merge into groups at increasing "
                "distance thresholds. Teams that join together low down (short "
                "vertical distance) are the most similar on the chosen features.",
                _dendro,
                ratio=(3, 1),
            )

# ============================================================================
# PAGE: Prediction Playground
# ============================================================================
elif page == "🔮 Prediction Playground":
    st.title("🔮 Prediction Playground")
    st.markdown(
        "Train a regression model on the simulated stats, then enter your own "
        "values to predict a team's simulated **points** and estimated **standing**."
    )

    features = ["GoalsFor_sim", "GoalsAgainst_sim", "Possession_sim", "PassAccuracy_sim", "ShotsOnTarget_sim"]
    X = df[features].values
    y_points = df["Points_sim"].values
    y_standing = df["Standing"].values

    reg_points = LinearRegression().fit(X, y_points)
    reg_standing = LinearRegression().fit(X, y_standing)

    st.subheader("Enter match statistics")
    st.caption(
        "Fields are pre-filled with the dataset average. Change any value and "
        "click Predict to see the model's estimate."
    )
    cols = st.columns(len(features))
    inputs = []
    defaults = df[features].mean()
    for c, feat in zip(cols, features):
        val = c.number_input(feat, value=float(defaults[feat]))
        inputs.append(val)

    if st.button("Predict"):
        x_new = np.array(inputs).reshape(1, -1)
        pred_points = reg_points.predict(x_new)[0]
        pred_standing = reg_standing.predict(x_new)[0]
        c1, c2 = st.columns(2)
        c1.metric("Predicted points", f"{pred_points:.1f}")
        c2.metric("Predicted standing", f"{pred_standing:.1f}")
        st.caption("Predictions are based on the simulated training data — for demonstration only.")

st.markdown("---")
st.caption(
    "Built with Streamlit · scikit-learn · Plotly · Seaborn. "
    "Dataset: team_standings.csv (Standing, Team) enriched with real country "
    "metadata and simulated performance statistics."
)
