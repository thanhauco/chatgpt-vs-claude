"""Streamlit dashboard: ChatGPT vs Claude verbosity comparison."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbosity_lab import analysis, config, scoring  # noqa: E402

st.set_page_config(page_title="ChatGPT vs Claude — Verbosity Lab", layout="wide", page_icon="📊")

PROVIDER_COLORS = {"openai": "#10a37f", "anthropic": "#d97757"}


@st.cache_data
def load_default() -> pd.DataFrame:
    path = config.DATA_DIR / "sample_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


st.title("📊 ChatGPT vs Claude — RLHF Verbosity Lab")
st.caption(
    "Quantify verbosity-bias, prompt-engineering mitigations, temperature effects, "
    "and RLHF transparency."
)

# --- Sidebar ---------------------------------------------------------------- #
st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload results CSV", type="csv")
df = pd.read_csv(uploaded) if uploaded else load_default()

if df.empty:
    st.warning("No data found. Run `python scripts/generate_sample_data.py` first.")
    st.stop()

providers = sorted(df["provider"].unique())
sel_providers = st.sidebar.multiselect("Providers", providers, default=providers)
cats = sorted(df["category"].unique())
sel_cats = st.sidebar.multiselect("Prompt categories", cats, default=cats)

df = df[df["provider"].isin(sel_providers) & df["category"].isin(sel_cats)]
if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

df = scoring.add_padding_score(df)
st.sidebar.markdown("---")
st.sidebar.info(
    "Default data is **synthetic** (mock providers). Add API keys and run "
    "`scripts/run_experiment.py` for real model data."
)

(tab_overview, tab_analysis, tab_bias, tab_tech, tab_temp, tab_transp, tab_data) = st.tabs(
    ["Overview", "Analysis", "Verbosity-bias", "Technique audit", "Temperature",
     "RLHF transparency", "Data"]
)

# --- Overview --------------------------------------------------------------- #
with tab_overview:
    summary = scoring.summarize(df)
    st.subheader("Headline metrics by provider")
    cols = st.columns(len(summary))
    for col, (_, r) in zip(cols, summary.iterrows()):
        col.metric(f"{r['provider']} · avg words", f"{r['avg_words']:.0f}")
        col.metric("padding score (0-100)", f"{r['padding_score']:.0f}")
        col.metric("hedge density", f"{r['avg_hedge_density']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            summary, x="provider", y="avg_words", color="provider",
            color_discrete_map=PROVIDER_COLORS, title="Average words per response",
            text_auto=".0f",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            summary, x="provider", y="padding_score", color="provider",
            color_discrete_map=PROVIDER_COLORS, title="Average padding score",
            text_auto=".0f",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True, hide_index=True)

# --- Analysis --------------------------------------------------------------- #
with tab_analysis:
    st.subheader("Statistical comparison (response length)")
    cmp = analysis.compare_providers(df, metric="word_count")
    if cmp:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"{cmp['a']} mean words", f"{cmp['mean_a']:.0f}",
            f"{cmp['pct_diff']:+.0f}% vs {cmp['b']}",
        )
        c2.metric("Cohen's d (effect size)", f"{cmp['cohens_d']:.2f}", cmp["effect"])
        pval = "n/a" if cmp["p_value"] is None else f"{cmp['p_value']:.4f}"
        c3.metric("Welch p-value", pval)
        more = cmp["a"] if cmp["diff"] > 0 else cmp["b"]
        less = cmp["b"] if cmp["diff"] > 0 else cmp["a"]
        st.markdown(
            f"On identical prompts, **{more}** averages **{abs(cmp['diff']):.0f} more words** "
            f"than **{less}** (effect size: _{cmp['effect']}_)."
        )

    st.markdown("---")
    st.subheader("Most effective conciseness techniques")
    eff = analysis.technique_effectiveness(df)
    fig = px.bar(
        eff, x="reduction_pct", y="technique", orientation="h",
        color="reduction_pct", color_continuous_scale="Greens",
        title="Word-count reduction vs baseline (%)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    recipe = analysis.best_recipe(df)
    if recipe:
        st.success(
            f"**Recommended recipe:** “{recipe['technique']}” cuts output by "
            f"~{recipe['reduction_pct']:.0f}% vs an unconstrained prompt."
        )

    if "answer_coverage" in df.columns:
        st.markdown("---")
        st.subheader("Conciseness vs. answer coverage")
        st.caption(
            "Top-left is best: few words, core answer retained. Bubble size = signal efficiency."
        )
        eff2 = (
            df.groupby(["technique", "provider"])
            .agg(
                words=("word_count", "mean"),
                coverage=("answer_coverage", "mean"),
                efficiency=("signal_efficiency", "mean"),
            )
            .reset_index()
        )
        fig = px.scatter(
            eff2, x="words", y="coverage", color="provider", text="technique",
            size="efficiency", color_discrete_map=PROVIDER_COLORS,
            title="Words vs. core-answer coverage by technique",
        )
        fig.update_traces(textposition="top center")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Temperature sensitivity")
    st.caption("Slope = extra words per +1.0 temperature. Higher = more length sensitivity.")
    st.dataframe(analysis.temperature_trend(df), use_container_width=True, hide_index=True)

# --- Verbosity-bias --------------------------------------------------------- #
with tab_bias:
    st.subheader("Verbosity-bias: extra words vs the leanest answer to the same prompt")
    st.caption("`verbosity_bias = (words − min_words_for_prompt) / min_words_for_prompt`")
    fig = px.box(
        df, x="provider", y="verbosity_bias", color="provider",
        color_discrete_map=PROVIDER_COLORS, points="all",
    )
    st.plotly_chart(fig, use_container_width=True)

    by_cat = df.groupby(["category", "provider"])["verbosity_bias"].mean().reset_index()
    fig = px.bar(
        by_cat, x="category", y="verbosity_bias", color="provider", barmode="group",
        color_discrete_map=PROVIDER_COLORS, title="Mean verbosity-bias by prompt category",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Technique audit -------------------------------------------------------- #
with tab_tech:
    st.subheader("Prompt-engineering audit: how much each technique trims output")
    order = (
        df.drop_duplicates("technique")
        .sort_values("conciseness_level")["technique"]
        .tolist()
    )
    by_tech = df.groupby(["technique", "provider"])["word_count"].mean().reset_index()
    fig = px.bar(
        by_tech, x="technique", y="word_count", color="provider", barmode="group",
        color_discrete_map=PROVIDER_COLORS, category_orders={"technique": order},
        title="Average words per response by technique",
    )
    st.plotly_chart(fig, use_container_width=True)

    base = df[df["technique_id"] == "baseline"].groupby("provider")["word_count"].mean()
    red = df.groupby(["provider", "technique", "conciseness_level"])["word_count"].mean().reset_index()
    red["reduction_%"] = red.apply(
        lambda r: round(100 * (1 - r["word_count"] / base.get(r["provider"], r["word_count"])), 1),
        axis=1,
    )
    red = red.sort_values(["provider", "conciseness_level"])
    st.markdown("**Word-count reduction vs each provider's baseline**")
    st.dataframe(
        red[["provider", "technique", "word_count", "reduction_%"]],
        use_container_width=True, hide_index=True,
    )

# --- Temperature ------------------------------------------------------------ #
with tab_temp:
    st.subheader("Impact of temperature on response conciseness")
    by_temp = df.groupby(["temperature", "provider"]).agg(
        avg_words=("word_count", "mean"),
        avg_hedge_density=("hedge_density", "mean"),
    ).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            by_temp, x="temperature", y="avg_words", color="provider", markers=True,
            color_discrete_map=PROVIDER_COLORS, title="Avg words vs temperature",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(
            by_temp, x="temperature", y="avg_hedge_density", color="provider", markers=True,
            color_discrete_map=PROVIDER_COLORS, title="Hedge density vs temperature",
        )
        st.plotly_chart(fig, use_container_width=True)

# --- RLHF transparency ------------------------------------------------------ #
with tab_transp:
    st.subheader("RLHF transparency scorecard")
    table, cfg = scoring.transparency_table()
    st.caption(cfg.get("meta", {}).get("disclaimer", ""))

    label_cols = [c for c in table.columns if c not in ("dimension", "weight", "description")]
    radar = go.Figure()
    for label in label_cols:
        radar.add_trace(go.Scatterpolar(
            r=table[label].tolist() + [table[label].tolist()[0]],
            theta=table["dimension"].tolist() + [table["dimension"].tolist()[0]],
            fill="toself", name=label,
        ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title=f"Transparency by dimension ({cfg.get('meta', {}).get('scale', '0-5')})",
    )
    st.plotly_chart(radar, use_container_width=True)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.info("Scores are an editable illustrative rubric. Edit `config/rlhf_transparency.yaml`.")

# --- Data ------------------------------------------------------------------- #
with tab_data:
    st.subheader("Raw responses & metrics")
    show_cols = [
        "provider", "prompt_id", "category", "technique", "temperature",
        "word_count", "token_count", "hedge_density", "caveat_density",
        "answer_coverage", "signal_efficiency",
        "verbosity_bias", "padding_score", "response",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        df.to_csv(index=False).encode("utf-8"),
        "verbosity_results.csv",
        "text/csv",
    )
