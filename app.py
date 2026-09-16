"""
app.py

The Streamlit demo for the Flood CVaR model.

This app is deliberately narrow in scope: it only asks for the handful of
inputs the calculation actually needs (see calc_engine/ for what each one
feeds into), and it only shows the results. It exists to demonstrate the
CALCULATION -- not to reproduce the full Rozvi platform experience. The
dev team's job, once this is handed over, is to combine this calculation
engine with the Rozvi risk-scoring model, inside the real Rozvi UI shown
in the Figma file.

STYLING NOTE
------------
Streamlit does not support arbitrary pixel-level layout the way the
Figma file's React/Tailwind export does, so this is not a pixel-perfect
reproduction. What IS applied directly from the real Rozvi design system
(pulled from the Figma file's "Financial Impact : Climate VaR" screen,
not guessed) is:

    - Font: Plus Jakarta Sans, the same family used throughout Rozvi
    - Colours: #1a1a1a (near-black, used for primary buttons and the
      active-tab background), #f3f4f5 (page background), white cards
      with a rgba(26,26,26,0.4) border and 8px corner radius
    - The metric-card pattern: a muted label, a large bold value, and a
      muted caption underneath -- exactly how Rozvi's own VaR/CVaR cards
      are built
    - "Run Climate Analysis" as the primary action button, black
      background / white text, matching Rozvi's own button style

Run this app with:
    streamlit run app.py
"""

import streamlit as st

from calc_engine.risk_score import score_to_probability_and_depth
from calc_engine.damage_curves import (
    ASSET_CLASSES,
    CLASSES_WITHOUT_STRUCTURE_CONTENTS_SPLIT,
    get_damage_estimate,
)
from calc_engine.simulation import run_flood_loss_simulation


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(page_title="Flood CVaR Model", page_icon="\U0001F30A", layout="wide")

# ---- Rozvi design tokens, applied as custom CSS -------------------------
# These colours, the font, and the card/button shapes are taken directly
# from the Rozvi Figma file (Financial Impact : Climate VaR screen), not
# invented for this app.
ROZVI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Page background, matching Rozvi's #f3f4f5 */
.stApp {
    background-color: #f3f4f5;
}

/* Rozvi's brand mark: a small black square + wordmark, top of the sidebar
   in the real product. Reproduced here at the top of the page instead,
   since this demo has no sidebar navigation. */
.rozvi-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 24px;
}
.rozvi-brand .mark {
    width: 40px;
    height: 40px;
    background-color: #1a1a1a;
    border-radius: 4px;
}
.rozvi-brand .wordmark p {
    margin: 0;
    line-height: 1.2;
}
.rozvi-brand .wordmark .name {
    font-weight: 600;
    font-size: 18px;
    color: #1a1a1a;
}
.rozvi-brand .wordmark .tagline {
    font-weight: 600;
    font-size: 11px;
    color: rgba(26,26,26,0.6);
}

/* Metric cards: white, bordered, rounded -- matching Rozvi's VaR/CVaR
   cards exactly (label, big bold value, muted caption). Streamlit's
   built-in st.metric doesn't expose enough hooks to restyle cleanly, so
   the app builds these cards directly in HTML instead -- see
   render_metric_card() below. */
.rozvi-card {
    background-color: white;
    border: 1px solid rgba(26,26,26,0.4);
    border-radius: 8px;
    padding: 24px;
    height: 100%;
}
.rozvi-card .label {
    font-weight: 600;
    font-size: 16px;
    color: rgba(26,26,26,0.6);
    margin-bottom: 12px;
}
.rozvi-card .value {
    font-weight: 700;
    font-size: 28px;
    color: #1a1a1a;
    margin-bottom: 8px;
}
.rozvi-card .caption {
    font-weight: 500;
    font-size: 13px;
    color: rgba(26,26,26,0.6);
}

/* Primary button ("Run Climate Analysis"): black background, white text,
   matching Rozvi's own primary button and active-tab styling. */
div.stButton > button[kind="primary"] {
    background-color: #1a1a1a;
    color: white;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    font-weight: 600;
    padding: 12px 24px;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #333333;
    border-color: #333333;
    color: white;
}

/* Section headers, matching Rozvi's H3 (Bold, 20px) */
h2 {
    font-weight: 700 !important;
    font-size: 20px !important;
    color: #1a1a1a !important;
}
</style>
"""
st.markdown(ROZVI_CSS, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, caption: str) -> str:
    """
    Build one Rozvi-style metric card (label / big value / caption) as raw
    HTML. This mirrors the exact structure of the VaR/CVaR cards in the
    Rozvi Figma file, which st.metric cannot reproduce closely enough on
    its own.
    """
    return (
        f'<div class="rozvi-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="caption">{caption}</div>'
        f'</div>'
    )


st.markdown(
    """
    <div class="rozvi-brand">
        <div class="mark"></div>
        <div class="wordmark">
            <p class="name">Flood CVaR Model</p>
            <p class="tagline">Demo calculation engine, styled to match Rozvi Climate Intelligence</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "This screen demonstrates the CALCULATION only -- it is not the production UI. "
    "The dev team's job is to combine this calculation engine with the Rozvi "
    "risk-scoring model, inside the real Rozvi platform shown in the Figma file."
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

st.header("Asset details")

asset_name = st.text_input(
    "Asset Name",
    placeholder="e.g. Mvurwi Grain Silos",
    help="For identifying this asset in the results only -- does not affect the calculation.",
)

asset_class = st.selectbox("Asset Type", ASSET_CLASSES)

# Transport and Infrastructure (roads) have no Structure/Contents split in
# either the JRC or Hazus data, so we show a single Replacement Value
# field for those two, and the Structure/Contents pair for everything else.
asset_has_split = asset_class not in CLASSES_WITHOUT_STRUCTURE_CONTENTS_SPLIT

if asset_has_split:
    st.write(
        "**Replacement Value** -- the current cost to rebuild the structure "
        "and replace the contents from a totally destroyed state, at "
        "today's prices. This is *not* book value, market value, or "
        "insured value."
    )
    input_col1, input_col2 = st.columns(2)
    structure_value = input_col1.number_input("Structure Value ($)", min_value=0.0, step=1000.0, format="%.2f")
    contents_value = input_col2.number_input("Contents Value ($)", min_value=0.0, step=1000.0, format="%.2f")
    st.caption(f"Total Replacement Value: ${structure_value + contents_value:,.2f}")
else:
    st.write(
        "**Replacement Value** -- the current cost to fully rebuild this "
        "asset, at today's prices. "
        f"{asset_class} has no Structure/Contents split, so a single "
        "value is used for the whole asset."
    )
    structure_value = st.number_input("Replacement Value ($)", min_value=0.0, step=1000.0, format="%.2f")
    contents_value = 0.0  # not used for these asset classes -- see damage_curves.py

st.header("Risk score")

risk_score = st.number_input(
    "Risk Score (1.00 - 10.00)",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.01,
    format="%.2f",
    help=(
        "For this demo, enter a Risk Score directly. In production, this "
        "value will come from the Rozvi risk-scoring model automatically, "
        "based on the asset's location, elevation, slope, and rainfall."
    ),
)


# --------------------------------------------------------------------------
# Run the calculation
# --------------------------------------------------------------------------

# Matches the button label used on the equivalent screen in the Rozvi
# Figma design, so the two products feel consistent even before they are
# formally combined.
run_clicked = st.button("Run Climate Analysis", type="primary")

if run_clicked:
    if structure_value == 0 and contents_value == 0:
        st.warning("Please enter a Replacement Value greater than zero before running the analysis.")
    else:
        # Step 1: turn the Risk Score into an annual probability and a
        # flood depth.
        annual_probability, flood_depth_m = score_to_probability_and_depth(risk_score)

        # Step 2: look up how much damage a flood of that depth causes,
        # for this asset's class.
        damage_estimate = get_damage_estimate(asset_class, flood_depth_m)

        # Step 3: run the full simulation to get the loss distribution and
        # every risk figure read off it.
        result = run_flood_loss_simulation(
            structure_value=structure_value,
            contents_value=contents_value,
            annual_probability=annual_probability,
            damage_estimate=damage_estimate,
        )

        # ------------------------------------------------------------
        # Results
        # ------------------------------------------------------------
        st.header("Results")
        if asset_name:
            st.subheader(asset_name)

        with st.expander("How this was calculated (intermediate values)", expanded=False):
            st.write(f"- Annual probability of flooding, from the risk score: **{annual_probability:.2%}**")
            st.write(f"- Flood depth assigned to this risk score: **{flood_depth_m:.2f} m**")
            if damage_estimate.structure_mean is not None:
                st.write(
                    f"- Structure damage at this depth: mean **{damage_estimate.structure_mean:.1%}**, "
                    f"std dev **{damage_estimate.structure_std:.1%}**"
                )
                st.write(
                    f"- Contents damage at this depth: mean **{damage_estimate.contents_mean:.1%}**, "
                    f"std dev **{damage_estimate.contents_std:.1%}**"
                )
            else:
                st.write(
                    f"- Overall damage at this depth: mean **{damage_estimate.overall_mean:.1%}**, "
                    f"std dev **{damage_estimate.overall_std:.1%}**"
                )

        # Top row: Mean and Median, matching the "Mean Climate VaR" card
        # style from the Rozvi Figma file.
        top_col1, top_col2 = st.columns(2)
        with top_col1:
            st.markdown(
                render_metric_card("Mean Climate VaR", f"${result.mean_annual_loss:,.0f}", "Expected annual loss"),
                unsafe_allow_html=True,
            )
        with top_col2:
            st.markdown(
                render_metric_card("Median Loss", f"${result.median_loss:,.0f}", "50th percentile outcome"),
                unsafe_allow_html=True,
            )

        st.write("")  # small spacer
        st.subheader("Value at Risk (VaR)")

        # Five VaR cards in a row, matching the five-card layout on the
        # Rozvi Financial Impact screen exactly (VaR90 / VaR95 / VaR99 /
        # VaR99.5 / VaR99.8).
        var_labels_values_captions = [
            ("VaR 90", result.var_90, "10% exceedance"),
            ("VaR 95", result.var_95, "5% exceedance"),
            ("VaR 99", result.var_99, "1% exceedance"),
            ("VaR 99.5", result.var_99_5, "0.5% exceedance"),
            ("VaR 99.8", result.var_99_8, "0.2% exceedance"),
        ]
        var_cols = st.columns(5)
        for col, (label, value, caption) in zip(var_cols, var_labels_values_captions):
            with col:
                st.markdown(render_metric_card(label, f"${value:,.0f}", caption), unsafe_allow_html=True)

        st.write("")
        st.subheader("Conditional Value at Risk (CVaR)")
        cvar_col, _, _ = st.columns(3)
        with cvar_col:
            st.markdown(
                render_metric_card("CVaR 95", f"${result.cvar_95:,.0f}", "Average loss in the worst 5% of outcomes"),
                unsafe_allow_html=True,
            )

        st.write("")
        st.subheader("Simulated loss distribution")
        st.caption(
            "Histogram of all 10,000 simulated years. Most years show zero "
            "loss (no flood); the right-hand tail shows what a bad year "
            "could cost."
        )
        st.bar_chart(result.simulated_losses)

        st.info(
            "These are gross decision-support estimates based on modelled "
            "hazard and damage curves -- not insured-loss values or "
            "accounting forecasts.",
            icon="\u2139\ufe0f",
        )
