"""
app.py

The Streamlit demo for the Flood CVaR model.

This app is deliberately narrow in scope: it only asks for the handful of
inputs the calculation actually needs (see calc_engine/ for what each one
feeds into), and it only shows the results. It exists to demonstrate the
CALCULATION -- not to reproduce the full Rozvi platform experience. The
dev team's job, once this is handed over, is to combine this calculation
engine with the Rozvi risk-scoring model, inside the real Rozvi UI shown
in the Figma file. Until then, the Risk Score below is entered manually,
standing in for whatever number the Rozvi model will eventually supply.

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

st.set_page_config(page_title="Flood CVaR Model", page_icon="\U0001F30A", layout="centered")

st.title("Flood CVaR Model")
st.caption(
    "A demo calculation engine for flood-related financial loss, built to "
    "plug into the Rozvi risk-scoring platform. This screen shows the "
    "calculation only -- the production UI will follow the Rozvi Figma design."
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
    structure_value = st.number_input("Structure Value ($)", min_value=0.0, step=1000.0, format="%.2f")
    contents_value = st.number_input("Contents Value ($)", min_value=0.0, step=1000.0, format="%.2f")
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
        #
        # material_factor is a vulnerability adjustment (construction
        # quality/strength), not a hazard input -- it only shifts which
        # point on the damage curve gets read inside get_damage_estimate.
        # flood_depth_m itself (the real, physical hazard depth from
        # risk_score.py) is passed through unchanged, and is exactly what
        # gets shown below in "How this was calculated". Hardcoded to 1.3
        # for this demo; in production this would vary by asset/material.
        material_factor = 1.3
        damage_estimate = get_damage_estimate(asset_class, flood_depth_m, material_factor=material_factor)

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

        col1, col2 = st.columns(2)
        col1.metric("Expected Annual Loss (mean)", f"${result.mean_annual_loss:,.0f}")
        col2.metric("Median Loss (P50)", f"${result.median_loss:,.0f}")

        st.subheader("Value at Risk (VaR)")
        var_cols = st.columns(5)
        var_cols[0].metric("VaR 90", f"${result.var_90:,.0f}")
        var_cols[1].metric("VaR 95", f"${result.var_95:,.0f}")
        var_cols[2].metric("VaR 99", f"${result.var_99:,.0f}")
        var_cols[3].metric("VaR 99.5", f"${result.var_99_5:,.0f}")
        var_cols[4].metric("VaR 99.8", f"${result.var_99_8:,.0f}")

        st.subheader("Conditional Value at Risk (CVaR)")
        st.metric("CVaR 95", f"${result.cvar_95:,.0f}")

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
