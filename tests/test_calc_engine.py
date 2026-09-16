"""
test_calc_engine.py

A small set of sanity checks for the calculation engine. These are not
exhaustive, they exist to catch the kind of mistake that's easy to make
by accident (a sign flipped, a column mixed up, an edge case forgotten)
and to give the dev team a working example of how each module is meant
to be called.

Run with:
    pytest tests/test_calc_engine.py -v
"""

import numpy as np
import pytest

from calc_engine.risk_score import score_to_probability_and_depth
from calc_engine.damage_curves import get_damage_estimate, ASSET_CLASSES
from calc_engine.simulation import run_flood_loss_simulation


# --------------------------------------------------------------------------
# risk_score.py
# --------------------------------------------------------------------------

def test_score_at_exact_band_midpoints_matches_the_original_assignment_table():
    """
    A risk score that lands exactly on one of the five original band
    midpoints should reproduce that band's assigned values exactly,
    this is the most basic check that the interpolation hasn't distorted
    the source assumptions.
    """
    probability, depth = score_to_probability_and_depth(1.5)
    assert probability == pytest.approx(0.002)
    assert depth == pytest.approx(0.0)

    probability, depth = score_to_probability_and_depth(9.5)
    assert probability == pytest.approx(0.20)
    assert depth == pytest.approx(2.4)


def test_score_below_first_anchor_holds_flat_rather_than_extrapolating():
    """A score of 1.00 is below the first anchor (1.5), it should be
    held at the same value as the anchor, not extrapolated past it."""
    probability_at_1_00, depth_at_1_00 = score_to_probability_and_depth(1.00)
    probability_at_1_50, depth_at_1_50 = score_to_probability_and_depth(1.50)
    assert probability_at_1_00 == pytest.approx(probability_at_1_50)
    assert depth_at_1_00 == pytest.approx(depth_at_1_50)


def test_score_outside_valid_range_is_rejected():
    with pytest.raises(ValueError):
        score_to_probability_and_depth(0.5)
    with pytest.raises(ValueError):
        score_to_probability_and_depth(10.5)


# --------------------------------------------------------------------------
# damage_curves.py
# --------------------------------------------------------------------------

def test_zero_depth_means_zero_damage_for_every_asset_class():
    """At 0m depth (no flooding), every asset class should show 0% damage,
    this is the anchor point every damage curve is built from."""
    for asset_class in ASSET_CLASSES:
        estimate = get_damage_estimate(asset_class, depth_m=0.0)
        assert estimate.overall_mean == pytest.approx(0.0, abs=1e-6)


def test_transport_and_infrastructure_have_no_structure_contents_split():
    for asset_class in ["Transport", "Infrastructure - roads"]:
        estimate = get_damage_estimate(asset_class, depth_m=1.0)
        assert estimate.structure_mean is None
        assert estimate.contents_mean is None


def test_residential_buildings_have_a_structure_contents_split():
    estimate = get_damage_estimate("Residential buildings", depth_m=1.0)
    assert estimate.structure_mean is not None
    assert estimate.contents_mean is not None
    assert 0 <= estimate.structure_mean <= 1
    assert 0 <= estimate.contents_mean <= 1


def test_depth_between_known_grid_points_is_interpolated_not_rejected():
    """full_curve.csv only has rows at 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6m,
    a depth like 1.075m (which can genuinely occur from the risk score)
    should still return a sensible, in-between answer."""
    estimate_at_1_0 = get_damage_estimate("Residential buildings", depth_m=1.0)
    estimate_at_1_5 = get_damage_estimate("Residential buildings", depth_m=1.5)
    estimate_between = get_damage_estimate("Residential buildings", depth_m=1.25)

    assert estimate_at_1_0.overall_mean < estimate_between.overall_mean < estimate_at_1_5.overall_mean


# --------------------------------------------------------------------------
# simulation.py
# --------------------------------------------------------------------------

def test_zero_probability_produces_zero_loss_every_time():
    """If the annual probability of flooding is 0, not a single simulated
    year should show a loss, no matter how large the asset value is."""
    damage_estimate = get_damage_estimate("Residential buildings", depth_m=2.0)
    result = run_flood_loss_simulation(
        structure_value=1_000_000,
        contents_value=500_000,
        annual_probability=0.0,
        damage_estimate=damage_estimate,
        random_seed=42,
    )
    assert result.mean_annual_loss == 0.0
    assert result.var_99_8 == 0.0


def test_higher_replacement_value_scales_up_the_loss_proportionally():
    """Doubling the replacement value should double every loss figure,
    the calculation should be linear in asset value."""
    damage_estimate = get_damage_estimate("Commercial buildings", depth_m=1.5)

    result_small = run_flood_loss_simulation(
        structure_value=100_000, contents_value=50_000,
        annual_probability=0.10, damage_estimate=damage_estimate, random_seed=1,
    )
    result_large = run_flood_loss_simulation(
        structure_value=200_000, contents_value=100_000,
        annual_probability=0.10, damage_estimate=damage_estimate, random_seed=1,
    )

    assert result_large.mean_annual_loss == pytest.approx(result_small.mean_annual_loss * 2, rel=0.01)


def test_cvar_95_is_never_smaller_than_var_95():
    """By definition, CVaR at a given confidence level is the average of
    everything AT OR BEYOND the matching VaR threshold, so it can never
    be smaller than that threshold."""
    damage_estimate = get_damage_estimate("Industrial buildings", depth_m=1.6)
    result = run_flood_loss_simulation(
        structure_value=800_000, contents_value=200_000,
        annual_probability=0.05, damage_estimate=damage_estimate, random_seed=7,
    )
    assert result.cvar_95 >= result.var_95


def test_var_ladder_is_non_decreasing():
    """A higher confidence level should never produce a SMALLER loss
    threshold, VaR90 <= VaR95 <= VaR99 <= VaR99.5 <= VaR99.8 always."""
    damage_estimate = get_damage_estimate("Residential buildings", depth_m=0.9)
    result = run_flood_loss_simulation(
        structure_value=350_000, contents_value=150_000,
        annual_probability=0.10, damage_estimate=damage_estimate, random_seed=3,
    )
    assert result.var_90 <= result.var_95 <= result.var_99 <= result.var_99_5 <= result.var_99_8
