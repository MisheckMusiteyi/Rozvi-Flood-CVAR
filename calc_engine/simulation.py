"""
simulation.py

This is where everything else comes together into an actual loss number.

THE BIG PICTURE
------------------
For one asset, we now have:

    - an annual PROBABILITY that a flood happens at all      (risk_score.py)
    - a flood DEPTH to expect, if one does happen             (risk_score.py)
    - a MEAN damage percentage at that depth, plus how much   (damage_curves.py)
      uncertainty surrounds it (its standard deviation)
    - the asset's REPLACEMENT VALUE (Structure and Contents,  (user input)
      or a single value for Transport/roads)

A single, one-shot calculation would just multiply these together and stop
-- but that only ever gives you ONE number: the *average* expected loss.
It can't tell you anything about how bad a genuinely bad year could be,
which is exactly what a Value at Risk (VaR) or Climate Value at Risk
(CVaR) figure is for -- Rozvi's own name for what risk literature more
generally calls Conditional VaR or Expected Shortfall: the average loss
in the worst slice of outcomes, not just the threshold itself.

To get those, we simulate many thousands of possible years. In each
simulated year:

    1. We flip a weighted coin: does a flood happen this year at all,
       given the annual probability we were handed?
    2. If yes, we don't assume the damage is always exactly the mean
       percentage -- real floods vary. So we draw a damage percentage
       from a distribution centred on that mean, with the right amount
       of spread (its standard deviation).
    3. We turn that damage percentage into an actual dollar loss for that
       one simulated year.

Doing this 10,000 times gives us 10,000 simulated possible outcomes for
the year. Most of them will be $0 (no flood). The non-zero ones describe
the shape of "what a bad year could look like" -- and that's exactly what
lets us read off proper VaR and CVaR figures at the end, instead of just
one average number.

WHY A BETA DISTRIBUTION, SPECIFICALLY
----------------------------------------
A damage percentage can never be below 0% or above 100%. A plain "bell
curve" (normal distribution) doesn't respect that -- it could technically
suggest a damage percentage of -5% or 110%, which makes no sense. The Beta
distribution is built specifically for values that must stay between 0
and 1, which is exactly what a damage fraction is. It's also the standard
choice in credit risk for modelling Loss Given Default, for the same
reason.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_distribution

from calc_engine.damage_curves import DamageEstimate

# How many possible years to simulate. 10,000 is the standard choice used
# throughout this project (and in the drought VaR methodology it borrows
# from) -- enough draws that the tail percentiles (95th, 99th, 99.8th) are
# reasonably stable, without being so many that the app feels slow.
DEFAULT_NUMBER_OF_SIMULATED_YEARS = 10_000


@dataclass
class LossSimulationResult:
    """
    The full set of risk figures produced by one simulation run, matching
    the metrics the Rozvi platform's own results screens already show
    (see the "Financial Impact : Climate VaR" screens in the Rozvi Figma
    file) -- so this output can be dropped straight into that UI without
    the dev team needing to ask for additional figures later.
    """
    mean_annual_loss: float   # the "Expected/Average Annual Loss" headline number
    median_loss: float        # the 50th percentile outcome -- often $0, see note below
    var_90: float
    var_95: float
    var_99: float
    var_99_5: float
    var_99_8: float
    cvar_95: float
    simulated_losses: np.ndarray  # the raw 10,000 simulated outcomes, kept for charting


def _fit_beta_distribution(mean: float, std_dev: float) -> tuple[float, float]:
    """
    Work out the two shape parameters (alpha, beta) of a Beta distribution
    that has the given mean and standard deviation.

    This uses the "method of moments" -- a standard, simple way to fit a
    distribution: rather than guessing and checking, there is a direct
    algebraic formula that gives you the alpha/beta which reproduce
    exactly the mean and standard deviation you started with.

    Edge cases: if the mean is exactly 0 or exactly 1, or the standard
    deviation is 0, there is no meaningful "spread" to model -- the
    outcome is certain. In that case we return parameters that make the
    distribution behave as a fixed point at that mean, rather than letting
    the formula divide by zero.

    A Beta distribution with a given mean can only support variances up
    to mean * (1 - mean) (its variance approaches that ceiling as the
    distribution becomes U-shaped/degenerate). Some rows in the damage
    curve data report a standard deviation that -- combined with a mean
    very close to 0 or 1 -- exceeds that ceiling (e.g. a mean of 0.995
    with a std of 0.156). That combination isn't a valid Beta shape at
    all: the method-of-moments formula below would produce a negative
    alpha/beta, which scipy rejects outright. We treat that case the same
    way as the "certain outcome" edge cases above, since a mean that
    close to 0 or 1 leaves essentially no room for real spread anyway.

    A further wrinkle: scipy requires BOTH alpha and beta to be strictly
    positive, not just non-negative. A mean of exactly 0 or exactly 1
    (which genuinely occurs in this data -- e.g. 0% damage at 0m depth,
    or 100% damage at the deepest floods for some asset classes) would
    otherwise make one of the two shape parameters exactly 0.0, which
    scipy rejects with the same error. The damage curve data also has one
    row (Transport at 5m depth) where the mean is a fraction of a percent
    over 1.0 -- a data quirk, not a real >100% damage figure -- which
    would make beta go negative. We guard against all of these the same
    way: clip the mean into a narrow-but-open (0, 1) interval before
    computing alpha/beta in the "certain outcome" branch, so the
    resulting distribution is still (for all practical purposes) a fixed
    point at the intended mean, just never at the literal boundary.
    """
    if std_dev <= 0 or mean <= 0 or mean >= 1 or std_dev ** 2 >= mean * (1 - mean):
        # A very large alpha+beta with the right ratio squeezes the Beta
        # distribution down to (almost) a single point at `mean`, without
        # the numerical problems of a literal zero-variance distribution.
        epsilon = 1e-6
        clipped_mean = min(max(mean, epsilon), 1 - epsilon)
        concentration = 1_000_000
        alpha = clipped_mean * concentration
        beta_param = (1 - clipped_mean) * concentration
        return alpha, beta_param

    variance = std_dev ** 2
    common_term = mean * (1 - mean) / variance - 1
    alpha = mean * common_term
    beta_param = (1 - mean) * common_term
    return alpha, beta_param


def _draw_damage_percentage(mean: float, std_dev: float, random_generator: np.random.Generator, n: int) -> np.ndarray:
    """
    Draw `n` random damage percentages from a Beta distribution fitted to
    the given mean and standard deviation.
    """
    alpha, beta_param = _fit_beta_distribution(mean, std_dev)
    return beta_distribution.rvs(alpha, beta_param, size=n, random_state=random_generator)


def run_flood_loss_simulation(
    structure_value: float,
    contents_value: float,
    annual_probability: float,
    damage_estimate: DamageEstimate,
    number_of_simulated_years: int = DEFAULT_NUMBER_OF_SIMULATED_YEARS,
    random_seed: int | None = None,
) -> LossSimulationResult:
    """
    Run the full Monte Carlo simulation for one asset and return its VaR/
    CVaR risk profile.

    Parameters
    ----------
    structure_value, contents_value:
        The asset's Replacement Value, split into Structure and Contents.
        For Transport and Infrastructure (roads), which have no
        Structure/Contents split, pass the asset's single Replacement
        Value as `structure_value` and 0 as `contents_value` -- the
        damage_estimate for those classes only has an "overall" figure
        anyway, so the split doesn't matter for the maths, but keeping
        the whole value in one bucket avoids silently losing half of it.
    annual_probability:
        The annual chance of a flood at all, from risk_score.py.
    damage_estimate:
        The DamageEstimate for this asset's class and depth, from
        damage_curves.py.
    number_of_simulated_years:
        How many possible years to simulate. Defaults to 10,000.
    random_seed:
        Optional. Pass a fixed number here to make a run reproducible
        (useful for testing); leave as None for genuinely random results.

    Returns
    -------
    A LossSimulationResult with the mean, median, VaR ladder, and CVaR95.
    """
    random_generator = np.random.default_rng(random_seed)

    # Step 1: decide, for every simulated year at once, whether a flood
    # happens that year. This is a single vectorised "coin flip" rather
    # than a loop -- draw one uniform random number per simulated year,
    # and a flood "happens" in that year if the draw falls below the
    # annual probability.
    uniform_draws = random_generator.random(number_of_simulated_years)
    flood_happens = uniform_draws < annual_probability

    # Step 2: for every simulated year, draw a damage percentage. We draw
    # for ALL years (not just the ones where a flood happens) because it's
    # simpler and faster to do this as one vectorised operation -- the
    # damage percentage for a no-flood year is simply discarded below.
    has_structure_contents_split = damage_estimate.structure_mean is not None

    if has_structure_contents_split:
        structure_damage_pct = _draw_damage_percentage(
            damage_estimate.structure_mean, damage_estimate.structure_std,
            random_generator, number_of_simulated_years,
        )
        contents_damage_pct = _draw_damage_percentage(
            damage_estimate.contents_mean, damage_estimate.contents_std,
            random_generator, number_of_simulated_years,
        )
        loss_if_flood = (structure_value * structure_damage_pct) + (contents_value * contents_damage_pct)
    else:
        # Transport / Infrastructure (roads): one damage curve, applied to
        # the single replacement value passed in as structure_value.
        overall_damage_pct = _draw_damage_percentage(
            damage_estimate.overall_mean, damage_estimate.overall_std,
            random_generator, number_of_simulated_years,
        )
        loss_if_flood = structure_value * overall_damage_pct

    # Step 3: a year only "counts" its loss if a flood actually happened
    # in that year. Everywhere else, the loss is zero.
    simulated_losses = np.where(flood_happens, loss_if_flood, 0.0)

    # Step 4: read every risk figure off the finished pile of simulated
    # outcomes. np.percentile does the ranking and threshold-finding for
    # us; CVaR is then simply the average of everything at or beyond its
    # matching VaR threshold.
    def value_at_risk(confidence_level_pct: float) -> float:
        return float(np.percentile(simulated_losses, confidence_level_pct))

    def climate_value_at_risk(confidence_level_pct: float) -> float:
        threshold = value_at_risk(confidence_level_pct)
        tail_losses = simulated_losses[simulated_losses >= threshold]
        # Guard against an empty tail (possible if the confidence level is
        # extremely high and every simulated loss below it happens to be
        # smaller than the threshold value itself).
        return float(tail_losses.mean()) if len(tail_losses) > 0 else threshold

    return LossSimulationResult(
        mean_annual_loss=float(simulated_losses.mean()),
        median_loss=value_at_risk(50),
        var_90=value_at_risk(90),
        var_95=value_at_risk(95),
        var_99=value_at_risk(99),
        var_99_5=value_at_risk(99.5),
        var_99_8=value_at_risk(99.8),
        cvar_95=climate_value_at_risk(95),
        simulated_losses=simulated_losses,
    )
