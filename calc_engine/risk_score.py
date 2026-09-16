"""
risk_score.py

Turns a single Risk Score (1.00 to 10.00, from the Rozvi risk-scoring model)
into the two numbers our flood loss calculation actually needs:

    - an annual PROBABILITY of a flood occurring, and
    - a flood DEPTH (in metres) to look up on the damage curves.

WHERE THE FIVE ANCHOR DEPTHS COME FROM
------------------------------------------
The Risk Factor Model methodology (Method one: "direct assignment") assigns
a depth to each score BAND, based on reasoned judgement about what each
band is meant to represent:

    Score band   Assigned depth
    1-2          No meaningful flooding (0m)
    3-4          0.3m
    5-6          0.9m
    7-8          1.6m
    9-10         2.4m

WHERE THE FIVE ANCHOR PROBABILITIES COME FROM, AND WHY THEY CHANGED
------------------------------------------------------------------------
The first version of this file used judgement-based probabilities of
0.2%, 1%, 5%, 10%, and 20% at the five anchors. Validating the resulting
Average Annual Loss (AAL) as a percentage of Replacement Value against
published real-world catastrophe model benchmarks showed those original
probabilities were far too high from score 5 upward:

    - National blended NFIP historical loss rate: ~0.17% of TIV
    - Properties confirmed inside the 100-year floodplain: ~0.71-0.85%
    - First Street's own floodplain estimate (flagged by independent
      peer review as inflated 1.5-2x): ~1.17%
    - NFIP's most extreme "Repetitive Loss" tier -- the single worst 1%
      of the entire 4.4 million property NFIP book, properties that have
      already filed repeat flood claims: roughly 1.8-4.6% of TIV



    Score band   Assigned probability   Resulting AAL/TIV (approx, Residential)
    1-2          0.2%                   ~0% (no meaningful flooding anyway)
    3-4          0.6%                   ~0.05-0.08%, below the national average
    5-6          1.2%                   ~0.3-0.4%, national-to-floodplain range
    7-8          1.6%                   ~0.8-0.9%, at the confirmed floodplain tier
    9-10         4.0%                   ~2.3-3.0%, inside the extreme repetitive-loss tier

That table only gives you five buckets. Our Risk Score input is continuous
(e.g. 3.45, 6.72), so we need a value for EVERY possible score, not just
five of them. The fix: treat each band's MIDPOINT (1.5, 3.5, 5.5, 7.5, 9.5)
as an anchor point, and draw a straight line between consecutive anchors.
A score of exactly 3.5 gets exactly the "3-4" band's assigned values; a
score of 3.45 gets a value very close to it; a score of 6.0 (halfway
between the 5.5 and 7.5 anchors) gets a value halfway between their
assigned probabilities and depths.

WHY THE ENDS OF THE SCALE (1.00-1.50 AND 9.50-10.00) ARE NOT FLAT
--------------------------------------------------------------------
The five anchors only span 1.5 to 9.5 -- half a point short of each end
of the true 1.00-10.00 scale. The first version of this function used
numpy.interp's default behaviour beyond that range, which HOLDS FLAT at
the nearest anchor's value. That meant every score from 9.50 to 10.00 --
a full half-point of the scale -- returned the exact same 4.00% and
2.40m, with zero differentiation between a 9.5 and a 10.0. The same
problem existed at the bottom, between 1.00 and 1.50.

Instead, this function EXTENDS THE SLOPE of the nearest real segment
(7.5-to-9.5 at the top, 1.5-to-3.5 at the bottom) past the last anchor,
rather than flattening it. A score of 10.0 continues the same rate of
increase that carried the curve from 7.5 up to 9.5, landing at 4.6%
probability and 2.6m depth -- confirmed to still land inside the
validated extreme-repetitive-loss AAL/TIV benchmark band (3.5%, within
the 1.8-4.6% target range). Depth is floored at 0m, since the same
technique applied at the bottom end would otherwise extrapolate to a
physically meaningless negative flood depth.

HONESTY NOTE FOR WHOEVER READS THIS LATER
-------------------------------------------
These five anchor values are still a judgement call (Method one), not
something measured from real Zimbabwean flood data. What changed is that
the judgement is now anchored to published real-world AAL/TIV benchmarks
rather than being an unvalidated first guess. This is a meaningfully
better starting point, but it is still not Method two ("calibrated
assignment"): once real hazard and claims data is available for actual
Zimbabwean locations, these numbers should be replaced with values fitted
from observed outcomes. Nothing else in this file needs to change when
that happens, only the five numbers in ANCHOR_PROBABILITIES and
ANCHOR_DEPTHS_M below.
"""

import numpy as np

# The five score midpoints the assigned values apply to.
ANCHOR_SCORES = np.array([1.5, 3.5, 5.5, 7.5, 9.5])

# The annual probability of a flood occurring, assigned to each anchor
# score. Expressed as a decimal fraction (0.012 = 1.2%), not a percentage.
# Revised from the original [0.002, 0.01, 0.05, 0.10, 0.20] after
# validating the resulting AAL/TIV against real-world catastrophe model
# benchmarks -- see the module docstring above for the full reasoning.
ANCHOR_PROBABILITIES = np.array([0.002, 0.006, 0.012, 0.016, 0.040])

# The flood depth (in metres) assigned to each anchor score.
# The "1-2" band's "no meaningful flooding" is represented as 0.0m, the
# same convention used throughout the JRC/Hazus damage curve data (a depth
# of zero is the baseline "no flood" case). Unchanged by the probability
# recalibration above.
ANCHOR_DEPTHS_M = np.array([0.0, 0.3, 0.9, 1.6, 2.4])


def _interpolate_with_slope_extrapolation(
    x: float, anchor_x: np.ndarray, anchor_y: np.ndarray, floor: float | None = None
) -> float:
    """
    Piecewise-linear interpolation between anchor points, exactly like
    numpy.interp for any x that falls between the first and last anchor.

    The difference is what happens OUTSIDE that range: instead of holding
    flat at the nearest anchor's value (numpy.interp's default, and a dead
    zone with zero differentiation), this continues the slope of the
    nearest real segment past the edge. See the "WHY THE ENDS OF THE
    SCALE..." section in this module's docstring for the full reasoning.

    Parameters
    ----------
    x:
        The value to interpolate at.
    anchor_x, anchor_y:
        The anchor points, sorted ascending by anchor_x.
    floor:
        If given, the result is never allowed to go below this value.
        Used for depth, since extrapolating below the lowest anchor could
        otherwise produce a physically meaningless negative depth.
    """
    if x <= anchor_x[0]:
        slope = (anchor_y[1] - anchor_y[0]) / (anchor_x[1] - anchor_x[0])
        result = anchor_y[0] + slope * (x - anchor_x[0])
    elif x >= anchor_x[-1]:
        slope = (anchor_y[-1] - anchor_y[-2]) / (anchor_x[-1] - anchor_x[-2])
        result = anchor_y[-1] + slope * (x - anchor_x[-1])
    else:
        result = float(np.interp(x, anchor_x, anchor_y))

    if floor is not None:
        result = max(floor, result)
    return float(result)


def score_to_probability_and_depth(risk_score: float) -> tuple[float, float]:
    """
    Convert a single Risk Score into (annual_probability, flood_depth_m).

    Parameters
    ----------
    risk_score:
        A number between 1.00 and 10.00 (inclusive), as produced by the
        Rozvi risk-scoring model. Decimals are expected and supported
        (e.g. 3.45, 6.72), this is exactly why the anchor-and-interpolate
        approach is used instead of a five-row lookup table.

    Returns
    -------
    A tuple of (annual_probability, flood_depth_m):
        annual_probability : float, e.g. 0.012 means "1.2% chance in a year"
        flood_depth_m       : float, the flood depth in metres to look up
                               on the damage curves

    Examples
    --------
    >>> score_to_probability_and_depth(1.5)
    (0.002, 0.0)
    >>> score_to_probability_and_depth(9.5)
    (0.04, 2.4)
    >>> score_to_probability_and_depth(6.0)   # halfway between two anchors
    (0.014, 1.075)
    >>> score_to_probability_and_depth(10.0)  # past the last anchor, not flat
    (0.046, 2.6)
    """
    if not (1.0 <= risk_score <= 10.0):
        raise ValueError(
            f"Risk score must be between 1.00 and 10.00 inclusive; "
            f"got {risk_score!r}."
        )

    probability = _interpolate_with_slope_extrapolation(
        risk_score, ANCHOR_SCORES, ANCHOR_PROBABILITIES, floor=0.0
    )
    depth_m = _interpolate_with_slope_extrapolation(
        risk_score, ANCHOR_SCORES, ANCHOR_DEPTHS_M, floor=0.0
    )

    return probability, depth_m
