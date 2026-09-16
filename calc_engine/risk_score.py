"""
risk_score.py

Turns a single Risk Score (1.00 to 10.00, from the Rozvi risk-scoring model)
into the two numbers our flood loss calculation actually needs:

    - an annual PROBABILITY of a flood occurring, and
    - a flood DEPTH (in metres) to look up on the damage curves.

WHERE THE FIVE ANCHOR POINTS BELOW COME FROM
----------------------------------------------
The Risk Factor Model methodology (Method one: "direct assignment") assigns
a probability and a depth to each score BAND, based on reasoned judgement
about what each band is meant to represent:

    Score band   Assigned probability   Assigned depth
    1-2          0.2%                   No meaningful flooding (0m)
    3-4          1%                     0.3m
    5-6          5%                     0.9m
    7-8          10%                    1.6m
    9-10         20%                    2.4m

That table only gives you five buckets. Our Risk Score input is continuous
(e.g. 3.45, 6.72), so we need a value for EVERY possible score, not just
five of them. The fix: treat each band's MIDPOINT (1.5, 3.5, 5.5, 7.5, 9.5)
as an anchor point, and draw a straight line between consecutive anchors.
A score of exactly 3.5 gets exactly the "3-4" band's assigned values; a
score of 3.45 gets a value very close to it; a score of 6.0 (halfway
between the 5.5 and 7.5 anchors) gets a value halfway between their
assigned probabilities and depths.

Below the first anchor (1.5) or above the last anchor (9.5), there is no
"next" band to interpolate towards, so the value is simply held flat at
whatever the nearest end anchor says. This matches exactly what was built
and verified in the "Risk Score Continuous Assignment" reference workbook.
"""

import numpy as np

# The five score midpoints the assigned values apply to.
ANCHOR_SCORES = np.array([1.5, 3.5, 5.5, 7.5, 9.5])

# The annual probability of a flood occurring, assigned to each anchor score.
# Expressed as a decimal fraction (0.05 = 5%), not a percentage.
ANCHOR_PROBABILITIES = np.array([0.002, 0.01, 0.05, 0.10, 0.20])

# The flood depth (in metres) assigned to each anchor score.
# The "1-2" band's "no meaningful flooding" is represented as 0.0m, the
# same convention used throughout the JRC/Hazus damage curve data (a depth
# of zero is the baseline "no flood" case).
ANCHOR_DEPTHS_M = np.array([0.0, 0.3, 0.9, 1.6, 2.4])


def score_to_probability_and_depth(risk_score: float) -> tuple[float, float]:
    """
    Convert a single Risk Score into (annual_probability, flood_depth_m).

    Parameters
    ----------
    risk_score:
        A number between 1.00 and 10.00 (inclusive), as produced by the
        Rozvi risk-scoring model. Decimals are expected and supported
        (e.g. 3.45, 6.72) -- this is exactly why the anchor-and-interpolate
        approach is used instead of a five-row lookup table.

    Returns
    -------
    A tuple of (annual_probability, flood_depth_m):
        annual_probability : float, e.g. 0.05 means "5% chance in a year"
        flood_depth_m       : float, the flood depth in metres to look up
                               on the damage curves

    Examples
    --------
    >>> score_to_probability_and_depth(1.5)
    (0.002, 0.0)
    >>> score_to_probability_and_depth(9.5)
    (0.2, 2.4)
    >>> score_to_probability_and_depth(6.0)   # halfway between two anchors
    (0.0625, 1.075)
    """
    if not (1.0 <= risk_score <= 10.0):
        raise ValueError(
            f"Risk score must be between 1.00 and 10.00 inclusive; "
            f"got {risk_score!r}."
        )

    # numpy.interp does exactly the piecewise-linear interpolation we want:
    # for a risk_score that falls between two anchor points, it draws a
    # straight line between them and reads off the in-between value. For a
    # risk_score below the smallest anchor or above the largest one, it
    # automatically "holds flat" at the nearest end value -- which is
    # exactly the behaviour we want, so no extra edge-case code is needed.
    probability = float(np.interp(risk_score, ANCHOR_SCORES, ANCHOR_PROBABILITIES))
    depth_m = float(np.interp(risk_score, ANCHOR_SCORES, ANCHOR_DEPTHS_M))

    return probability, depth_m
