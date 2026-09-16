"""
damage_curves.py

Looks up "how much damage does a flood of this depth cause" for a given
asset type, using the Full Curve dataset (data/full_curve.csv).

WHERE full_curve.csv COMES FROM
---------------------------------
This file is built from a mix of real and derived sources, per the flood
risk methodology:

    - Residential and Industrial buildings: real JRC (Joint Research
      Centre) African damage data.
    - Commercial buildings, Transport, and Infrastructure (roads): JRC has
      no African data for these, so their values are interpolated from
      JRC's own worldwide averages using a relationship borrowed from
      Industrial buildings (the only class with both real African and
      real worldwide data).
    - The Structure/Contents split (for Residential, Commercial, and
      Industrial only, Transport and roads have no such split in either
      JRC or Hazus) is derived by taking the shape of Hazus's (US) split
      and rescaling it to match JRC's African total.
    - Every "std" (standard deviation) column describes how much
      uncertainty surrounds that damage estimate, this is what lets the
      simulation model draw a realistic SPREAD of possible damage
      outcomes, rather than pretending every flood of a given depth does
      exactly the same amount of damage.

WHY INTERPOLATION IS NEEDED HERE TOO
--------------------------------------
full_curve.csv only has rows at nine fixed depths: 0, 0.5, 1, 1.5, 2, 3, 4,
5, and 6 metres. But the Risk Score can produce ANY depth in between (e.g.
1.075m, from risk_score.py). So exactly like the risk score itself, we
interpolate between the two nearest known depths to get a sensible answer
for a depth that isn't one of the nine exact rows.

HAZARD vs. VULNERABILITY (why material_factor lives here, not in
risk_score.py)
------------------------------------------------------------------
Annual probability and flood depth (from risk_score.py) describe the
HAZARD at a site -- will water arrive, and how deep will it get. That is
determined entirely by elevation, slope, rainfall, and distance to water.
It has nothing to do with what is built on the land: a mud-brick house
and a reinforced-concrete house on the same plot face the identical
flood -- same probability, same depth.

Building material is a VULNERABILITY question instead -- it affects how
much damage a given depth of water causes, which is exactly this file's
job. `material_factor` is applied here, and only here, as an adjustment
to which point on the damage curve gets read.

Two depths, kept deliberately separate:
    - The REAL depth (`depth_m`, as passed in) is what gets displayed to
      the user, what any "how this was calculated" panel reports, and
      what any future audit trail shows. It is never modified by this
      function.
    - The EFFECTIVE depth (`depth_m * material_factor`) is used only
      internally, purely to choose a point on the damage curve. It is
      not returned, logged, or displayed anywhere.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Path to the data file, relative to this file's own location, this
# means the code works no matter what folder you run the app from.
_DATA_PATH = Path(__file__).parent.parent / "data" / "full_curve.csv"

# The five asset classes the demo currently supports. These must match the
# "damage_class" values in full_curve.csv exactly.
ASSET_CLASSES = [
    "Residential buildings",
    "Commercial buildings",
    "Industrial buildings",
    "Transport",
    "Infrastructure - roads",
]

# Transport and roads have no meaningful Structure/Contents split, a
# road doesn't have "contents" the way a building does. For these two
# classes, only the single overall damage figure applies.
CLASSES_WITHOUT_STRUCTURE_CONTENTS_SPLIT = {"Transport", "Infrastructure - roads"}


@dataclass
class DamageEstimate:
    """
    The damage percentages (and their uncertainty) for one asset, at one
    specific flood depth.

    All percentages are expressed as decimal fractions (0.35 = 35%
    damage), matching the convention used everywhere else in this project.

    For Transport and Infrastructure (roads), there is no Structure/
    Contents split, so those four fields are left as None, the
    simulation code checks for this and treats the whole asset as a
    single value in that case.
    """
    overall_mean: float
    overall_std: float
    structure_mean: float | None
    structure_std: float | None
    contents_mean: float | None
    contents_std: float | None


def _load_full_curve() -> pd.DataFrame:
    """
    Read full_curve.csv into a pandas DataFrame.

    This is a small, cheap file to read, but we still only want to read it
    from disk once rather than on every single calculation. In the
    Streamlit app, this function is wrapped with @st.cache_data so it only
    actually runs once per app session, see app.py.
    """
    return pd.read_csv(_DATA_PATH)


def get_damage_estimate(
    asset_class: str,
    depth_m: float,
    material_factor: float = 1.0,
) -> DamageEstimate:
    """
    Look up the damage percentages for a given asset class, at a given
    flood depth, interpolating between the nearest known depths if needed.

    Parameters
    ----------
    asset_class:
        One of the values in ASSET_CLASSES above.
    depth_m:
        The REAL flood depth in metres (can be any value from 0 up to 6,
        the curves don't extend past 6m, since JRC/Hazus treat anything
        beyond that as effectively total loss). This is the physical
        hazard depth -- unaffected by material_factor, and unchanged by
        this function.
    material_factor:
        A vulnerability adjustment describing how much more (or less)
        damage this asset's construction sustains at a given real depth,
        relative to the baseline curve. Applied only as a multiplier on
        the depth used to read the curve (the "effective depth") -- it
        never changes depth_m itself, does not get returned, and plays
        no part in the hazard side of the calculation (probability or
        depth, both of which come from risk_score.py). Defaults to 1.0
        (no adjustment, i.e. baseline construction).

    Returns
    -------
    A DamageEstimate with the interpolated mean and standard deviation for
    the overall damage, and for Structure/Contents separately where that
    split applies.
    """
    if asset_class not in ASSET_CLASSES:
        raise ValueError(
            f"Unknown asset class {asset_class!r}. "
            f"Expected one of: {ASSET_CLASSES}"
        )

    curve_data = _load_full_curve()
    class_rows = curve_data[curve_data["damage_class"] == asset_class].sort_values("depth_m")

    known_depths = class_rows["depth_m"].to_numpy()

    # The effective depth is what actually gets looked up on the damage
    # curve -- the real depth adjusted by material_factor to reflect how
    # vulnerable this construction type is. depth_m itself is left
    # completely untouched; it is not reassigned, returned, or altered in
    # any way here.
    effective_depth = depth_m * material_factor

    # effective_depth is clamped to the 0-6m range the curves actually
    # cover. A flood deeper than 6m (whether from a large real depth or
    # from material_factor pushing the effective depth past it) is not
    # modelled separately, by that point the damage curves are already at
    # or near total loss (100%), so extrapolating further out would not
    # add anything meaningful, and would risk producing a nonsensical
    # damage figure above 100%.
    clamped_depth = float(np.clip(effective_depth, known_depths.min(), known_depths.max()))

    def interpolate_column(column_name: str) -> float:
        return float(np.interp(clamped_depth, known_depths, class_rows[column_name].to_numpy()))

    overall_mean = interpolate_column("africa_overall")
    overall_std = interpolate_column("overall_std")

    if asset_class in CLASSES_WITHOUT_STRUCTURE_CONTENTS_SPLIT:
        structure_mean = structure_std = contents_mean = contents_std = None
    else:
        structure_mean = interpolate_column("structure")
        structure_std = interpolate_column("structure_std")
        contents_mean = interpolate_column("contents")
        contents_std = interpolate_column("contents_std")

    return DamageEstimate(
        overall_mean=overall_mean,
        overall_std=overall_std,
        structure_mean=structure_mean,
        structure_std=structure_std,
        contents_mean=contents_mean,
        contents_std=contents_std,
    )
