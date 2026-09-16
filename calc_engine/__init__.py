"""
calc_engine

The flood CVaR calculation logic, kept deliberately separate from the
Streamlit UI in app.py. This means the calculation itself can be tested,
reused, or imported into a different front end (or combined with the
Rozvi risk-scoring model's own codebase) without needing to touch or
depend on Streamlit at all.

    risk_score.py      Risk Score (1-10)  ->  (probability, depth)
    damage_curves.py   asset class + depth  ->  damage % and its spread
    simulation.py       everything above  ->  AAL, VaR ladder, CVaR95
"""
