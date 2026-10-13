"""Conservative descriptive regimes require current and agreeing control duty fractions."""

import numpy as np
import pandas as pd


def classify_regime(frame: pd.DataFrame) -> pd.Series:
    required = ["Motor_current", "COMP", "DV_eletric"]
    if not set(required).issubset(frame):
        raise ValueError("Regime classification needs current, COMP and DV_eletric")
    current, comp, duty = (frame[c] for c in required)
    result = pd.Series("ambiguous", index=frame.index, dtype="string", name="regime")
    complete = (
        frame[required].notna().all(axis=1)
        & comp.between(0, 1)
        & duty.between(0, 1)
        & (current >= 0)
        & np.isfinite(current)
    )
    consistent_unloaded = (comp >= 0.8) & (duty <= 0.2)
    consistent_loaded = (comp <= 0.2) & (duty >= 0.8)
    result.loc[(current < 1) & consistent_unloaded & complete] = "off"
    result.loc[current.between(1, 5.5, inclusive="left") & consistent_unloaded & complete] = (
        "unloaded"
    )
    result.loc[(current >= 5.5) & consistent_loaded & complete] = "loaded"
    mixed = comp.between(0.2, 0.8, inclusive="neither") | duty.between(
        0.2, 0.8, inclusive="neither"
    )
    result.loc[mixed & complete] = "transition"
    result.loc[~complete] = "unknown"
    return result


def regime_audit(frame: pd.DataFrame) -> dict:
    regimes = classify_regime(frame)
    current_only = np.select(
        [frame.Motor_current < 1, frame.Motor_current < 5.5], ["off", "unloaded"], "loaded"
    )
    stable = regimes.isin(["off", "unloaded", "loaded"])
    controls = np.select(
        [
            (frame.COMP <= 0.2) & (frame.DV_eletric >= 0.8),
            (frame.COMP >= 0.8) & (frame.DV_eletric <= 0.2),
        ],
        ["loaded_controls", "unloaded_controls"],
        "mixed_or_disagreeing",
    )
    cross_tab = pd.crosstab(
        pd.Series(current_only, index=frame.index), pd.Series(controls, index=frame.index)
    ).to_dict()
    eligible = controls != "mixed_or_disagreeing"
    agreement = (np.array(current_only) == "loaded") == (controls == "loaded_controls")
    return {
        "current_control_cross_tab": cross_tab,
        "current_control_load_agreement_fraction": float(agreement[eligible].mean())
        if eligible.any()
        else None,
        "counts": regimes.value_counts().to_dict(),
        "stable_bin_fraction": float(stable.mean()),
        "current_rule_agreement_on_stable_bins": float(
            (regimes[stable] == current_only[stable]).mean()
        )
        if stable.any()
        else None,
        "control_complement_mean_absolute_disagreement": float(
            ((1 - frame.COMP) - frame.DV_eletric).abs().mean()
        ),
        "limitation": "Agreement is a consistency check by construction, not independent physical regime ground truth. Digital fractions describe mixed five-minute bins.",
    }
