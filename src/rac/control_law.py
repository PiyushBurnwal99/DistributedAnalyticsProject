"""Per-station control law.

This is plain pandas/numpy so that it can be unit-tested, run in-process by the
offline simulator, and run inside Spark (``applyInPandasWithState``), where each
station is a key and Spark spreads the stations over its executors.

Each call handles one micro-batch of telemetry for one station:

1. Health:   score every robot's fault risk with a logistic model trained in Spark MLlib.
2. Safety:   hard interlocks (temperature, vibration, position error) -> ESTOP / RESET.
3. Predictive maintenance: high risk -> SCHEDULE_MAINTENANCE (at most one robot at a time
   per station unless the risk is urgent); medium risk -> derate the robot's speed.
4. Throughput balancing: feed-forward speed from the takt time plus a PI correction on the
   measured station output rate, with anti-windup when the station is blocked by a slow
   downstream station or starved by a slow upstream one.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, LineConfig

FEATURES = ["joint_temp_c", "vibration_mm_s", "torque_nm", "position_error_mm", "speed_factor"]
INACTIVE = {"MAINTENANCE", "FAULT", "ESTOP"}
STATE_FIELDS = ["rate_ewma", "integral", "last_speed", "last_step", "derated"]
COMMAND_COLUMNS = [
    "line_id", "station_id", "robot_id", "issued_for_step", "issued_at", "action",
    "speed_setpoint", "risk_score", "reason", "robot_state", "station_rate_uph",
    "target_uph", "frac_blocked", "frac_starved",
]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "health_model_default.json"
TRAINED_MODEL = PROJECT_ROOT / "models" / "health_model.json"


@dataclass(frozen=True)
class ControlModes:
    balancing: bool = True
    predictive: bool = True


@dataclass
class HealthModel:
    """Standardised logistic regression: p = sigmoid(b + sum_i w_i * (x_i - mu_i) / sd_i)."""

    features: list
    means: list
    stds: list
    coefficients: list
    intercept: float
    source: str = "default"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "HealthModel":
        if path is None:
            path = TRAINED_MODEL if TRAINED_MODEL.exists() else DEFAULT_MODEL
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            features=d["features"], means=d["means"], stds=d["stds"],
            coefficients=d["coefficients"], intercept=float(d["intercept"]),
            source=d.get("source", str(path)),
        )

    def to_dict(self) -> dict:
        return {
            "features": list(self.features), "means": list(self.means), "stds": list(self.stds),
            "coefficients": list(self.coefficients), "intercept": self.intercept, "source": self.source,
        }

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy(dtype=float)
        sd = np.where(np.asarray(self.stds) > 0, self.stds, 1.0)
        z = self.intercept + ((x - np.asarray(self.means)) / sd) @ np.asarray(self.coefficients)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def init_state() -> dict:
    return {"rate_ewma": -1.0, "integral": 0.0, "last_speed": 1.0, "last_step": -1, "derated": ""}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def control_station(events: pd.DataFrame, state: dict, cfg: LineConfig, model: HealthModel,
                    modes: ControlModes = ControlModes()) -> tuple[pd.DataFrame, dict]:
    """Compute commands for every robot of one station from one micro-batch of telemetry."""
    if events.empty:
        return pd.DataFrame(columns=COMMAND_COLUMNS), dict(state)

    c, s = cfg.control, cfg.safety
    ev = events.sort_values(["step", "robot_id"])
    station_id = str(ev["station_id"].iloc[0])
    st_cfg = cfg.station(station_id)
    dt = cfg.sim_dt_seconds
    target = cfg.target_rate_per_s

    # Ignore telemetry older than what we already processed (e.g. late files).
    ev = ev[ev["step"] > int(state.get("last_step", -1))]
    if ev.empty:
        return pd.DataFrame(columns=COMMAND_COLUMNS), dict(state)

    n_steps = ev["step"].nunique()
    duration = n_steps * dt
    rate = float(ev["cycle_completed"].sum()) / duration
    prev = float(state.get("rate_ewma", -1.0))
    alpha = 1.0 - math.exp(-duration / c.rate_tau_s)
    rate_ewma = rate if prev < 0 else prev + alpha * (rate - prev)
    frac_blocked = float((ev["state"] == "BLOCKED").mean())
    frac_starved = float((ev["state"] == "STARVED").mean())

    latest = ev.groupby("robot_id", sort=True).tail(1).set_index("robot_id").sort_index()
    # Score risk on the robot's mean signals over the batch (less noise than a single sample).
    mean_signals = ev.groupby("robot_id", sort=True)[model.features].mean().reindex(latest.index)
    risk = model.predict_proba(mean_signals)
    was_derated = {r for r in str(state.get("derated", "") or "").split(",") if r}
    derated: set[str] = set()
    last_step = int(ev["step"].max())
    issued_at = str(latest["ts"].max())

    # ---- 1-3: safety and health decisions per robot
    actions: dict[str, tuple[str, str]] = {}
    in_maintenance = int((latest["state"] == "MAINTENANCE").sum())
    for i, (rid, row) in enumerate(latest.iterrows()):
        r = float(risk[i])
        unsafe = (row["joint_temp_c"] > s.max_joint_temp_c
                  or row["vibration_mm_s"] > s.max_vibration_mm_s
                  or row["position_error_mm"] > s.max_position_error_mm)
        cooled = row["joint_temp_c"] < s.max_joint_temp_c - s.reset_temp_margin_c
        if row["state"] in ("MAINTENANCE", "FAULT"):
            actions[rid] = ("HOLD", f"robot in {row['state'].lower()}")
        elif row["state"] == "ESTOP":
            if modes.predictive and r >= c.maintenance_risk:
                actions[rid] = ("SCHEDULE_MAINTENANCE", f"e-stopped and risk {r:.2f}")
                in_maintenance += 1
            elif cooled:
                actions[rid] = ("RESET", "interlock cleared")
            else:
                actions[rid] = ("HOLD", "e-stop: waiting to cool down")
        elif unsafe:
            actions[rid] = ("ESTOP", "safety limit exceeded")
        elif modes.predictive and r >= c.maintenance_risk and (in_maintenance == 0 or r >= c.urgent_risk):
            actions[rid] = ("SCHEDULE_MAINTENANCE", f"predicted fault risk {r:.2f}")
            in_maintenance += 1
        elif modes.predictive and (r >= c.derate_risk
                                   or (rid in was_derated and r >= c.derate_risk - c.derate_hysteresis)):
            actions[rid] = ("SET_SPEED", f"derated: risk {r:.2f}")
            derated.add(str(rid))
        else:
            actions[rid] = ("SET_SPEED", "nominal")

    # ---- 4: throughput balancing for the robots that will keep producing
    producing = sum(1 for a, _ in actions.values() if a in ("SET_SPEED", "RESET"))
    last_speed = float(state.get("last_speed", c.fixed_speed))
    integral = float(state.get("integral", 0.0))
    if modes.balancing:
        err = (target - rate_ewma) / target
        ff = target * st_cfg.nominal_cycle_s / max(1, producing) * c.ff_margin
        blocked = frac_blocked > c.blocked_threshold
        starved = frac_starved > c.starved_threshold
        if not (blocked or starved):
            integral = _clamp(integral + err * duration / 60.0, -c.integral_limit, c.integral_limit)
        speed = ff * (1.0 + c.kp * err + c.ki * integral)
        if blocked:
            speed = min(speed, last_speed - c.max_step)  # downstream is the bottleneck: back off
        if starved:
            speed = min(speed, last_speed)  # upstream is the bottleneck: speeding up is pointless
        speed = _clamp(speed, last_speed - c.max_step, last_speed + c.max_step)
        speed = _clamp(speed, c.speed_min, c.speed_max)
    else:
        speed = c.fixed_speed

    rows = []
    for i, (rid, row) in enumerate(latest.iterrows()):
        action, reason = actions[rid]
        sp = speed
        if reason.startswith("derated"):
            sp = min(speed, c.derate_speed_cap)
        rows.append({
            "line_id": str(row["line_id"]),
            "station_id": station_id,
            "robot_id": str(rid),
            "issued_for_step": last_step,
            "issued_at": issued_at,
            "action": action,
            "speed_setpoint": round(float(sp), 4),
            "risk_score": round(float(risk[i]), 4),
            "reason": reason,
            "robot_state": str(row["state"]),
            "station_rate_uph": round(rate_ewma * 3600.0, 2),
            "target_uph": round(target * 3600.0, 2),
            "frac_blocked": round(frac_blocked, 3),
            "frac_starved": round(frac_starved, 3),
        })
    new_state = {"rate_ewma": rate_ewma, "integral": integral, "last_speed": speed, "last_step": last_step,
                 "derated": ",".join(sorted(derated))}
    return pd.DataFrame(rows, columns=COMMAND_COLUMNS), new_state
