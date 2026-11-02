"""Line configuration: stations, demand, control gains and safety limits."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "line_config.json"


@dataclass(frozen=True)
class StationConfig:
    id: str
    name: str
    robots: int
    nominal_cycle_s: float
    buffer_capacity: int


@dataclass(frozen=True)
class ControlConfig:
    kp: float = 0.5
    ki: float = 0.05
    integral_limit: float = 3.0
    rate_tau_s: float = 300.0
    ff_margin: float = 1.05
    speed_min: float = 0.6
    speed_max: float = 1.25
    max_step: float = 0.08
    fixed_speed: float = 1.0
    blocked_threshold: float = 0.3
    starved_threshold: float = 0.3
    derate_risk: float = 0.5
    derate_speed_cap: float = 0.85
    derate_hysteresis: float = 0.1
    maintenance_risk: float = 0.8
    urgent_risk: float = 0.95


@dataclass(frozen=True)
class SafetyConfig:
    max_joint_temp_c: float = 88.0
    max_position_error_mm: float = 0.6
    max_vibration_mm_s: float = 11.0
    reset_temp_margin_c: float = 15.0


@dataclass(frozen=True)
class LineConfig:
    line_id: str
    demand_units_per_hour: float
    sim_dt_seconds: float
    fault_repair_steps: int
    maintenance_steps: int
    stations: tuple[StationConfig, ...]
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @property
    def target_rate_per_s(self) -> float:
        return self.demand_units_per_hour / 3600.0

    def station(self, station_id: str) -> StationConfig:
        for s in self.stations:
            if s.id == station_id:
                return s
        raise KeyError(f"Unknown station {station_id!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LineConfig":
        return cls(
            line_id=d["line_id"],
            demand_units_per_hour=float(d["demand_units_per_hour"]),
            sim_dt_seconds=float(d["sim_dt_seconds"]),
            fault_repair_steps=int(d["fault_repair_steps"]),
            maintenance_steps=int(d["maintenance_steps"]),
            stations=tuple(StationConfig(**s) for s in d["stations"]),
            control=ControlConfig(**d.get("control", {})),
            safety=SafetyConfig(**d.get("safety", {})),
        )


def load_config(path: str | Path | None = None) -> LineConfig:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return LineConfig.from_dict(json.load(f))
