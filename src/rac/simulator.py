"""Discrete-time simulator of a robotic assembly line.

The line is a serial flow: S1 -> buffer -> S2 -> buffer -> ... -> finished goods.
Each station has several identical robots working in parallel. Every robot
emits one telemetry record per simulation step (joint temperature, vibration,
torque, position error, state, completed cycles).

Robots wear out as they work (faster when driven harder), which raises
temperature, vibration and position error and eventually causes a fault.
Control commands (speed set-points, maintenance, e-stop, reset) are applied
between steps, which closes the loop with the Spark controller.

Two ways to run it:

* ``online``  writes one JSON-lines file per step into a directory that Spark
  Structured Streaming watches, and reads the commands Spark writes back.
* ``offline`` runs fast in-process (optionally with the same control law in
  the loop) to generate labelled history for model training and benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import LineConfig, load_config

RUNNING, STARVED, BLOCKED, IDLE = "RUNNING", "STARVED", "BLOCKED", "IDLE"
FAULT, MAINTENANCE, ESTOP = "FAULT", "MAINTENANCE", "ESTOP"
AMBIENT_C = 30.0


@dataclass
class Robot:
    robot_id: str
    station_id: str
    station_idx: int
    nominal_cycle_s: float
    wear_rate: float
    speed: float = 1.0
    state: str = IDLE
    holding: bool = False
    progress: float = 0.0
    cycle_elapsed: float = 0.0
    cycle_factor: float = 1.0
    wear: float = 0.0
    temp: float = AMBIENT_C + 2.0
    vibration: float = 1.2
    torque: float = 20.0
    position_error: float = 0.04
    hold_steps: int = 0
    cycles_done: int = 0
    last_cycle_time: float = 0.0
    last_cmd_step: int = -1
    faults: int = 0
    maintenances: int = 0
    estops: int = 0
    last_service_step: int = -1


class AssemblyLine:
    """State of the whole line plus the physics of one simulation step."""

    def __init__(self, cfg: LineConfig, seed: int = 7, start_time: datetime | None = None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.step_no = 0
        self.start_time = start_time or datetime(2026, 11, 3, 8, 0, 0)
        self.finished_units = 0
        n = len(cfg.stations)
        # buffers[i] = parts waiting in front of station i (i >= 1). Station 0 has unlimited raw parts.
        self.buffers = [0] * n
        self.robots: list[list[Robot]] = []
        for idx, st in enumerate(cfg.stations):
            group = []
            for k in range(st.robots):
                r = Robot(
                    robot_id=f"{st.id}_R{k + 1}",
                    station_id=st.id,
                    station_idx=idx,
                    nominal_cycle_s=st.nominal_cycle_s,
                    wear_rate=self.rng.uniform(0.6, 1.6),
                    wear=self.rng.uniform(0.0, 0.35),
                )
                group.append(r)
            self.robots.append(group)
        self.index = {r.robot_id: r for g in self.robots for r in g}

    # ------------------------------------------------------------------ commands
    def apply_command(self, cmd: dict) -> bool:
        """Apply one control command. Stale or duplicate commands are ignored."""
        r = self.index.get(cmd.get("robot_id"))
        if r is None:
            return False
        issued = int(cmd.get("issued_for_step", -1))
        if issued <= r.last_cmd_step:
            return False
        r.last_cmd_step = issued
        c = self.cfg.control
        speed = cmd.get("speed_setpoint")
        if speed is not None and not math.isnan(float(speed)):
            r.speed = min(max(float(speed), c.speed_min), c.speed_max)
        action = cmd.get("action")
        if (action == "SCHEDULE_MAINTENANCE" and r.state not in (MAINTENANCE, FAULT)
                and issued > r.last_service_step):  # ignore work orders raised before the last service
            r.state = MAINTENANCE
            r.last_service_step = self.step_no
            r.hold_steps = self.cfg.maintenance_steps
            r.maintenances += 1
        elif action == "ESTOP" and r.state not in (MAINTENANCE, FAULT, ESTOP):
            r.state = ESTOP
            r.estops += 1
        elif action == "RESET" and r.state == ESTOP:
            r.state = IDLE
        return True

    # ------------------------------------------------------------------ physics
    def _buffer_capacity_after(self, idx: int) -> int:
        return self.cfg.stations[idx].buffer_capacity

    def _update_signals(self, r: Robot, active: bool) -> None:
        dt = self.cfg.sim_dt_seconds
        g = self.rng.gauss
        a = 1.0 if active else 0.0
        target = AMBIENT_C + a * (22.0 * r.speed ** 2 + 18.0 * r.wear + 12.0 * max(0.0, r.wear - 0.7) / 0.3)
        r.temp += (target - r.temp) * (1.0 - math.exp(-dt / 120.0)) + g(0, 0.4)
        r.vibration = max(0.2, 1.2 + a * r.speed * (1.5 + 9.0 * r.wear ** 2) + g(0, 0.3))
        r.torque = max(0.0, 20.0 + a * (30.0 * r.speed + 25.0 * r.wear) + g(0, 2.0))
        r.position_error = max(0.0, 0.04 + 0.5 * r.wear ** 2.5 + g(0, 0.015))

    def step(self) -> list[dict]:
        """Advance the line by one time step and return one telemetry record per robot."""
        cfg = self.cfg
        dt = cfg.sim_dt_seconds
        n = len(cfg.stations)
        completed_flags: dict[str, int] = {}

        # Downstream stations first so that freed buffer space is visible upstream.
        for idx in reversed(range(n)):
            for r in self.robots[idx]:
                completed_flags[r.robot_id] = 0
                active = False
                if r.state in (MAINTENANCE, FAULT):
                    r.hold_steps -= 1
                    if r.hold_steps <= 0:
                        r.wear = 0.05 if r.state == MAINTENANCE else 0.10
                        r.state = IDLE
                elif r.state == ESTOP:
                    pass
                else:
                    if not r.holding:
                        if idx == 0 or self.buffers[idx] > 0:
                            if idx > 0:
                                self.buffers[idx] -= 1
                            r.holding = True
                            r.progress = 0.0
                            r.cycle_elapsed = 0.0
                            r.cycle_factor = self.rng.lognormvariate(0.0, 0.05 + 0.25 * r.wear)
                        else:
                            r.state = STARVED
                    if r.holding:
                        if r.progress < 1.0:
                            r.progress += dt * r.speed / (r.nominal_cycle_s * r.cycle_factor)
                            r.cycle_elapsed += dt
                            r.state = RUNNING
                            active = True
                            r.wear = min(1.0, r.wear + r.wear_rate * 0.0025 * (dt / 10.0)
                                         * r.speed ** 3 * self.rng.uniform(0.5, 1.5))
                        if r.progress >= 1.0:
                            last = idx == n - 1
                            if last or self.buffers[idx + 1] < self._buffer_capacity_after(idx):
                                if last:
                                    self.finished_units += 1
                                else:
                                    self.buffers[idx + 1] += 1
                                r.holding = False
                                r.cycles_done += 1
                                r.last_cycle_time = r.cycle_elapsed
                                completed_flags[r.robot_id] = 1
                                r.state = RUNNING
                            else:
                                r.state = BLOCKED
                    if active:
                        hazard = 0.0002 + 0.35 * max(0.0, r.wear - 0.55) ** 2
                        if r.temp > 90.0:
                            hazard += 0.03
                        if self.rng.random() < hazard:
                            r.state = FAULT
                            r.hold_steps = cfg.fault_repair_steps
                            r.faults += 1
                self._update_signals(r, active)

        ts = (self.start_time + timedelta(seconds=self.step_no * dt)).isoformat()
        records = []
        for idx, group in enumerate(self.robots):
            for r in group:
                records.append({
                    "ts": ts,
                    "step": self.step_no,
                    "line_id": cfg.line_id,
                    "station_id": r.station_id,
                    "robot_id": r.robot_id,
                    "state": r.state,
                    "speed_factor": round(r.speed, 4),
                    "cycle_completed": completed_flags[r.robot_id],
                    "cycles_done": r.cycles_done,
                    "last_cycle_time_s": round(r.last_cycle_time, 2),
                    "joint_temp_c": round(r.temp, 3),
                    "vibration_mm_s": round(r.vibration, 3),
                    "torque_nm": round(r.torque, 3),
                    "position_error_mm": round(r.position_error, 4),
                    "buffer_in": self.buffers[idx] if idx > 0 else -1,
                    "buffer_out": self.buffers[idx + 1] if idx < n - 1 else -1,
                    "buffer_out_capacity": cfg.stations[idx].buffer_capacity if idx < n - 1 else -1,
                    "line_finished_units": self.finished_units,
                    "true_wear": round(r.wear, 4),  # ground truth; the controller never reads it
                })
        self.step_no += 1
        return records

    def summary(self) -> dict:
        robots = list(self.index.values())
        hours = self.step_no * self.cfg.sim_dt_seconds / 3600.0
        return {
            "steps": self.step_no,
            "sim_hours": round(hours, 3),
            "finished_units": self.finished_units,
            "throughput_uph": round(self.finished_units / hours, 2) if hours else 0.0,
            "faults": sum(r.faults for r in robots),
            "maintenances": sum(r.maintenances for r in robots),
            "estops": sum(r.estops for r in robots),
        }


# ---------------------------------------------------------------------- online mode
def _read_new_commands(commands_dir: Path, seen: dict[str, int]) -> list[dict]:
    """Read command rows Spark appended since the last call (Spark writes part-*.json files)."""
    rows: list[dict] = []
    if not commands_dir.exists():
        return rows
    for p in sorted(commands_dir.glob("part-*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        start = seen.get(p.name, 0)
        for line in lines[start:]:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        seen[p.name] = len(lines)
    rows.sort(key=lambda c: c.get("issued_for_step", -1))
    return rows


def run_online(cfg: LineConfig, telemetry_dir: Path, commands_dir: Path, steps: int,
               real_interval: float, seed: int, steps_per_file: int = 1) -> dict:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    line = AssemblyLine(cfg, seed=seed)
    seen: dict[str, int] = {}
    applied = 0
    newest_cmd_step = -1
    lags: list[int] = []
    buf: list[dict] = []
    for _ in range(steps):
        for cmd in _read_new_commands(commands_dir, seen):
            applied += line.apply_command(cmd)
            newest_cmd_step = max(newest_cmd_step, int(cmd.get("issued_for_step", -1)))
        if newest_cmd_step >= 0:
            lags.append(line.step_no - newest_cmd_step)  # steps between observation and actuation
        buf.extend(line.step())
        if line.step_no % steps_per_file == 0:
            name = f"telemetry_{line.step_no:07d}.json"
            tmp = telemetry_dir / f".{name}.tmp"  # hidden files are ignored by Spark
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in buf:
                    rec = {k: v for k, v in rec.items() if k != "true_wear"}
                    f.write(json.dumps(rec) + "\n")
            os.replace(tmp, telemetry_dir / name)
            buf = []
        if line.step_no % 30 == 0:
            s = line.summary()
            print(f"[sim] step={s['steps']} units={s['finished_units']} uph={s['throughput_uph']} "
                  f"faults={s['faults']} maint={s['maintenances']} estops={s['estops']} "
                  f"commands_applied={applied} cmd_lag_steps={lags[-1] if lags else '-'}", flush=True)
        if real_interval > 0:
            time.sleep(real_interval)
    s = line.summary()
    s["commands_applied"] = applied
    if lags:
        s["median_cmd_lag_steps"] = sorted(lags)[len(lags) // 2]
        s["median_cmd_lag_s_wall"] = round(s["median_cmd_lag_steps"] * real_interval, 2)
    return s


# --------------------------------------------------------------------- offline mode
def run_offline(cfg: LineConfig, steps: int, policy: str, seed: int, control_every: int = 3,
                model_path: str | None = None, record: bool = True, model=None):
    """Run the line in-process.

    policy:
      * ``fixed``      - every robot at the fixed speed; hard-wired safety interlock only.
      * ``explore``    - random speed changes, run-to-failure (used to generate training data).
      * ``balancing``  - the control law with throughput balancing, no predictive maintenance.
      * ``full``       - balancing + predictive maintenance (what the Spark controller runs).
    Returns (records, summary).
    """
    import pandas as pd

    from .control_law import ControlModes, HealthModel, control_station, init_state

    line = AssemblyLine(cfg, seed=seed)
    rng = random.Random(seed + 1000)
    records: list[dict] = []
    window: list[dict] = []
    modes = {
        "fixed": ControlModes(balancing=False, predictive=False),
        "explore": None,
        "balancing": ControlModes(balancing=True, predictive=False),
        "full": ControlModes(balancing=True, predictive=True),
    }[policy]
    model = model if model is not None else HealthModel.load(model_path)
    states = {s.id: init_state() for s in cfg.stations}

    for _ in range(steps):
        if policy == "explore":
            if line.step_no % 50 == 0:
                for r in line.index.values():
                    r.speed = rng.uniform(0.7, cfg.control.speed_max)
            for r in line.index.values():  # simple safety interlock with auto reset
                if r.state == ESTOP and r.temp < cfg.safety.max_joint_temp_c - cfg.safety.reset_temp_margin_c:
                    r.state = IDLE
                elif r.state not in (ESTOP, MAINTENANCE, FAULT) and (
                        r.temp > cfg.safety.max_joint_temp_c
                        or r.vibration > cfg.safety.max_vibration_mm_s
                        or r.position_error > cfg.safety.max_position_error_mm):
                    r.state = ESTOP
                    r.estops += 1
        recs = line.step()
        if record:
            records.extend(recs)
        window.extend(recs)
        if modes is not None and line.step_no % control_every == 0:
            df = pd.DataFrame(window)
            for sid, g in df.groupby("station_id"):
                cmds, states[sid] = control_station(g, states[sid], cfg, model, modes)
                for c in cmds.to_dict("records"):
                    line.apply_command(c)
            window = []
        elif modes is None:
            window = []
    return records, line.summary()


def label_failures(df, horizon_steps: int):
    """label = 1 if the robot faults within the next ``horizon_steps`` steps."""
    import numpy as np

    df = df.sort_values(["run_id", "robot_id", "step"]).reset_index(drop=True)
    df["fault_start"] = (df["state"] == FAULT) & (df.groupby(["run_id", "robot_id"])["state"].shift(1) != FAULT)
    label = np.zeros(len(df), dtype=np.int8)
    for _, g in df.groupby(["run_id", "robot_id"]):
        steps = g["step"].to_numpy()
        fault_steps = g.loc[g["fault_start"], "step"].to_numpy()
        if len(fault_steps) == 0:
            continue
        nxt = np.searchsorted(fault_steps, steps, side="right")
        has = nxt < len(fault_steps)
        dist = np.where(has, fault_steps[np.minimum(nxt, len(fault_steps) - 1)] - steps, 10 ** 9)
        label[g.index.to_numpy()] = ((dist > 0) & (dist <= horizon_steps)).astype(np.int8)
    df["label_fault_soon"] = label
    return df.drop(columns=["fault_start"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Robotic assembly line simulator")
    sub = ap.add_subparsers(dest="mode", required=True)

    on = sub.add_parser("online", help="stream telemetry to files for Spark and apply Spark commands")
    on.add_argument("--config", default=None)
    on.add_argument("--telemetry-dir", default="data/stream/telemetry")
    on.add_argument("--commands-dir", default="data/stream/commands")
    on.add_argument("--steps", type=int, default=720)
    on.add_argument("--real-interval", type=float, default=0.5, help="wall-clock seconds per step")
    on.add_argument("--steps-per-file", type=int, default=1)
    on.add_argument("--seed", type=int, default=7)

    off = sub.add_parser("offline", help="generate labelled history (fast, in-process)")
    off.add_argument("--config", default=None)
    off.add_argument("--steps", type=int, default=6000)
    off.add_argument("--runs", type=int, default=4)
    off.add_argument("--policy", default="explore", choices=["fixed", "explore", "balancing", "full"])
    off.add_argument("--horizon-steps", type=int, default=30, help="label window: fault within N steps")
    off.add_argument("--seed", type=int, default=100)
    off.add_argument("--out", default="data/history/telemetry_labeled.parquet")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.mode == "online":
        s = run_online(cfg, Path(args.telemetry_dir), Path(args.commands_dir), args.steps,
                       args.real_interval, args.seed, args.steps_per_file)
        print("[sim] finished:", json.dumps(s))
    else:
        import pandas as pd

        frames = []
        for i in range(args.runs):
            recs, s = run_offline(cfg, args.steps, args.policy, seed=args.seed + i)
            df = pd.DataFrame(recs)
            df["run_id"] = i
            frames.append(df)
            print(f"[sim] run {i}: {s}")
        df = label_failures(pd.concat(frames, ignore_index=True), args.horizon_steps)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[sim] wrote {len(df):,} rows to {out} (positive labels: {df['label_fault_soon'].mean():.2%})")


if __name__ == "__main__":
    main()
