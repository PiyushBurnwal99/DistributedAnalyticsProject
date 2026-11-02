"""Charts and a summary of a streaming run (reads what the Spark controller wrote)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_config  # noqa: E402

COLORS = ["#2f6f9f", "#c2611b", "#3a8f5c", "#8a5fb0", "#b8494a", "#5b6670", "#d4a017", "#1f9e9e"]


def _read_commands(path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(path.glob("part-*.json")):
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return pd.DataFrame(rows)


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7ea")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Report for a streaming control run")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default="data/output")
    ap.add_argument("--commands-dir", default="data/stream/commands")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)

    hist = pd.read_parquet(Path(args.output_dir) / "history")
    hist["station_id"] = hist["station_id"].astype(str)
    cmds = _read_commands(Path(args.commands_dir))
    kpis_path = Path(args.output_dir) / "kpis"
    kpis = pd.read_parquet(kpis_path) if kpis_path.exists() and any(kpis_path.glob("*.parquet")) else pd.DataFrame()

    # 1. line throughput (finished units from the last station) per simulated minute
    last_station = cfg.stations[-1].id
    fig, ax = plt.subplots(figsize=(10, 4))
    done = hist[hist["station_id"] == last_station].copy()
    done["minute"] = done["step"] * cfg.sim_dt_seconds // 600 * 10  # 10-minute buckets
    g = done.groupby("minute").agg(units=("cycle_completed", "sum"), steps=("step", "nunique"))
    g = g[g["steps"] >= 0.5 * 600 / cfg.sim_dt_seconds]  # drop partial buckets
    per = g["units"] * 3600.0 / (g["steps"] * cfg.sim_dt_seconds)  # units per hour
    ax.plot(per.index, per.values, color=COLORS[0], lw=2, label="Line output")
    ax.axhline(cfg.demand_units_per_hour, color=COLORS[1], ls="--", lw=1.5, label="Demand (takt)")
    ax.set_xlabel("Simulated time (minutes, 10-minute buckets)")
    _style(ax, "Line throughput vs demand", "units / hour")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)

    # 2. speed set-points and 3. fault risk per robot
    if not cmds.empty:
        cmds = cmds.sort_values("issued_for_step")
        for col, fname, title, ylab in [
            ("speed_setpoint", "speed_setpoints.png", "Speed set-points issued by Spark", "speed factor"),
            ("risk_score", "fault_risk.png", "Predicted fault risk (MLlib model) and maintenance orders", "risk"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for i, (rid, g) in enumerate(cmds.groupby("robot_id")):
                ax.plot(g["issued_for_step"], g[col], color=COLORS[i % len(COLORS)], lw=1.4, label=rid)
                if col == "risk_score":
                    m = g[g["action"] == "SCHEDULE_MAINTENANCE"]
                    ax.scatter(m["issued_for_step"], m[col], color=COLORS[i % len(COLORS)], marker="v", s=40)
            if col == "risk_score":
                ax.axhline(cfg.control.maintenance_risk, color="#5b6670", ls="--", lw=1)
            ax.set_xlabel("Simulation step")
            _style(ax, title, ylab)
            ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
            fig.tight_layout()
            fig.savefig(out / fname, dpi=150)
            plt.close(fig)

    # 4. state share per station
    share = pd.crosstab(hist["station_id"], hist["state"], normalize="index")
    order = [c for c in ["RUNNING", "BLOCKED", "STARVED", "IDLE", "MAINTENANCE", "FAULT", "ESTOP"] if c in share]
    fig, ax = plt.subplots(figsize=(10, 4))
    share[order].plot(kind="barh", stacked=True, ax=ax, color=COLORS[: len(order)], width=0.6)
    ax.set_xlabel("share of robot-time")
    _style(ax, "Where robot time went, by station", "")
    ax.legend(frameon=False, ncol=len(order), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out / "state_share.png", dpi=150)
    plt.close(fig)

    steps = int(hist["step"].max()) + 1
    hours = steps * cfg.sim_dt_seconds / 3600
    finished = int(hist["line_finished_units"].max())
    starts = hist.sort_values("step").groupby("robot_id")["state"].apply(
        lambda s: int(((s == "FAULT") & (s.shift() != "FAULT")).sum())).sum()
    lines = [
        "# Streaming run summary", "",
        f"- Simulated time: {hours:.2f} h ({steps} steps)",
        f"- Finished units: {finished} ({finished / hours:.1f} units/h vs demand {cfg.demand_units_per_hour:.0f})",
        f"- Robot faults: {starts}",
        f"- Commands issued by Spark: {len(cmds)}",
    ]
    if not cmds.empty:
        for a, n in cmds["action"].value_counts().items():
            lines.append(f"  - {a}: {n}")
    if not kpis.empty:
        lines.append(f"- KPI windows written: {len(kpis)}")
    (out / "run_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[report] charts written to {out}/")


if __name__ == "__main__":
    main()
