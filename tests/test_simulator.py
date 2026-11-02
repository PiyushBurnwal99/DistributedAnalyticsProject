import pandas as pd

from rac.simulator import FAULT, MAINTENANCE, AssemblyLine, label_failures, run_offline


def test_one_record_per_robot_per_step(cfg):
    line = AssemblyLine(cfg, seed=1)
    recs = line.step()
    assert len(recs) == sum(s.robots for s in cfg.stations)
    assert {r["step"] for r in recs} == {0}


def test_material_is_conserved(cfg):
    line = AssemblyLine(cfg, seed=2)
    for _ in range(500):
        line.step()
    s1_done = sum(r.cycles_done for r in line.robots[0])
    in_process = sum(line.buffers) + sum(r.holding for g in line.robots[1:] for r in g)
    assert s1_done == line.finished_units + in_process


def test_buffers_respect_capacity(cfg):
    line = AssemblyLine(cfg, seed=3)
    for _ in range(800):
        line.step()
        for i in range(1, len(cfg.stations)):
            assert 0 <= line.buffers[i] <= cfg.stations[i - 1].buffer_capacity


def test_commands_are_applied_once_and_in_order(cfg):
    line = AssemblyLine(cfg, seed=4)
    rid = "S2_WELD_R1"
    assert line.apply_command({"robot_id": rid, "issued_for_step": 5, "action": "SET_SPEED", "speed_setpoint": 0.7})
    assert line.index[rid].speed == 0.7
    assert not line.apply_command({"robot_id": rid, "issued_for_step": 4, "action": "SET_SPEED", "speed_setpoint": 1.2})
    assert line.index[rid].speed == 0.7
    line.apply_command({"robot_id": rid, "issued_for_step": 6, "action": "SET_SPEED", "speed_setpoint": 9})
    assert line.index[rid].speed == cfg.control.speed_max  # clamped


def test_maintenance_resets_wear(cfg):
    line = AssemblyLine(cfg, seed=5)
    r = line.index["S1_PICK_R1"]
    r.wear = 0.8
    line.apply_command({"robot_id": r.robot_id, "issued_for_step": 1, "action": "SCHEDULE_MAINTENANCE",
                        "speed_setpoint": 1.0})
    assert r.state == MAINTENANCE
    for _ in range(cfg.maintenance_steps):
        line.step()
    assert r.state != MAINTENANCE and r.wear <= 0.05 + 0.01


def test_stale_maintenance_order_is_ignored(cfg):
    line = AssemblyLine(cfg, seed=6)
    r = line.index["S1_PICK_R1"]
    for _ in range(10):
        line.step()
    line.apply_command({"robot_id": r.robot_id, "issued_for_step": 8, "action": "SCHEDULE_MAINTENANCE"})
    assert r.state == MAINTENANCE
    for _ in range(cfg.maintenance_steps + 2):
        line.step()
    # an order raised (step 9) before the service started (step 10) arrives late: no second service
    line.apply_command({"robot_id": r.robot_id, "issued_for_step": 9, "action": "SCHEDULE_MAINTENANCE"})
    assert r.state != MAINTENANCE and r.maintenances == 1


def test_labels_mark_steps_before_a_fault():
    rows = [{"run_id": 0, "robot_id": "R", "step": t, "state": FAULT if 10 <= t < 13 else "RUNNING"}
            for t in range(20)]
    df = label_failures(pd.DataFrame(rows), horizon_steps=3)
    lab = dict(zip(df["step"], df["label_fault_soon"]))
    assert [lab[t] for t in range(5, 13)] == [0, 0, 1, 1, 1, 0, 0, 0]


def test_predictive_control_reduces_faults(cfg):
    _, fixed = run_offline(cfg, 1500, "fixed", seed=11, record=False)
    _, full = run_offline(cfg, 1500, "full", seed=11, record=False)
    assert full["faults"] < fixed["faults"]
