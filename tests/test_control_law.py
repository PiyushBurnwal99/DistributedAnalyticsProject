import pandas as pd
import pytest

from rac.control_law import COMMAND_COLUMNS, ControlModes, HealthModel, control_station, init_state


def make_events(station="S2_WELD", robots=3, steps=3, start=0, state="RUNNING", completed=0,
                temp=50.0, vib=3.0, torque=50.0, pos=0.05, speed=1.0):
    rows = []
    for t in range(start, start + steps):
        for k in range(robots):
            rows.append({
                "ts": f"2026-11-03T08:00:{t:02d}", "step": t, "line_id": "LINE-A", "station_id": station,
                "robot_id": f"{station}_R{k + 1}", "state": state, "speed_factor": speed,
                "cycle_completed": completed, "cycles_done": 0, "last_cycle_time_s": 0.0,
                "joint_temp_c": temp, "vibration_mm_s": vib, "torque_nm": torque,
                "position_error_mm": pos, "buffer_in": 3, "buffer_out": 0, "buffer_out_capacity": 15,
                "line_finished_units": 0,
            })
    return pd.DataFrame(rows)


def test_output_has_one_command_per_robot(cfg, model):
    cmds, _ = control_station(make_events(), init_state(), cfg, model)
    assert list(cmds.columns) == COMMAND_COLUMNS
    assert sorted(cmds["robot_id"]) == ["S2_WELD_R1", "S2_WELD_R2", "S2_WELD_R3"]
    assert (cmds["issued_for_step"] == 2).all()


def test_speeds_up_when_output_below_target(cfg, model):
    cmds, st = control_station(make_events(completed=0), init_state(), cfg, model)
    assert (cmds["speed_setpoint"] > 1.0).all()
    assert cmds["speed_setpoint"].max() <= 1.0 + cfg.control.max_step + 1e-9  # rate limited
    assert st["integral"] > 0


def test_slows_down_when_blocked(cfg, model):
    st = init_state()
    st["last_speed"] = 1.1
    cmds, new = control_station(make_events(state="BLOCKED"), st, cfg, model)
    assert (cmds["speed_setpoint"] < 1.1).all()
    assert new["integral"] == 0.0  # anti-windup: integral frozen


def test_does_not_speed_up_when_starved(cfg, model):
    st = init_state()
    st["last_speed"] = 0.9
    cmds, _ = control_station(make_events(state="STARVED"), st, cfg, model)
    assert (cmds["speed_setpoint"] <= 0.9 + 1e-9).all()


def test_speeds_stay_within_limits(cfg, model):
    st = init_state()
    st["last_speed"] = cfg.control.speed_max
    for i in range(20):
        cmds, st = control_station(make_events(start=3 * i), st, cfg, model)
    assert cmds["speed_setpoint"].max() <= cfg.control.speed_max


def test_safety_interlock_triggers_estop(cfg, model):
    cmds, _ = control_station(make_events(temp=95.0), init_state(), cfg, model)
    assert set(cmds["action"]) == {"ESTOP"}


def test_estop_reset_after_cooling(cfg, model):
    cmds, _ = control_station(make_events(state="ESTOP", temp=40.0), init_state(), cfg, model,
                              ControlModes(predictive=False))
    assert set(cmds["action"]) == {"RESET"}


def test_predictive_maintenance_one_robot_at_a_time(cfg):
    risky = HealthModel(features=["joint_temp_c", "vibration_mm_s", "torque_nm", "position_error_mm", "speed_factor"],
                        means=[0] * 5, stds=[1] * 5, coefficients=[0, 0, 0, 0, 0], intercept=2.0)  # p = 0.88
    cmds, _ = control_station(make_events(), init_state(), cfg, risky)
    assert (cmds["action"] == "SCHEDULE_MAINTENANCE").sum() == 1


def test_medium_risk_derates(cfg):
    medium = HealthModel(features=["joint_temp_c", "vibration_mm_s", "torque_nm", "position_error_mm", "speed_factor"],
                         means=[0] * 5, stds=[1] * 5, coefficients=[0] * 5, intercept=0.5)  # p = 0.62
    cmds, _ = control_station(make_events(), init_state(), cfg, medium)
    assert (cmds["speed_setpoint"] <= cfg.control.derate_speed_cap).all()


def test_fixed_mode_keeps_nominal_speed(cfg, model):
    cmds, _ = control_station(make_events(), init_state(), cfg, model, ControlModes(False, False))
    assert (cmds["speed_setpoint"] == cfg.control.fixed_speed).all()


def test_late_data_is_ignored(cfg, model):
    _, st = control_station(make_events(start=10), init_state(), cfg, model)
    cmds, st2 = control_station(make_events(start=5), st, cfg, model)
    assert cmds.empty and st2 == st


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_model_probability_bounds(model, p):
    df = make_events(temp=30 + 100 * p, vib=1 + 20 * p, pos=0.01 + p)
    probs = model.predict_proba(df)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_derating_has_hysteresis(cfg):
    def model_with_p(intercept):
        return HealthModel(features=["joint_temp_c", "vibration_mm_s", "torque_nm", "position_error_mm",
                                     "speed_factor"], means=[0] * 5, stds=[1] * 5, coefficients=[0] * 5,
                           intercept=intercept)
    _, st = control_station(make_events(), init_state(), cfg, model_with_p(0.2))   # p = 0.55 -> derate
    assert st["derated"]
    cmds, st = control_station(make_events(start=3), st, cfg, model_with_p(-0.1), ControlModes())  # p = 0.475
    assert cmds["reason"].str.startswith("derated").all()  # stays derated inside the band
    cmds, st = control_station(make_events(start=6), st, cfg, model_with_p(-1.0), ControlModes())  # p = 0.27
    assert not cmds["reason"].str.startswith("derated").any() and st["derated"] == ""
