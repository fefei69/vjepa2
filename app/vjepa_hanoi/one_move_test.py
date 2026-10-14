# Real-robot one-move test for V-JEPA 2-AC: the first move of the six H15 tasks, under the paper's pick-and-place
# protocol (app/vjepa_hanoi/goal_planner.py --protocol paper).
#
#   server (robot machine's GPU; the bundle comes from app/vjepa_hanoi/export_robot_bundle.py):
#       python -m app.vjepa_hanoi.goal_planner --fname <bundle>/config.yaml --checkpoint <bundle>/weights.pt \
#           --goal_bundle <bundle>/goals.npz --protocol paper --host 127.0.0.1 --port 8766
#       (16 GB GPU: add --predictor_chunk 200 and run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
#   runner (tower_hanoi's environment, another terminal; keep $PYTHONPATH, it carries ROS's rclpy for the camera):
#       EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.one_move_test --execute \
#           --out runs/one_move_1 --tower-hanoi <tower_hanoi>
#   The real arm moves only with --execute plus EXECUTE=1 (tower_hanoi's hardware rule); a dry run takes neither.
#   (or serve on a cluster GPU with app/vjepa_hanoi/serve.sbatch and an ssh tunnel to port 8766)
#   rehearsal:  ... one_move_test --dry-run --replay-archive <data/expert6_first_move/test.npz> --yes --out <dir>
#               (needs the recordings at the archive's paths)
#
# Cases: ring 1 off a full stack, AAAA>BAAA, AAAA>CAAA, BBBB>ABBB, BBBB>CBBB, CCCC>ACCC, CCCC>BCCC. These are the six
# test cases of data/expert6_first_move (episode 9 of each recording), the dataset the VLA and world-action models
# train on. Variants (--variants): faithful (the paper protocol as published) and hold (--hold_grasp: the gripper is
# held closed while goal 2 is active, a disclosed deviation). The server's fix_x option is not offered here: in these
# moves the arm starts 8 cm back and has to move x to reach the pegs.
#
# Each run (case x variant x repeat):
#   1. the operator sets the board, ring 1 on top of a full stack on the start peg, and checks the gripper is empty;
#   2. the server sets the case's three goal images (frames of a training demo of the same move), the variant's options
#      and the case's schedule (the median timing of that move's training demos), all checked before the arm moves;
#   3. a camera frame must arrive; then the gripper opens (its reading is checked against the recorded open width) and
#      the arm parks at the start: straight up first, then one leg above the rods. --start hover (default, as
#      tower_hanoi's Q-WM runner) starts above the source peg's grasp point at the start height, where the demos pass
#      at the end of their approach; goal 1's step budget is then shortened by the demos' approach steps
#      (HOVER_APPROACH_STEPS), so [29, 16, 11] becomes [13, 16, 11]. For a model trained at another fps (the server
#      reports step_rows != 8) the server's hover_schedule is used instead, the same medians counted from each demo's
#      hover, since rounding the approach to whole 1 s steps can be off by one. --start episode starts at the demos'
#      first pose, 8 cm behind the rods in line with peg B, with the full schedule;
#   4. the paper schedule runs, each step executed as robot_io describes (Q-WM's move timing, settle window, still-arm
#      frames, and the abort-only missed-grasp and low-lateral checks); then the retreat: below the low floor the jaw
#      opens first (a ring stays on its rod), then straight up to the start height (a ring held higher stays held);
#   5. the operator records how far the move got (and, for a placement, whether another ring moved), or marks the run
#      invalid (set-up, network or server problem) so --resume redoes it; the gripper questions come only when the jaw
#      reads closed. Logs (every step with its camera frame) go to <out>/<run>[_aN].npz, never overwritten, one
#      JSON line per step to <out>/<run>[_aN].steps.jsonl as the run goes, and one row per attempt to results.jsonl;
#      If the retreat opened a jaw that was not open, the runner (not the model) released whatever it held: that is
#      recorded (runner_released_jaw) and the run cannot count as a success. A run that fails before any motion is
#      recorded as invalid without asking;
#   6. after a run without errors the arm moves at the start height to the start pose, 8 cm behind the rods, so the
#      board is reset under a clear arm (after an error, the next run's park does it once the operator is ready).
# Ctrl-C during a run stops it with no further motion; the operator is then asked before the retreat. Ctrl-C at a
# prompt is ignored (answer q where offered); it no longer stops the ROS camera (robot_io turns off rclpy's SIGINT
# handler). On exit the arm retreats, then tower_hanoi parks it; if the retreat fails, it parks only when the operator
# types PARK.

import argparse
import datetime
import hashlib
import itertools
import json
import os
import subprocess
import sys
import urllib.request
import zlib

import numpy as np

from app.vjepa_hanoi.arm_goal_client import EXECUTE_HELP, new_log, post, require_execute, run_steps, save_log
from app.vjepa_hanoi.robot_io import RECORDED_OPEN_COMMAND, SPEED_CAP_M_S, START_POSE, STEP_SLOW_S, Robot

BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]  # board_indices order, ring 1 first
CASES = ["AAAA>BAAA", "AAAA>CAAA", "BBBB>ABBB", "BBBB>CBBB", "CCCC>ACCC", "CCCC>BCCC"]
_OFF = {"fix_x": False, "hold_grasp": False, "hold_open": False, "hold_xy": False, "subgoals": "paper3"}
VARIANTS = {  # every server option, so /goal echoes them exactly; see robot_results/2026-10-13/cluster_offline
    "faithful": _OFF,  # paper baseline (with the server's default --rollout 1)
    "hold": _OFF | {"hold_grasp": True},  # task aid
    "stages": _OFF | {"subgoals": "stages"},  # planner deviation: one goal per motion stage (use --rollout 3)
    "stages_xy": _OFF | {"subgoals": "stages", "hold_xy": True, "hold_grasp": True},  # + task aids
    "stages_open_xy": _OFF | {"subgoals": "stages", "hold_open": True, "hold_xy": True, "hold_grasp": True},
}
PROGRESS = {  # furthest point the move reached, judged by the operator
    "0": "never reached the ring",
    "1": "reached the ring, no grasp",
    "2": "grasped the ring",
    "3": "lifted the ring off the stack",
    "4": "carried it over the target peg",
    "5": "ring 1 placed on the target peg",
    "x": "INVALID run (set-up, network or server problem): redo it",
}
OPEN_JAW_MIN = RECORDED_OPEN_COMMAND - 0.001  # an open jaw below this reads as > 30% closed to the model
# Model steps from a one-move demo's first row to the end of its approach (above the source peg's grasp point), median
# over the 8 training demos of each case: 15.9 for source pegs A and C, 13.2 for B (range +-0.1). With --start hover
# goal 1's budget drops by this, which is the demos' own timing from the hover: [13, 16, 11] or [13, 24, 11].
HOVER_APPROACH_STEPS = {"A": 16, "B": 13, "C": 16}


def ask(prompt, choices=None, default=None, auto=None):
    """Operator input. Pending keystrokes are discarded first; Ctrl-C at a prompt is ignored."""
    if auto is not None:
        return auto
    try:
        import termios

        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass
    while True:
        try:
            raw = input(prompt).strip()
        except KeyboardInterrupt:
            print("\n  (Ctrl-C is ignored at prompts; answer the question, or q where offered)")
            continue
        if not raw and default is not None:
            return default
        if choices is None:
            return raw
        if raw.lower() in choices:
            return raw.lower()
        print(f"  choose one of {sorted(choices)}")


def git_state():
    """HEAD, the files that differ from it, and a hash of that difference: the code that actually ran."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*cmd):
        return subprocess.check_output(["git", "-C", here, *cmd], text=True)

    try:
        return {"vjepa2_commit": git("rev-parse", "HEAD").strip(),
                "vjepa2_changed_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
                "vjepa2_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()}  # fmt: skip
    except Exception:
        return {"vjepa2_commit": "unknown"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8766")
    p.add_argument("--out", required=True, help="session directory (logs + results.jsonl)")
    p.add_argument("--cases", default=",".join(CASES), help="comma-separated transitions")
    p.add_argument("--variants", default="faithful,hold")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--order", choices=("case", "variant"), default="case",
                   help="case: every variant of a case before the next case; variant: every case per variant")
    p.add_argument("--schedule", help="steps per goal for every case, e.g. 29,16,11 (default: the server's per case)")
    p.add_argument("--resume", action="store_true", help="continue a session: skip runs already validly recorded")
    p.add_argument("--tower-hanoi", default=os.environ.get("TOWER_HANOI", os.path.expanduser("~/ws/tower_hanoi")))
    p.add_argument("--address", default="192.168.1.3")
    p.add_argument("--dry-run", action="store_true", help="tower_hanoi DryRunArm + a replayed test move as camera")
    p.add_argument("--replay-archive", help="dry run: data/expert6_first_move/test.npz (each case replays its move)")
    p.add_argument("--yes", action="store_true", help="dry run only: no operator prompts")
    p.add_argument("--start", choices=("hover", "episode"), default="hover",
                   help="hover: above the source peg (Q-WM's start; goal 1 budget shortened); episode: the demos' "
                        "first pose 8 cm behind the rods")  # fmt: skip
    p.add_argument("--execute", action="store_true", help=EXECUTE_HELP)
    args = p.parse_args()
    require_execute(p, args)
    cases, variants = args.cases.split(","), args.variants.split(",")
    for c in cases:
        if c not in CASES:
            p.error(f"unknown case {c}; the one-move cases are {CASES}")
    for v in variants:
        if v not in VARIANTS:
            p.error(f"unknown variant {v}; choose from {list(VARIANTS)}")
    if args.yes and not args.dry_run:
        p.error("--yes skips the operator's board and outcome checks: dry run only")
    fixed_schedule = None
    if args.schedule:
        try:
            fixed_schedule = [int(x) for x in args.schedule.split(",")]
        except ValueError:
            fixed_schedule = []
        if not fixed_schedule or min(fixed_schedule) < 1:
            p.error("--schedule needs positive step counts, one per goal (e.g. 29,16,11; 6 with stage subgoals)")
    replay = {}
    if args.dry_run:
        if not args.replay_archive:
            p.error("--dry-run needs --replay-archive <expert6_first_move/test.npz>")
        z = np.load(args.replay_archive)
        paths = [str(x) for x in z["source_paths"]]
        for e in np.unique(z["episode_indices"]):
            i = np.nonzero(z["episode_indices"] == e)[0][0]
            t = f"{BOARDS[z['board_indices'][i]]}>{BOARDS[z['goal_board_indices'][i]]}"
            replay[t] = (paths[z["file_indices"][i]], int(z["segment_bounds"][i, 0]))
        missing = [c for c in cases if c not in replay]
        if missing:
            p.error(f"the replay archive has no test move for {missing}")
        args.replay, args.replay_episode = replay[cases[0]][0], 0

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    rows = [json.loads(x) for x in open(results_path) if x.strip()] if os.path.exists(results_path) else []
    if rows and not args.resume:
        p.error(f"{results_path} already has {len(rows)} rows: pass --resume to continue it, or use a new --out")
    if any(r.get("dry_run") != args.dry_run for r in rows):
        p.error(f"{results_path} mixes dry-run and real runs with this session; use a new --out")
    done = {r["run"] for r in rows if r.get("valid", True)}
    with urllib.request.urlopen(args.server + "/health", timeout=30) as resp:
        health = json.loads(resp.read())
    if health.get("protocol") != "paper":
        p.error(f"the server runs protocol {health.get('protocol')!r}; start it with --protocol paper")
    step_rows = int(health.get("step_rows", 8))  # raw rows per model step (8 at fps 4)
    if float(health.get("max_step_m", 0.0)) > SPEED_CAP_M_S * STEP_SLOW_S + 1e-9:  # the move-time rule's assumption
        p.error(f"the server allows {health['max_step_m'] * 1e3:.1f} mm steps; the move-time rule keeps steps under "
                f"36 mm/s only up to {SPEED_CAP_M_S * STEP_SLOW_S * 1e3:.0f} mm: use the default --max_speed 0.10"
                + ("" if step_rows == 8 else f" with an fps-4 model (this model's step is {step_rows} rows; a longer "
                   "step needs a longer move time, which robot_io does not do yet)"))  # fmt: skip
    print(f"server: {health.get('checkpoint')} | protocol {health['protocol']} | planner {health['planner']} | model "
          f"step {step_rows} rows ({step_rows / 30:.3f} s)")  # fmt: skip

    runs = []
    for r in range(args.repeats):
        if args.order == "case":
            runs += [(c, v, r) for c in cases for v in variants]
        else:
            runs += [(c, v, r) for v in variants for c in cases]
    runs = [x for x in runs if f"{x[0].replace('>', '-')}_{x[1]}_r{x[2]}" not in done]
    print(f"{len(runs)} runs to do ({len(done)} already recorded)")

    robot = Robot(args)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(args.out, f"session_{started.replace(':', '')}.json"), "w") as f:
        json.dump({"started": started, "args": vars(args), "server": health, **git_state(),
                   "start": args.start, "episode_start_pose": START_POSE, "hover_approach_steps": HOVER_APPROACH_STEPS,
                   "robot_config": robot.config}, f, indent=1)  # fmt: skip

    def q(prompt, choices=None, default=None, dry=None):
        """Ask the operator; with --yes (dry run only) answer `dry` instead."""
        return ask(prompt, choices, default, dry if args.yes else None)

    def jaw_closed():
        """The gripper questions are asked only when the jaw does not read open (or cannot be read)."""
        try:
            return robot.jaw() < OPEN_JAW_MIN
        except Exception:
            return True

    def offer_retreat():
        """(error or None, the jaw reading before the retreat opened it or None)."""
        if q("Retreat now: below the low floor open the jaw first (a ring stays on its rod), then straight up to the "
             "start height? [y/n]: ", {"y", "n"}, dry="y") == "y":  # fmt: skip
            try:
                return None, robot.retreat()
            except BaseException as e:  # Ctrl-C included: keep this run's log and result
                print(f"retreat failed: {type(e).__name__}: {e}")
                return f"{type(e).__name__}: {e}", None
        return "operator declined the retreat", None

    try:
        for n, (case, variant, rep) in enumerate(runs, 1):
            run = f"{case.replace('>', '-')}_{variant}_r{rep}"
            start, target = case.split(">")
            print(f"\n=== run {n}/{len(runs)}: {run}  ({start} -> {target}: ring 1 from peg {start[0]} to peg "
                  f"{target[0]}; options {VARIANTS[variant]})")  # fmt: skip
            a = q(f"Set the board to {start} (all four rings on peg {start[0]}), make sure the gripper is empty. "
                  f"[Enter] go, [s] skip, [q] quit: ", {"", "s", "q"}, dry="")  # fmt: skip
            if a == "q":
                break
            if a == "s":
                continue
            attempt = 0  # a killed attempt leaves only its .steps.jsonl: it keeps that file; the redo gets a new seed
            while any(os.path.exists(os.path.join(args.out, f"{run}{f'_a{attempt}' if attempt else ''}{ext}"))
                      for ext in (".npz", ".steps.jsonl")):  # fmt: skip
                attempt += 1
            log_name = f"{run}{f'_a{attempt}' if attempt else ''}.npz"
            step_log = log_name.replace(".npz", ".steps.jsonl")
            meta = {"run": run, "log_file": log_name, "step_log_file": step_log, "case": case, "variant": variant,
                    "repeat": rep, "options": VARIANTS[variant], "dry_run": args.dry_run, "steps": 0,
                    "error": None}  # fmt: skip
            log, phase, retreated = new_log(), "set-up", False
            try:
                # everything that can fail without moving the arm comes first
                goal = post(args.server, "/goal", transition=case, **VARIANTS[variant])
                if goal.get("options") != VARIANTS[variant]:
                    raise RuntimeError(f"server options {goal.get('options')} != requested {VARIANTS[variant]}")
                schedule = fixed_schedule or goal.get("schedule")
                if args.start == "hover" and not fixed_schedule and schedule:  # the demos' timing from the hover
                    if step_rows == 8:
                        schedule = [int(schedule[0]) - HOVER_APPROACH_STEPS[start[0]]] + [int(s) for s in schedule[1:]]
                    elif goal.get("hover_schedule"):
                        schedule = [int(s) for s in goal["hover_schedule"]]
                    else:
                        raise RuntimeError("a model with a non-fps-4 step needs the server's hover_schedule: "
                                           "re-export the bundle (export_robot_bundle --goals_only)")  # fmt: skip
                if not schedule or len(schedule) != goal["n_goals"] or min(schedule) < 1:
                    raise RuntimeError(f"bad schedule {schedule} for {goal['n_goals']} goals")
                switch, budget = [int(x) for x in np.cumsum(schedule)[:-1]], int(sum(schedule))
                start_pose = robot.hover_pose(start[0]) if args.start == "hover" else START_POSE
                meta.update({"schedule": schedule, "server_schedule": goal.get("schedule"), "start": args.start,
                             "step_rows": step_rows, "server_hover_schedule": goal.get("hover_schedule"),
                             "start_pose": start_pose, "goal": goal["goal"], "planner": goal["planner"],
                             "box": goal.get("box")})  # fmt: skip
                where = f"above peg {start[0]}" if args.start == "hover" else "at the demos' first pose"
                print(f"goal: {goal['goal']}\nstart {where} {np.round(np.asarray(start_pose[:3]) * 1e3, 1).tolist()} "
                      f"mm; schedule {schedule} = {budget} model steps")  # fmt: skip
                robot.check_camera()  # no frames (camera down): fails here, before any motion
                phase = "parking"
                if args.dry_run:  # with --start hover, replay from the demo's approach end (the approach in raw rows)
                    path, row = replay[case]
                    robot.replay_from(path, row + (8 * HOVER_APPROACH_STEPS[start[0]] if args.start == "hover" else 0),
                                      step_rows)  # fmt: skip
                robot.gripper("open")
                jaw = robot.jaw()
                meta["open_jaw"] = jaw
                if jaw < OPEN_JAW_MIN:
                    raise RuntimeError(f"open jaw reads {jaw:.4f} < {OPEN_JAW_MIN:.4f}: the model would see a "
                                       f"half-closed gripper (recorded open width {RECORDED_OPEN_COMMAND})")  # fmt: skip
                robot.park(start_pose)
                phase = "running"
                seed = zlib.crc32(f"{run}_{attempt}".encode()) % 100000
                meta["seed"] = seed
                t0 = datetime.datetime.now()
                run_steps(robot, args.server, goal, budget, switch, log, out=lambda m: print(m, flush=True), seed=seed,
                          intent="open", step_log=os.path.join(args.out, step_log))  # fmt: skip
                meta["seconds"] = (datetime.datetime.now() - t0).total_seconds()
                phase = "retreating"
                opened_from = robot.retreat()
                retreated, phase = True, "done"
            except KeyboardInterrupt:
                meta["error"] = f"stopped by the operator (Ctrl-C) while {phase}"
            except Exception as e:  # arm readback, grasp, step-check, planner or network errors: stop, keep the log
                meta["error"] = f"{type(e).__name__} while {phase}: {e}"
            meta["steps"] = len(log["pose"])
            meta["final_gripper_intent"] = log["gripper_intent"][-1] if log["gripper_intent"] else None
            if meta["error"]:
                print(f"\nrun stopped after {meta['steps']} steps: {meta['error']}")
                opened_from = None
                if phase in ("parking", "running", "retreating"):
                    meta["retreat_error"], opened_from = offer_retreat()
                    retreated = meta["retreat_error"] is None
            # the runner, not the model, released a ring if the retreat opened a jaw that was not open
            runner_released = opened_from is not None and opened_from < OPEN_JAW_MIN
            meta.update({"retreat_opened_jaw_from": opened_from, "runner_released_jaw": runner_released})
            save_log(os.path.join(args.out, log_name), log, **meta)
            if meta["error"] and phase == "set-up":  # nothing moved: an invalid run that --resume redoes
                print("nothing moved: recorded as invalid (x); --resume redoes it")
                progress = "x"
            else:
                if runner_released:
                    print(f"NOTE: the retreat opened the jaw (reading {opened_from:.4f}) below the low floor: a ring "
                          f"now on a rod was released by the runner, not the model, and cannot count as success")
                progress = q("How far did it get?\n" + "\n".join(f"  {k}: {v}" for k, v in PROGRESS.items())
                             + "\n> ", set(PROGRESS), dry="0")  # fmt: skip
            valid = progress != "x"
            # only a placement can be a success, and success needs nothing else disturbed: ask only then
            disturbed = None
            if progress == "5":
                disturbed = q("Did it move or knock any other ring? [y/n]: ", {"y", "n"}, dry="n")
            meta.update({"valid": valid, "progress": int(progress) if valid else None,
                         "progress_text": PROGRESS[progress],
                         "other_rings_disturbed": None if disturbed is None else disturbed == "y",
                         "success": (valid and progress == "5" and disturbed == "n" and not runner_released
                                     and not args.dry_run),
                         "time": datetime.datetime.now().isoformat(timespec="seconds")})  # fmt: skip
            with open(results_path, "a") as f:
                f.write(json.dumps(meta) + "\n")
            if jaw_closed() and q("The jaw reads closed. Open it now (a held ring drops)? [y/n]: ", {"y", "n"},
                                  dry="n") == "y":  # fmt: skip
                robot.gripper("open")
            if retreated and not meta["error"]:  # one leg at the start height to the start pose, 8 cm behind the rods
                # for the board reset (after an error the next run's park does the same once the operator is ready)
                print("clearing the rods for the board reset: one leg at the start height to the start pose")
                try:
                    robot.park(START_POSE)
                except Exception as e:  # the arm's state is uncertain: end the session (the exit retreat follows)
                    print(f"move to the start pose failed: {type(e).__name__}: {e}")
                    break
            more = n < len(runs)  # no question after the last run
            if meta["error"] and more and q("Continue with the next run? [y/n]: ", {"y", "n"}, dry="y") != "y":
                break
    finally:
        try:
            if not args.yes and jaw_closed() and q("Session over and the jaw reads closed. Open it before parking? "
                                                   "[y/n]: ", {"y", "n"}) == "y":  # fmt: skip
                robot.gripper("open")
        finally:
            robot.close(ask=ask)  # the retreat, then tower_hanoi's joint-space park (on PARK only, if it failed)
    rows = [json.loads(x) for x in open(results_path)] if os.path.exists(results_path) else []
    print(f"\nresults ({results_path}, valid runs only):")
    for v in variants:
        rv = [x for x in rows if x["variant"] == v and x.get("valid", True)]
        if rv:
            print(f"  {v:9s} success {sum(x['success'] for x in rv)}/{len(rv)}, progress {[x['progress'] for x in rv]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
