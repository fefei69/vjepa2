# Real-robot one-move test for V-JEPA 2-AC: the first move of the six H15 tasks, under the paper's pick-and-place
# protocol (app/vjepa_hanoi/goal_planner.py --protocol paper).
#
#   GPU node:   sbatch app/vjepa_hanoi/serve.sbatch      (prints the node and the tunnel command)
#   robot host: ssh -N -L 8766:<gpu node>:8766 <cluster login>        # keep open in another terminal
#               PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi.one_move_test --out runs/one_move_1
#   rehearsal:  ... one_move_test --dry-run --replay-archive <data/expert6_first_move/test.npz> --yes --out <dir>
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
#   3. the gripper opens (its reading is checked against the recorded open width) and the arm parks at the recorded
#      episode start pose: straight up first, then one leg above the rods;
#   4. the paper schedule runs; the arm then lifts straight up to the start height (a held ring stays in the gripper);
#   5. the operator records how far the move got, or marks the run invalid (set-up, network or server problem) so
#      --resume redoes it. Logs (every step with its camera frame) go to <out>/<run>[_aN].npz, never overwritten, and
#      one row per attempt to <out>/results.jsonl.
# Ctrl-C during a run stops it with no further motion; the operator is then asked before the arm lifts. Ctrl-C at a
# prompt is ignored (answer q where offered). On exit the arm lifts straight up, then tower_hanoi parks it.

import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys
import urllib.request
import zlib

import numpy as np

from app.vjepa_hanoi.arm_goal_client import new_log, post, run_steps, save_log
from app.vjepa_hanoi.robot_io import RECORDED_OPEN_COMMAND, START_POSE, STEP_S, Robot

BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]  # board_indices order, ring 1 first
CASES = ["AAAA>BAAA", "AAAA>CAAA", "BBBB>ABBB", "BBBB>CBBB", "CCCC>ACCC", "CCCC>BCCC"]
VARIANTS = {"faithful": {"fix_x": False, "hold_grasp": False}, "hold": {"fix_x": False, "hold_grasp": True}}
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


def git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
    args = p.parse_args()
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
        if len(fixed_schedule) != 3 or min(fixed_schedule) < 1:
            p.error("--schedule needs three positive step counts, one per goal (e.g. 29,16,11)")
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
    print(f"server: {health.get('checkpoint')} | protocol {health['protocol']} | planner {health['planner']}")

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
        json.dump({"started": started, "args": vars(args), "server": health, "vjepa2_commit": git_commit(),
                   "start_pose": START_POSE, "robot_config": robot.config}, f, indent=1)  # fmt: skip

    def q(prompt, choices=None, default=None, dry=None):
        """Ask the operator; with --yes (dry run only) answer `dry` instead."""
        return ask(prompt, choices, default, dry if args.yes else None)

    def offer_lift():
        if q("Lift the arm straight up to the start height now? [y/n]: ", {"y", "n"}, dry="y") == "y":
            try:
                robot.lift(START_POSE[2])
                return None
            except Exception as e:
                print(f"lift failed: {type(e).__name__}: {e}")
                return f"{type(e).__name__}: {e}"
        return "operator declined the lift"

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
            attempt = 0
            while os.path.exists(os.path.join(args.out, f"{run}{f'_a{attempt}' if attempt else ''}.npz")):
                attempt += 1
            log_name = f"{run}{f'_a{attempt}' if attempt else ''}.npz"
            meta = {"run": run, "log_file": log_name, "case": case, "variant": variant, "repeat": rep,
                    "options": VARIANTS[variant], "dry_run": args.dry_run, "steps": 0, "error": None}  # fmt: skip
            log, phase = new_log(), "set-up"
            try:
                # everything that can fail without moving the arm comes first
                goal = post(args.server, "/goal", transition=case, **VARIANTS[variant])
                if goal.get("options") != VARIANTS[variant]:
                    raise RuntimeError(f"server options {goal.get('options')} != requested {VARIANTS[variant]}")
                schedule = fixed_schedule or goal.get("schedule")
                if not schedule or len(schedule) != goal["n_goals"] or min(schedule) < 1:
                    raise RuntimeError(f"bad schedule {schedule} for {goal['n_goals']} goals")
                switch, budget = [int(x) for x in np.cumsum(schedule)[:-1]], int(sum(schedule))
                meta.update({"schedule": schedule, "goal": goal["goal"], "planner": goal["planner"],
                             "box": goal.get("box")})  # fmt: skip
                print(f"goal: {goal['goal']}\nschedule {schedule} steps ({budget * STEP_S:.0f} s of motion)")
                phase = "parking"
                if args.dry_run:
                    robot.replay_from(*replay[case])
                robot.gripper("open")
                jaw = robot.jaw()
                meta["open_jaw"] = jaw
                if jaw < OPEN_JAW_MIN:
                    raise RuntimeError(f"open jaw reads {jaw:.4f} < {OPEN_JAW_MIN:.4f}: the model would see a "
                                       f"half-closed gripper (recorded open width {RECORDED_OPEN_COMMAND})")  # fmt: skip
                robot.park(START_POSE)
                phase = "running"
                seed = zlib.crc32(f"{run}_{attempt}".encode()) % 100000
                meta["seed"] = seed
                t0 = datetime.datetime.now()
                run_steps(robot, args.server, goal, budget, switch, log, out=lambda m: print(m, flush=True), seed=seed)
                meta["seconds"] = (datetime.datetime.now() - t0).total_seconds()
                phase = "lifting"
                robot.lift(START_POSE[2])
                phase = "done"
            except KeyboardInterrupt:
                meta["error"] = f"stopped by the operator (Ctrl-C) while {phase}"
            except Exception as e:  # arm readback, planner or network errors: stop this run, keep the log
                meta["error"] = f"{type(e).__name__} while {phase}: {e}"
            meta["steps"] = len(log["pose"])
            if meta["error"]:
                print(f"\nrun stopped after {meta['steps']} steps: {meta['error']}")
                if phase in ("parking", "running", "lifting"):
                    meta["lift_error"] = offer_lift()
            save_log(os.path.join(args.out, log_name), log, **meta)
            progress = q("How far did it get?\n" + "\n".join(f"  {k}: {v}" for k, v in PROGRESS.items()) + "\n> ",
                         set(PROGRESS), dry="0")  # fmt: skip
            valid = progress != "x"
            disturbed = q("Did it move or knock any other ring? [y/n]: ", {"y", "n"}, dry="n") if valid else None
            note = q("Note (optional): ", None, default="", dry="dry run")
            meta.update({"valid": valid, "progress": int(progress) if valid else None,
                         "progress_text": PROGRESS[progress], "other_rings_disturbed": disturbed == "y" if valid else None,
                         "success": valid and progress == "5" and disturbed == "n" and not args.dry_run,
                         "note": note, "time": datetime.datetime.now().isoformat(timespec="seconds")})  # fmt: skip
            with open(results_path, "a") as f:
                f.write(json.dumps(meta) + "\n")
            if q("Is the gripper holding a ring? Open it now? [y/n]: ", {"y", "n"}, dry="n") == "y":
                robot.gripper("open")
            if meta["error"] and q("Continue with the next run? [y/n]: ", {"y", "n"}, dry="y") != "y":
                break
    finally:
        try:
            if not args.yes and q("Session over. Is a ring still in the gripper? Open it before parking? [y/n]: ",
                                  {"y", "n"}) == "y":  # fmt: skip
                robot.gripper("open")
        finally:
            robot.close()  # lifts straight up, then tower_hanoi's joint-space park
    rows = [json.loads(x) for x in open(results_path)] if os.path.exists(results_path) else []
    print(f"\nresults ({results_path}, valid runs only):")
    for v in variants:
        rv = [x for x in rows if x["variant"] == v and x.get("valid", True)]
        if rv:
            print(f"  {v:9s} success {sum(x['success'] for x in rv)}/{len(rv)}, progress {[x['progress'] for x in rv]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
