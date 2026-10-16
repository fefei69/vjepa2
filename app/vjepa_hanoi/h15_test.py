# Real-robot H15 test for V-JEPA 2-AC: the six full-stack tasks (AAAA>BBBB, AAAA>CCCC, BBBB>AAAA, BBBB>CCCC, CCCC>AAAA,
# CCCC>BBBB), each played as its 15 optimal game moves, every move run exactly as one_move_test runs one move.
#
# Per move, the H1 protocol of one_move_test (the setup that works on one move): the server sets the move's goals
# (--variants stages: the ends of motion stages 3-7 and the final stage of the first training demo of the same
# transition) and its schedule (the median timing of that transition's training demos, counted from the hover); the
# gripper opens and the arm parks above the moving ring's peg at the start height (robot_io.hover_pose; the demos of
# every H15 move pass there at the end of their approach, within 1 mm); the schedule runs (robot_io's step rules and
# abort-only checks, unchanged); then the retreat (below the low floor the jaw opens first, then straight up). The
# moves between two game moves are the runner's (retreat, park at the next hover), as one_move_test's start is.
# So a move here differs from an H1 run only in the board it starts from, which the previous moves left.
#
#   server (as for one_move_test, with a bundle that has every H15 move, e.g. export.sbatch MODEL=efps1h15):
#       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m app.vjepa_hanoi.goal_planner \
#           --fname <bundle>/config.yaml --checkpoint <bundle>/weights.pt --goal_bundle <bundle>/goals.npz \
#           --protocol paper --subgoals stages --sampling cat --guard_floor_mm 166.13 --box_feasible \
#           --max_speed 0.036 --predictor_chunk 200
#   runner (tower_hanoi's environment):
#       EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.h15_test --execute \
#           --tasks AAAA>CCCC --variants stages --schedule_scale 1.5 --out runs/h15_efps1_1 --tower-hanoi <tower_hanoi>
#   rehearsal: ... h15_test --dry-run --replay-archive <data/expert6_single_move/test.npz> --yes --out <dir>
#
# The operator judges every move (there is no success detector): how far it got (0-5) and, for a placement, whether
# another ring moved. A move succeeds on 5 with nothing else disturbed and the jaw not released by the runner. After a
# failed move the operator chooses:
#   c  set the board by hand to the board this move should have made and continue with the next move
#   r  set the board back to this move's start board and run the move again
#   q  end this task
# For c and r the arm first moves to the start pose behind the rods (as one_move_test after a run). After a move whose
# retreat was declined or failed, the arm does not move again: the session ends with its exit retreat (jaw open first
# below the low floor; PARK only if that fails), as one_move_test's next motion after an error waits for the operator.
# c and r are interventions: they are recorded, and the task's strict result (moves completed before the first
# failure, success = all 15 without one) ends at the first failure. The moves after an intervention still give
# per-move results, so later moves are measured too. Logs: <out>/<task>_<variant>_r<rep>_m<move>[_aN].npz (every step
# with its camera frame) and .steps.jsonl; one JSON line per move attempt in <out>/results.jsonl; --append adds to an
# existing session, --from-move N starts a task at its move N (the operator sets that move's start board).
# Ctrl-C during a move stops it with no further motion; the operator is then asked before the retreat.

import argparse
import datetime
import json
import os
import sys
import urllib.request
import zlib

import numpy as np

from app.vjepa_hanoi.arm_goal_client import EXECUTE_HELP, new_log, post, require_execute, run_steps, save_log
from app.vjepa_hanoi.one_move_test import OPEN_JAW_MIN, VARIANTS, ask, git_state
from app.vjepa_hanoi.robot_io import SPEED_CAP_M_S, START_POSE, STEP_SLOW_S, Robot, workspace_low_floor

TASKS = ["AAAA>BBBB", "AAAA>CCCC", "BBBB>AAAA", "BBBB>CCCC", "CCCC>AAAA", "CCCC>BBBB"]
PROGRESS = {  # furthest point the move reached, judged by the operator (one_move_test's, for any ring)
    "0": "never reached the ring",
    "1": "reached the ring, no grasp",
    "2": "grasped the ring",
    "3": "lifted the ring off its peg",
    "4": "carried it over the target peg",
    "5": "ring placed on the target peg",
    "x": "INVALID attempt (set-up, network or server problem): redo it",
}


def solve(start, goal):
    """The optimal game moves start -> goal for full stacks: transitions 'board>board', ring 1 (the smallest) first
    in a board string. 15 for four rings."""
    board, out = list(start), []

    def move(n, src, dst, aux):  # rings 1..n from src to dst
        if n == 0:
            return
        move(n - 1, src, aux, dst)
        before = "".join(board)
        board[n - 1] = dst
        out.append(f"{before}>{''.join(board)}")
        move(n - 1, aux, dst, src)

    move(len(start), start[0], goal[0], ({"A", "B", "C"} - {start[0], goal[0]}).pop())
    assert "".join(board) == goal
    return out


def moving_ring(transition):
    """(ring number 1-4, source peg, target peg) of a one-ring transition."""
    a, b = transition.split(">")
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    assert len(diff) == 1, transition
    return diff[0] + 1, a[diff[0]], b[diff[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8766")
    p.add_argument("--out", required=True, help="session directory (logs + results.jsonl)")
    p.add_argument("--tasks", default=",".join(TASKS), help="comma-separated, e.g. AAAA>CCCC")
    p.add_argument("--variants", default="stages", help="one_move_test's variants (the server options per move)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--from-move", type=int, default=1, help="start each task at this move (1-15)")
    p.add_argument("--schedule_scale", type=float, default=1.0,
                   help="as one_move_test: each goal's steps x this, rounded up (1.5 with --max_speed 0.036)")
    p.add_argument("--append", action="store_true", help="add to an existing session's results.jsonl")
    p.add_argument("--tower-hanoi", default=os.environ.get("TOWER_HANOI", os.path.expanduser("~/ws/tower_hanoi")))
    p.add_argument("--address", default="192.168.1.3")
    p.add_argument("--dry-run", action="store_true", help="tower_hanoi DryRunArm + the replayed held-out move as camera")
    p.add_argument("--replay-archive", help="dry run: data/expert6_single_move/test.npz (each move replays its own)")
    p.add_argument("--yes", action="store_true", help="dry run only: no operator prompts (every move judged 0, then c)")
    p.add_argument("--execute", action="store_true", help=EXECUTE_HELP)
    args = p.parse_args()
    require_execute(p, args)
    tasks, variants = args.tasks.split(","), args.variants.split(",")
    for t in tasks:
        if t not in TASKS:
            p.error(f"unknown task {t}; the H15 tasks are {TASKS}")
    for v in variants:
        if v not in VARIANTS:
            p.error(f"unknown variant {v}; choose from {list(VARIANTS)}")
    if not 1 <= args.from_move <= 15:
        p.error("--from-move is 1-15")
    if args.yes and not args.dry_run:
        p.error("--yes skips the operator's board and outcome checks: dry run only")
    replay = {}
    if args.dry_run:
        if not args.replay_archive:
            p.error("--dry-run needs --replay-archive <expert6_single_move/test.npz>")
        from app.vjepa_hanoi.paper_protocol_eval import moves

        test_moves, paths, _ = moves(args.replay_archive)
        for m in test_moves:  # each move's camera: the held-out demo of that move from its hover (last stage-2 row)
            replay.setdefault(m["transition"], (paths[m["file"]], int(m["rows"][m["stages"] == 2].max())))
        missing = sorted({mv for t in tasks for mv in solve(*t.split(">")) if mv not in replay})
        if missing:
            p.error(f"the replay archive has no held-out move {missing}")
        args.replay, args.replay_episode = replay[solve(*tasks[0].split(">"))[0]][0], 0

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    rows = [json.loads(x) for x in open(results_path) if x.strip()] if os.path.exists(results_path) else []
    if rows and not args.append:
        p.error(f"{results_path} already has {len(rows)} rows: pass --append to add to it, or use a new --out")
    if any(r.get("dry_run") != args.dry_run for r in rows):
        p.error(f"{results_path} mixes dry-run and real runs with this session; use a new --out")
    with urllib.request.urlopen(args.server + "/health", timeout=30) as resp:
        health = json.loads(resp.read())
    if health.get("protocol") != "paper":
        p.error(f"the server runs protocol {health.get('protocol')!r}; start it with --protocol paper")
    step_rows = int(health.get("step_rows", 8))
    if float(health.get("max_step_m", 0.0)) > SPEED_CAP_M_S * STEP_SLOW_S + 1e-9:  # one_move_test's move-time check
        p.error(f"the server allows {health['max_step_m'] * 1e3:.1f} mm steps; the move-time rule keeps steps under "
                f"36 mm/s only up to {SPEED_CAP_M_S * STEP_SLOW_S * 1e3:.0f} mm (an fps-1 model: --max_speed 0.036)")
    server_floor = (health.get("planner") or {}).get("low_floor")
    if server_floor is not None:  # the planner must avoid this arm's low-lateral guard, not another arm's
        floor = workspace_low_floor(args.tower_hanoi)
        if abs(float(server_floor) - floor) > 1e-3:
            p.error(f"the server plans with a low floor of {float(server_floor) * 1e3:.2f} mm "
                    f"(--guard_floor_mm) but this arm's is {floor * 1e3:.2f} mm (robot_io: transit_z - 25 mm); "
                    f"restart it with --guard_floor_mm {floor * 1e3:.2f}")  # fmt: skip
    print(f"server: {health.get('checkpoint')} | protocol {health['protocol']} | planner {health['planner']} | model "
          f"step {step_rows} rows ({step_rows / 30:.3f} s)")  # fmt: skip
    # every move of every task must have its goals and hover schedule in the server's bundle, before any motion
    for t in tasks:
        for v in variants:
            for mv in solve(*t.split(">"))[args.from_move - 1 :]:
                try:
                    goal = post(args.server, "/goal", transition=mv, **VARIANTS[v])
                except RuntimeError as e:  # e.g. a bundle without this move (a one-move bundle such as efps1)
                    p.error(f"{t} move {mv}: {e}; the server needs a bundle with every H15 move (MODEL=efps1h15)")
                if goal.get("options") != VARIANTS[v]:
                    p.error(f"server options {goal.get('options')} != requested {VARIANTS[v]} ({mv})")
                if not goal.get("hover_schedule") or len(goal["hover_schedule"]) != goal["n_goals"]:
                    p.error(f"the server has no hover schedule for {mv}: re-export the bundle")
    print(f"checked: the server has goals and a schedule for every move of {tasks} ({variants})")

    robot = Robot(args)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(args.out, f"session_{started.replace(':', '')}.json"), "w") as f:
        json.dump({"started": started, "args": vars(args), "server": health, **git_state(),
                   "robot_config": robot.config}, f, indent=1)  # fmt: skip

    def q(prompt, choices=None, default=None, dry=None):
        return ask(prompt, choices, default, dry if args.yes else None)

    def jaw_closed():
        try:
            return robot.jaw() < OPEN_JAW_MIN
        except Exception:
            return True

    def offer_retreat():
        if q("Retreat now: below the low floor open the jaw first (a ring stays on its rod), then straight up to the "
             "start height? [y/n]: ", {"y", "n"}, dry="y") == "y":  # fmt: skip
            try:
                return None, robot.retreat()
            except BaseException as e:
                print(f"retreat failed: {type(e).__name__}: {e}")
                return f"{type(e).__name__}: {e}", None
        return "operator declined the retreat", None

    def run_move(task, variant, rep, k, mv, intervened):
        """One game move as one_move_test runs one move. Returns (result row, retreated without error)."""
        ring, src, dst = moving_ring(mv)
        run = f"{task.replace('>', '-')}_{variant}_r{rep}_m{k:02d}"
        attempt = 0
        while any(os.path.exists(os.path.join(args.out, f"{run}{f'_a{attempt}' if attempt else ''}{ext}"))
                  for ext in (".npz", ".steps.jsonl")):  # fmt: skip
            attempt += 1
        log_name = f"{run}{f'_a{attempt}' if attempt else ''}.npz"
        step_log = log_name.replace(".npz", ".steps.jsonl")
        meta = {"run": run, "log_file": log_name, "step_log_file": step_log, "task": task, "variant": variant,
                "repeat": rep, "move": k, "transition": mv, "ring": ring, "source_peg": src, "target_peg": dst,
                "after_intervention": intervened, "options": VARIANTS[variant], "dry_run": args.dry_run, "steps": 0,
                "error": None}  # fmt: skip
        log, phase, retreated, opened_from = new_log(), "set-up", False, None
        print(f"\n--- {task} move {k}/15: {mv}  (ring {ring} from peg {src} to peg {dst})")
        try:
            goal = post(args.server, "/goal", transition=mv, **VARIANTS[variant])
            if goal.get("options") != VARIANTS[variant]:
                raise RuntimeError(f"server options {goal.get('options')} != requested {VARIANTS[variant]}")
            schedule = [int(s) for s in goal["hover_schedule"]]  # the demos' timing from the hover
            if args.schedule_scale != 1.0:
                schedule = [int(np.ceil(s * args.schedule_scale - 1e-9)) for s in schedule]
            if len(schedule) != goal["n_goals"] or min(schedule) < 1:
                raise RuntimeError(f"bad schedule {schedule} for {goal['n_goals']} goals")
            switch, budget = [int(x) for x in np.cumsum(schedule)[:-1]], int(sum(schedule))
            start_pose = robot.hover_pose(src)
            meta.update({"schedule": schedule, "server_hover_schedule": goal.get("hover_schedule"),
                         "step_rows": step_rows, "start_pose": start_pose, "goal": goal["goal"],
                         "planner": goal["planner"], "box": goal.get("box")})  # fmt: skip
            print(f"goal: {goal['goal']}\nstart above peg {src} {np.round(np.asarray(start_pose[:3]) * 1e3, 1).tolist()} "
                  f"mm; schedule {schedule} = {budget} model steps")  # fmt: skip
            robot.check_camera()
            phase = "parking"
            if args.dry_run:
                robot.replay_from(*replay[mv], step_rows)
            robot.gripper("open")
            jaw = robot.jaw()
            meta["open_jaw"] = jaw
            if jaw < OPEN_JAW_MIN:
                raise RuntimeError(f"open jaw reads {jaw:.4f} < {OPEN_JAW_MIN:.4f}: the model would see a "
                                   f"half-closed gripper")  # fmt: skip
            robot.park(start_pose)
            phase = "running"
            seed = zlib.crc32(f"{run}_{attempt}".encode()) % 100000
            meta["seed"] = seed
            t0 = datetime.datetime.now()
            run_steps(robot, args.server, goal, budget, switch, log, out=lambda m: print(m, flush=True), seed=seed,
                      intent="open", step_log=os.path.join(args.out, step_log))  # fmt: skip
            meta["seconds"] = (datetime.datetime.now() - t0).total_seconds()
            phase = "retreating"
            if float(robot.pose()[2]) < robot.low_floor:
                opened_from = robot.jaw()  # read before retreat() opens the jaw: a failed lift must not lose it
            robot.retreat()
            retreated, phase = True, "done"
        except KeyboardInterrupt:
            meta["error"] = f"stopped by the operator (Ctrl-C) while {phase}"
        except Exception as e:  # arm readback, grasp, step-check, planner or network errors: stop, keep the log
            meta["error"] = f"{type(e).__name__} while {phase}: {e}"
        meta["steps"] = len(log["pose"])
        meta["final_gripper_intent"] = log["gripper_intent"][-1] if log["gripper_intent"] else None
        if meta["error"]:
            print(f"\nmove stopped after {meta['steps']} steps: {meta['error']}")
            if phase in ("parking", "running", "retreating"):
                meta["retreat_error"], again = offer_retreat()
                opened_from = again if opened_from is None else opened_from
                retreated = meta["retreat_error"] is None
        runner_released = opened_from is not None and opened_from < OPEN_JAW_MIN
        meta.update({"retreat_opened_jaw_from": opened_from, "runner_released_jaw": runner_released})
        save_log(os.path.join(args.out, log_name), log, **meta)
        if meta["error"] and phase == "set-up":
            print("nothing moved: recorded as invalid (x)")
            progress = "x"
        else:
            if runner_released:
                print(f"NOTE: the retreat opened the jaw (reading {opened_from:.4f}) below the low floor: the runner, "
                      f"not the model, released the ring; the move cannot count as a success")  # fmt: skip
            progress = q(f"How far did move {k} ({mv}, ring {ring} {src}->{dst}) get?\n"
                         + "\n".join(f"  {a}: {b}" for a, b in PROGRESS.items()) + "\n> ", set(PROGRESS), dry="0")
        valid = progress != "x"
        disturbed = None
        if progress == "5":
            disturbed = q("Did it move or knock any other ring? [y/n]: ", {"y", "n"}, dry="n")
        meta.update({"valid": valid, "progress": int(progress) if valid else None, "progress_text": PROGRESS[progress],
                     "other_rings_disturbed": None if disturbed is None else disturbed == "y",
                     "success": (valid and progress == "5" and disturbed == "n" and not runner_released
                                 and not args.dry_run),
                     "time": datetime.datetime.now().isoformat(timespec="seconds")})  # fmt: skip
        with open(results_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return meta, retreated and not meta["error"]

    def clear_for_board():
        """One leg at the start height to the start pose, 8 cm behind the rods, so the operator can set the board.
        Only after a move that retreated (the task loop ends the session otherwise); the retreat again first, a no-op
        at the start height, so below the low floor the jaw would open before any lift."""
        robot.retreat()
        if jaw_closed() and q("The jaw reads closed. Open it now (a held ring drops)? [y/n]: ", {"y", "n"},
                              dry="n") == "y":  # fmt: skip
            robot.gripper("open")
        print("clearing the rods: one leg at the start height to the start pose")
        robot.park(START_POSE)

    summary = []
    stop_session = False
    try:
        for rep in range(args.repeats):
            for task in tasks:
                for variant in variants:
                    seq = solve(*task.split(">"))
                    first = seq[args.from_move - 1].split(">")[0]
                    print(f"\n=== {task} ({variant}, repeat {rep}): {len(seq)} moves, from move {args.from_move}")
                    a = q(f"Set the board to {first}, make sure the gripper is empty. [Enter] go, [s] skip, [q] quit: ",
                          {"", "s", "q"}, dry="")  # fmt: skip
                    if a == "q":
                        stop_session = True
                        break
                    if a == "s":
                        continue
                    # strict: moves completed before the first failed move (None while there was none)
                    k, intervened, strict, attempts, cleared = args.from_move, False, None, [], False
                    while k <= len(seq):
                        mv = seq[k - 1]
                        before, after = mv.split(">")
                        meta, _ = run_move(task, variant, rep, k, mv, intervened)
                        cleared = False
                        attempts.append({"move": k, "transition": mv, "success": meta["success"],
                                         "progress": meta["progress"], "valid": meta["valid"],
                                         "log_file": meta["log_file"]})  # fmt: skip
                        if meta["valid"] and not meta["success"] and strict is None:
                            strict = k - args.from_move
                        if meta.get("retreat_error"):  # retreat declined or failed: as one_move_test, no motion
                            # here; the session's exit retreat (jaw open first below the low floor, PARK if it fails)
                            print("the arm did not retreat: ending the session; continue later with --append "
                                  f"--from-move {k if not meta['success'] else k + 1} (and that move's start board)")
                            stop_session = True
                            break
                        if meta["success"]:
                            k += 1
                            if meta["error"] and q(f"Move {k - 1} succeeded, but its run stopped with an error. Continue "
                                                   f"with the next move? [y/n]: ", {"y", "n"}, dry="y") != "y":
                                break  # fmt: skip
                            continue
                        if meta["valid"]:
                            what = q(f"Move {k} did not succeed. [c] set the board by hand to {after} and continue, "
                                     f"[r] set it back to {before} and redo move {k} (either way the arm first moves to "
                                     f"the start pose behind the rods), [q] end this task: ",
                                     {"c", "r", "q"}, dry="c")  # fmt: skip
                        else:  # an invalid attempt (set-up, network or server problem) is redone, not a failure
                            what = q(f"Invalid attempt. [r] redo move {k} (the arm first moves to the start pose; the "
                                     f"board back to {before} if anything moved), [q] end this task: ", {"r", "q"},
                                     dry="q")  # fmt: skip
                        if what == "q":
                            break
                        try:
                            clear_for_board()
                            cleared = True
                        except BaseException as e:  # Ctrl-C included: the arm's state is uncertain, end the session
                            print(f"move to the start pose failed: {type(e).__name__}: {e}")
                            stop_session = True
                            break
                        board = after if what == "c" else before
                        if q(f"Set the board to {board} now, make sure the gripper is empty. [Enter] when done, [q] "
                             f"end this task: ", {"", "q"}, dry="") == "q":  # fmt: skip
                            break
                        if meta["valid"]:
                            intervened = True  # the operator corrected the board after a failed move
                        k = k + 1 if what == "c" else k
                    done = strict if strict is not None else (k - args.from_move)
                    row = {"task": task, "variant": variant, "repeat": rep, "from_move": args.from_move,
                           "moves_before_first_failure": done,
                           "strict_success": (args.from_move == 1 and strict is None and k > len(seq)
                                              and not args.dry_run),
                           "interventions": intervened, "attempts": attempts}  # fmt: skip
                    summary.append(row)
                    with open(os.path.join(args.out, "tasks.jsonl"), "a") as f:
                        f.write(json.dumps(row | {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                                                  "dry_run": args.dry_run}) + "\n")  # fmt: skip
                    if not stop_session and not cleared:
                        try:
                            clear_for_board()
                        except BaseException as e:  # Ctrl-C included: end the session (its exit retreat)
                            print(f"move to the start pose failed: {type(e).__name__}: {e}")
                            stop_session = True
                    if stop_session:
                        break
                if stop_session:
                    break
            if stop_session:
                break
    finally:
        try:
            if not args.yes and jaw_closed() and q("Session over and the jaw reads closed. Open it before parking? "
                                                   "[y/n]: ", {"y", "n"}) == "y":  # fmt: skip
                robot.gripper("open")
        finally:
            robot.close(ask=ask)
    rows = [json.loads(x) for x in open(results_path)] if os.path.exists(results_path) else []
    valid = [r for r in rows if r.get("valid", True)]
    print(f"\nper-move results ({results_path}, valid attempts): success {sum(r['success'] for r in valid)}/"
          f"{len(valid)}")  # fmt: skip
    for s in summary:
        print(f"  {s['task']} {s['variant']} r{s['repeat']}: {s['moves_before_first_failure']} moves before the first "
              f"failure, strict success {s['strict_success']}, interventions {s['interventions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
