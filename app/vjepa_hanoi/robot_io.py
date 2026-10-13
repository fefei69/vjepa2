# Robot I/O for V-JEPA 2-AC clients: the real Trossen arm + RealSense through tower_hanoi, or a dry run.
#
# Only stdlib + numpy; tower_hanoi's modules (robot.arm, robot.camera, robot.geometry, robot.wm_dataset) are imported
# from the robot repository (put it on PYTHONPATH), so the hardware path is the one the data collector uses. The dry
# run uses tower_hanoi's DryRunArm and replays a recorded walk as the camera (no ROS).
#
# How a planned step is executed follows tower_hanoi's Q-WM runner (scripts/run_qwm_h15_live.py), proven on this arm
# (docs/handover-hardware-test-pipeline.md). It changes how the arm reaches the planned target, never the target:
#   - the move's goal time is Q-WM's rule (step_goal_time) instead of the data's 0.267 s: 0.5-1.0 s at up to 36 mm/s,
#     1.0 s for a jaw change, the 2 steps after one and a loaded step below the low floor;
#   - the 3 mm / 2 deg readback re-reads the pose for up to 0.15 s before it fails (same tolerances);
#   - the next frame is one received after the arm is still, with the pose and jaw read then;
#   - abort-only checks, never steering: a missed grasp after a close below the low floor (tower_hanoi check_grip;
#     higher up a close can hold nothing and is only reported) and no sideways step over 3 mm with an end below the
#     low floor (Q-WM's low-lateral guard, without its recovery);
#   - retreat (end of a run, an abort, the session end): below the low floor open the jaw first, so a ring stays on its
#     rod, then straight up; the joint-space park runs only after a successful retreat or when the operator types PARK.

import json
import math
import os
import time

STEP_S = 8 / 30.0  # one model step (fps 4 on 30 Hz): the planning unit and the schedule's clock, not the move time
CLOSE_S, OPEN_S = 1.2, 1.0  # gripper timing of the recordings
# The commanded pose at the start of every six-task expert episode (reference_pose at each ep_offset, identical in all
# 60): the first move of the one-move test starts here. It is taken from the recordings rather than recomputed from
# workspace.json, whose current peg heights put collect_wm.start_pose 9 mm higher.
START_POSE = (0.41400, 0.01574, 0.19113, 0.0, 0.7854, 0.0)
PARK_S = 2.304  # workspace.json goal_time: the commissioned duration of one Cartesian leg
# The open gripper command of the recordings (execution_config_json: open_fraction 0.85 x max_stroke 0.04). The model's
# gripper state is closedness = (0.0340 - jaw) / 0.0031 (hanoi.GRIPPER_OPEN/CLOSED), a 3.1 mm band, so an open jaw at
# another width reads as half closed. The committed tower_hanoi workspace.json says open_fraction 0.80 (0.032): the
# real arm refuses to start unless the robot host's config opens to the recorded width.
RECORDED_OPEN_COMMAND = 0.034
RECORDED_TRANSFORM = ("square_roi", 151, 90, 360, 224)  # wm-transform.json of the recordings
STEP_FAST_S, STEP_SLOW_S = 0.5, 1.0  # Q-WM campaign STEP_TIME_FAST / STEP_TIME (run_qwm_campaign.py)
SPEED_CAP_M_S = 0.036  # Q-WM SPEED_CAP_M_S: the fastest commanded speed proven at 1.0 s
SETTLE_TIMEOUT_S = 0.15  # Q-WM campaign SETTLE: TrossenArm.move_to's readback re-read window
LOW_FLOOR_BELOW_TRANSIT_M = 0.025  # Q-WM LOW_FLOOR_M = transit z - 25 mm (166.13 mm today)
LOW_LATERAL_LIMIT_M = 0.003  # Q-WM LOW_LATERAL_GUARD_MM; expert first-move steps below the floor stay under 1.3 mm
STILL_M_S, STILL_SAMPLES, STILL_POLL_S, STILL_TIMEOUT_S = 0.02, 3, 0.01, 5.0  # the EC / Q-WM stationary wait


class UnsafeStep(RuntimeError):
    """An abort-only check refused a planned step before any of its motion."""


def step_goal_time(distance_m, jaw_change, steps_since_gripper, loaded_low):
    """Goal time of one step's Cartesian move: Q-WM's waypoint_goal_time rule (run_qwm_h15_live.py). Returns
    (seconds, class). steps_since_gripper counts the steps since the one with the last gripper command (None: none)."""
    if jaw_change:
        return STEP_SLOW_S, "jaw change"
    if steps_since_gripper is not None and steps_since_gripper < 2:
        return STEP_SLOW_S, "after a gripper change"
    if loaded_low:
        return STEP_SLOW_S, "loaded, low"
    return min(max(distance_m / SPEED_CAP_M_S, STEP_FAST_S), STEP_SLOW_S), "fast"


class ReplayCamera:
    """Dry run: serves recorded 224 px frames of one walk in order (stands in for camera + transform)."""

    def __init__(self, path, episode=0, start_row=None):
        import h5py

        self.f = h5py.File(path, "r")
        self.row = int(self.f["ep_offset"][episode]) if start_row is None else int(start_row)

    def image(self):
        img = self.f["pixels"][self.row]
        self.row += 8
        return img

    def pose_and_jaw(self):
        return self.f["state"][self.row].astype(float), float(self.f["proprio"][self.row, 6])


class Robot:
    """Real hardware through tower_hanoi, or DryRunArm + ReplayCamera."""

    def __init__(self, args):
        from robot import arm as arm_module
        from robot.geometry import Workspace

        config_dir = os.path.join(args.tower_hanoi, "robot", "config")
        with open(os.path.join(config_dir, "workspace.json")) as fh:
            self.workspace = Workspace(json.load(fh))
        self.dry = args.dry_run
        if self.dry:
            self.arm = arm_module.DryRunArm(self.workspace)
            self.replay = ReplayCamera(args.replay, args.replay_episode)
        else:
            from robot.camera import RosCamera
            from robot.wm_dataset import ImageTransformProfile

            with open(os.path.join(config_dir, "camera.json")) as fh:
                cam = json.load(fh)
            import rclpy
            from rclpy.signals import SignalHandlerOptions

            # rclpy's own SIGINT handler would shut the ROS context down on any Ctrl-C (also at a prompt), which kills
            # the camera for the rest of the process. Without it, Ctrl-C is a plain KeyboardInterrupt in the runner.
            self._owns_rclpy = not rclpy.ok()
            if self._owns_rclpy:
                rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
            self.camera = RosCamera(cam["topic"], cam["width"], cam["height"], cam["max_age_s"])
            self.camera.start()
            self.transform = ImageTransformProfile.load(os.path.join(config_dir, "wm-transform.json"))
            t = self.transform
            if (t.mode, t.x, t.y, t.size, t.output_size) != RECORDED_TRANSFORM:
                raise RuntimeError(f"wm-transform.json is {t.describe()}, the recordings used {RECORDED_TRANSFORM}")
            if abs(self.workspace.open_command - RECORDED_OPEN_COMMAND) > 5e-4:
                raise RuntimeError(
                    f"{config_dir}/workspace.json opens the gripper to {self.workspace.open_command:.4f} (open_fraction "
                    f"{self.workspace.open_fraction}); the recordings and the model use {RECORDED_OPEN_COMMAND} "
                    f"(open_fraction 0.85). Use the robot config the recordings were made with."
                )
            self.arm = arm_module.TrossenArm(self.workspace, args.address)
        self.check_grip = arm_module.check_grip
        self.low_floor = float(self.workspace.transit_z) - LOW_FLOOR_BELOW_TRANSIT_M
        self.config = {"workspace_json": os.path.join(config_dir, "workspace.json"),
                       "open_command": self.workspace.open_command, "grip_force": self.workspace.grip_force,
                       "low_floor": self.low_floor}  # fmt: skip
        self.arm.connect()

    def observe(self):
        """The planner's input: a frame received after the arm came to rest, and the pose and jaw read right after."""
        if self.dry:
            pose, jaw = self.replay.pose_and_jaw()
            return self.replay.image(), pose, jaw
        self.wait_still()
        frame = self.camera.fresh_snapshot(not_before_receipt_monotonic_ns=time.monotonic_ns()).rgb
        pose = tuple(float(v) for v in self.arm.driver.get_cartesian_positions())[:6]
        return self.transform.apply(frame), pose, float(self.arm.gripper_position())

    def wait_still(self):
        """Block until the tool's linear speed reads <= STILL_M_S on STILL_SAMPLES consecutive polls (the EC / Q-WM
        stationary wait); raise if it does not within STILL_TIMEOUT_S."""
        deadline, still, top = time.monotonic() + STILL_TIMEOUT_S, 0, 0.0
        while time.monotonic() <= deadline:
            speed = math.hypot(*(float(v) for v in list(self.arm.driver.get_cartesian_velocities())[:3]))
            top = max(top, speed)
            still = still + 1 if speed <= STILL_M_S else 0
            if still >= STILL_SAMPLES:
                return
            time.sleep(STILL_POLL_S)
        raise RuntimeError(f"the arm did not come to rest (<= {STILL_M_S} m/s) within {STILL_TIMEOUT_S} s; "
                           f"fastest reading {top:.3f} m/s")  # fmt: skip

    def check_camera(self):
        """Before a run moves anything: a frame received from now on, or raise (e.g. the ROS camera was shut down by a
        Ctrl-C at a prompt; restart the runner with --resume)."""
        if not self.dry:
            self.camera.fresh_snapshot(not_before_receipt_monotonic_ns=time.monotonic_ns(), timeout_s=3.0)

    def check_step(self, pose, target):
        """Abort-only, before any motion of a step (Q-WM's low-lateral guard, without its recovery): no sideways move
        over LOW_LATERAL_LIMIT_M with either end below the low floor, where a sideways step can strike a rod."""
        low = min(float(pose[2]), float(target[2])) < self.low_floor
        lateral = math.dist([float(v) for v in pose[:2]], [float(v) for v in target[:2]])
        if low and lateral > LOW_LATERAL_LIMIT_M:
            raise UnsafeStep(f"planned step moves {lateral * 1e3:.1f} mm sideways with an end below the low floor "
                             f"({self.low_floor * 1e3:.1f} mm; z {pose[2] * 1e3:.1f} -> {target[2] * 1e3:.1f} mm); "
                             f"limit {LOW_LATERAL_LIMIT_M * 1e3:.0f} mm")  # fmt: skip

    def move(self, pose, goal_time, settle=False):
        """Blocking Cartesian move with tower_hanoi's 3 mm / 2 deg readback. settle (model steps): re-read the pose for
        up to SETTLE_TIMEOUT_S before the check fails; the dry run's arm has no readback."""
        if settle and not self.dry:
            self.arm.move_to(tuple(pose), goal_time, settle_timeout_s=SETTLE_TIMEOUT_S)
        else:
            self.arm.move_to(tuple(pose), goal_time)

    def replay_from(self, path, start_row):
        """Dry run: replay `path` from raw row `start_row` (e.g. the start of a held-out test move)."""
        self.replay = ReplayCamera(path, start_row=start_row)

    def pose(self):
        if self.dry:
            return self.replay.pose_and_jaw()[0]
        return tuple(float(v) for v in self.arm.driver.get_cartesian_positions())[:6]

    def hover_pose(self, peg):
        """Q-WM's trial start: above the source peg's calibrated grasp xy (tower_hanoi grasp_position) at the start
        height, recorded orientation. The one-move demos pass it at the end of their approach (within 0.4 mm), jaw
        open and nearly still, 13-16 model steps after the episode start pose."""
        from robot.geometry import grasp_position

        x, y = grasp_position(self.workspace, peg)
        return (float(x), float(y), START_POSE[2]) + tuple(START_POSE[3:])

    def lift(self, z, goal_time=PARK_S):
        """Straight up (pure Z) to height z at the current x, y; never down. Clears the rods before any travel."""
        pose = list(self.pose())
        if pose[2] < z:
            pose[2] = z
            self.move(pose, goal_time)

    def park(self, pose=START_POSE, goal_time=PARK_S):
        """Up to the start height first, then one Cartesian leg to `pose` at that height (above every rod)."""
        self.lift(pose[2], goal_time)
        self.move(pose, goal_time)

    def gripper(self, intent):
        """Open, or close by force. A close below the low floor, where a ring or a rod can be between the pads, is then
        checked: tower_hanoi's check_grip raises ArmError when the jaws nearly closed (miss threshold 0.2 x max_stroke)
        before the arm moves on. Higher up a missed close is harmless (the planner closing in mid-air, which it may
        undo): the reading is only reported. Returns the jaw reading after a close, None after an open."""
        if intent == "open":
            self.arm.open_gripper(self.workspace.open_command, OPEN_S)
            return None
        self.arm.close_gripper(self.workspace.grip_force, CLOSE_S)
        z = float(self.pose()[2])
        if z < self.low_floor:
            self.check_grip(self.arm)
        else:
            print(f"[robot] close at z {z * 1e3:.1f} mm, above the low floor: grasp reading {self.jaw():.4f} "
                  f"(not checked)")  # fmt: skip
        return self.jaw()

    def jaw(self):
        return float(self.replay.pose_and_jaw()[1]) if self.dry else float(self.arm.gripper_position())

    def retreat(self):
        """Q-WM's safe retreat (tower_hanoi scripts/run_graph_cidm_v2_aaaa_cccc_live.py::safe_retreat): below the low
        floor, where a ring or the pads can be on a rod, open the jaw first so a ring stays captive on its rod; then
        straight up to the start height. Above the floor a held ring is clear of the rods and stays held. Raises if a
        command fails. Returns the jaw reading from just before it opened the jaw, or None if it did not open it."""
        z, opened_from = float(self.pose()[2]), None
        if z < self.low_floor:
            opened_from = self.jaw()
            print(f"[robot] tool at z {z * 1e3:.1f} mm, below {self.low_floor * 1e3:.1f} mm: opening the jaw (reading "
                  f"{opened_from:.4f}), then straight up")  # fmt: skip
            self.gripper("open")
        self.lift(START_POSE[2])
        return opened_from

    def close(self, ask=None):
        """End of session: the retreat, then tower_hanoi's disconnect (joint-space park at zero, then idle). If the
        retreat fails or is interrupted, the park could drag a ring or a rod, so the operator chooses: PARK, or keep
        the arm holding its pose in position mode until they have recovered it (the process does not exit before, as
        the controller goes idle when the connection drops). The camera stop can never skip the disconnect."""
        park = True
        try:
            self.retreat()
        except BaseException as e:  # Ctrl-C included: an interrupted retreat is a failed one
            print(f"[robot] retreat FAILED ({type(e).__name__}: {e}); the arm is holding its pose in position mode")
            if not self.dry:
                try:
                    reply = (ask or input)("Type PARK to run the joint-space park anyway (no lift first). Any other "
                                           "reply leaves the arm holding: ")  # fmt: skip
                except BaseException:
                    reply = ""
                park = reply.strip().upper() == "PARK"
        try:
            if not self.dry:
                self.camera.stop()
                if getattr(self, "_owns_rclpy", False):
                    import rclpy

                    if rclpy.ok():
                        rclpy.shutdown()
        except Exception as e:
            print(f"[robot] camera stop failed ({type(e).__name__}: {e})")
        finally:
            if park:
                self.arm.disconnect()
            else:
                self.hold_for_operator(ask)

    def hold_for_operator(self, ask=None):
        """Q-WM's hold window after a declined park: do not let the process exit (trossen_arm >= 1.8.0 returns the
        controller to idle when the connection drops, and idle lets the arm sag) until the operator has recovered the
        arm, or types PARK."""
        print("[robot] NOT parked: the arm holds its pose in position mode.")
        while True:
            try:
                reply = (ask or input)("Recover the arm by hand or with the e-stop, then press Enter to exit (the "
                                       "controller goes idle when this process exits), or type PARK to park now: ")
            except EOFError:
                return
            except KeyboardInterrupt:
                continue
            if reply.strip().upper() == "PARK":
                self.arm.disconnect()
            return
