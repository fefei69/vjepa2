# Robot I/O for V-JEPA 2-AC clients: the real Trossen arm + RealSense through tower_hanoi, or a dry run.
#
# Only stdlib + numpy; tower_hanoi's modules (robot.arm, robot.camera, robot.geometry, robot.wm_dataset) are imported
# from the robot repository (put it on PYTHONPATH), so the hardware path is the one the data collector uses. The dry
# run uses tower_hanoi's DryRunArm and replays a recorded walk as the camera (no ROS).

import json
import os

STEP_S = 8 / 30.0  # one model step (fps 4 on 30 Hz)
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
        self.config = {"workspace_json": os.path.join(config_dir, "workspace.json"),
                       "open_command": self.workspace.open_command, "grip_force": self.workspace.grip_force}
        self.arm.connect()

    def observe(self):
        if self.dry:
            pose, jaw = self.replay.pose_and_jaw()
            return self.replay.image(), pose, jaw
        frame = list(self.camera.fresh_frames(1))[-1]
        pose = tuple(float(v) for v in self.arm.driver.get_cartesian_positions())[:6]
        return self.transform.apply(frame), pose, float(self.arm.gripper_position())

    def move(self, pose, goal_time):
        self.arm.move_to(tuple(pose), goal_time)

    def replay_from(self, path, start_row):
        """Dry run: replay `path` from raw row `start_row` (e.g. the start of a held-out test move)."""
        self.replay = ReplayCamera(path, start_row=start_row)

    def pose(self):
        if self.dry:
            return self.replay.pose_and_jaw()[0]
        return tuple(float(v) for v in self.arm.driver.get_cartesian_positions())[:6]

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
        if intent == "open":
            self.arm.open_gripper(self.workspace.open_command, OPEN_S)
        else:
            self.arm.close_gripper(self.workspace.grip_force, CLOSE_S)

    def jaw(self):
        return float(self.replay.pose_and_jaw()[1]) if self.dry else float(self.arm.gripper_position())

    def close(self):
        """Lift straight up first (best effort), then tower_hanoi's disconnect: joint-space park at zero, then idle."""
        try:
            self.lift(START_POSE[2])
        except Exception as e:
            print(f"[robot] could not lift before parking ({type(e).__name__}: {e})")
        if not self.dry:
            self.camera.stop()
        self.arm.disconnect()
