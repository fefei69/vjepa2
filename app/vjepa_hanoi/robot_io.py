# Robot I/O for V-JEPA 2-AC clients: the real Trossen arm + RealSense through tower_hanoi, or a dry run.
#
# Only stdlib + numpy; tower_hanoi's modules (robot.arm, robot.camera, robot.geometry, robot.wm_dataset) are imported
# from the robot repository (put it on PYTHONPATH), so the hardware path is the one the data collector uses. The dry
# run uses tower_hanoi's DryRunArm and replays a recorded walk as the camera (no ROS).

import json
import os

STEP_S = 8 / 30.0  # one model step (fps 4 on 30 Hz)
CLOSE_S, OPEN_S = 1.2, 1.0  # gripper timing of the recordings


class ReplayCamera:
    """Dry run: serves recorded 224 px frames of one walk in order (stands in for camera + transform)."""

    def __init__(self, path, episode=0):
        import h5py

        self.f = h5py.File(path, "r")
        self.row = int(self.f["ep_offset"][episode])

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
            self.arm = arm_module.TrossenArm(self.workspace, args.address)
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

    def gripper(self, intent):
        if intent == "open":
            self.arm.open_gripper(self.workspace.open_command, OPEN_S)
        else:
            self.arm.close_gripper(self.workspace.grip_force, CLOSE_S)

    def close(self):
        if not self.dry:
            self.camera.stop()
        self.arm.disconnect()
