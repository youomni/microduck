import numpy as np

class Env:
    """
    SKILL SEQUENCE:

    1) crouch
    2) jump
    3) rotate in air (30° target)
    4) land WITH crouch (important)
    5) stabilize (stand straight)
    """

    def __init__(self):
        self.state = None

        self.target_angle = 30.0

        self.phase = None
        self.t = 0

    def reset(self):
        # [x, y, angle, posture]
        # posture: 0 = full crouch, 1 = fully straight
        self.state = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self.phase = "crouch"
        self.t = 0

        return self._get_obs()

    def step(self, action):
        """
        action = [dx, dy, d_angle, d_posture]
        """

        x, y, angle, posture = self.state
        dx, dy, d_angle, d_posture = action

        self.t += 1

        # update state
        x += dx
        y += dy
        angle += d_angle
        posture = float(np.clip(posture + d_posture, 0.0, 1.0))

        self.state = np.array([x, y, angle, posture], dtype=np.float32)

        reward = 0.0
        done = False

        # -------------------------
        # 1) CROUCH
        # -------------------------
        if self.phase == "crouch":
            reward += (1.0 - posture)  # reward lowering body

            if posture < 0.2:
                self.phase = "jump"

        # -------------------------
        # 2) JUMP
        # -------------------------
        elif self.phase == "jump":
            reward += y  # reward lift-off

            if y > 1.0:
                self.phase = "air"

        # -------------------------
        # 3) AIR ROTATION (30°)
        # -------------------------
        elif self.phase == "air":
            angle_error = abs(angle - self.target_angle)

            reward += -angle_error

            # transition to landing when falling
            if y <= 0.3:
                self.phase = "land"

        # -------------------------
        # 4) LAND WITH CROUCH (CRITICAL)
        # -------------------------
        elif self.phase == "land":
            # IMPORTANT: crouch must already be present at contact
            crouch_bonus = max(0.0, 0.6 - posture)  # must be bent knees

            stability_penalty = abs(y) + abs(x)

            reward += crouch_bonus - stability_penalty

            # enforce requirement: landing is ONLY valid if crouched
            if y <= 0.0 and posture < 0.5:
                self.phase = "stabilize"
            elif y <= 0.0 and posture >= 0.5:
                reward -= 5.0  # hard penalty for stiff landing

        # -------------------------
        # 5) STABILIZE (STAND STRAIGHT)
        # -------------------------
        elif self.phase == "stabilize":
            reward += posture  # reward standing straight

            if abs(posture - 1.0) < 0.05 and abs(x) < 0.05:
                done = True

        # global stability penalty
        reward -= 0.1 * (abs(x) + abs(y))

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.state.copy()
