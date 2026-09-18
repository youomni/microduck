import numpy as np

class DuckEnv:
    """
    Simple toy environment for training a 'duck agent'.
    State: [x_position, energy]
    Actions: 0=walk, 1=sit, 2=stand, 3=stop
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0.0
        self.energy = 1.0
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.x, self.energy], dtype=np.float32)

    def step(self, action):
        self.t += 1
        reward = 0.0
        done = False

        # decay energy over time
        self.energy -= 0.01

        if action == 0:  # walk
            if self.energy > 0.1:
                self.x += 0.1
                self.energy -= 0.05
                reward += 1.0
            else:
                reward -= 1.0

        elif action == 1:  # sit (rest)
            self.energy = min(1.0, self.energy + 0.08)
            reward += 0.2

        elif action == 2:  # stand
            reward += 0.1

        elif action == 3:  # stop
            reward += 0.0

        # terminal condition
        if self.energy <= 0 or self.t >= 200:
            done = True

        return self._get_obs(), reward, done, {}
