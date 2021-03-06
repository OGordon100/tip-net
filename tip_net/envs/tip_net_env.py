import sys

import gym
import numpy as np
from gym import spaces
from nOmicron.microscope import IO
from nOmicron.microscope.conditioning import tip_pulse, tip_crash

try:
    from MicroscopeFuncs.coarse_improvements import ImprovementController
    from NetworkFuncs.make_coach_CSV import CoachCSV
except ImportError:
    for strng in ['/home/mltest1/tmp/pycharm_project_510', '../Autonomous-Probes']:
        try:
            sys.path.insert(1, strng)
            break
        except ImportError:
            pass


class TipNet(gym.Env):
    metadata = {'render.modes': ['human']}
    np.seterr(divide='ignore', invalid='ignore')

    def __init__(self, target_state=None):
        self.controller_name = "TipNet"
        IO.connect()
        self.scan_bias_voltage = -3.5

        self.improver = ImprovementController(person_name=self.controller_name)
        self.improver.saveloc = "/home/mltest1/tmp/pycharm_project_510/Data/ImprovementSessions2/"
        self.improver.reset(destroy_tip=False)

        self.observation_space = spaces.Box(low=0, high=1, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Discrete(24)

        self.target_state = target_state

    def step(self, action):
        self._take_action(action)
        self.improver.step()
        obs = self.improver.raw_log.tail(1)

        state_feature_obs = 1  ##### TODO
        reward = CoachCSV(obs)

        # Done if last 20 lines are what we want
        if self.improver.raw_log.tail(20) is True:  #####TODO
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def _take_action(self, action):

        action = np.argmax(action)

        if action == 0:  # do nothing
            pass
        elif action == 1:
            self.improver.scan_pos.big_move_and_reset_size()
        elif action == 2:
            self.improver.scan_pos.reduce_scan_size(scandata=self.improver.scan_prog.scan_matrix_raw[0, :, :])
        elif 3 <= action <= 13:
            intensities = np.arange(-10, 12, 2)
            pulse_intensity = intensities[action]
            tip_pulse(voltage=pulse_intensity, time=200e-3)
        elif action <= 23:
            distances = np.arange(0.2, 2.2, 0.2)
            press_distance = distances[action]
            tip_crash(press_distance)
        else:
            raise RuntimeError("Output action is too high!!!!")

    def reset(self):
        self.improver.reset()
        obs = self.improver.raw_log.tail(1)

        return obs

    def render(self, mode='human', close=False):
        self.improver.render()
