import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
import math

from samples import Factory
from src.race import ActionCarState, Race
from src.jsoner import *

class RaceTest(unittest.TestCase):

    def test_000_sample_0_run(self):
        print('\n===\ntest_000_sample_0_run()')
        race = Factory.sample_race_0()
        # print('track:\r', race.race_info.track_info)

        race.run(debug=True)


    def test_001_sample_0_check(self):
        print('\n===\ntest_001_sample_0_check()')
        race = Factory.sample_race_0()
        race.race_info.start_state = CarState(position = Point2D(y = 5.5, x = 14.5), wheel_angle=3.14)
        print('track:\r', race.race_info.track_info)
        self.assertTrue(race.race_info.track_info.round_distance == 28)

        race.run(debug=True)
        self.assertTrue(race.steps[-1].car_state.round_count == -1)
        self.assertTrue(race.steps[-1].car_state.track_state.last_road_tile_total_distance < 0)


    def test_200_too_low_power(self):
        print('\n===\ntest_400_too_low_power')

        race = Factory.sample_race_0()
        
        low_power_action = car.Action(1,0)
        print('low_power_action = ', low_power_action)
        state_1 = race.track_field.get_next_state(
            car_config=race.race_info.car_config, 
            car_state=race.race_info.start_state, 
            action=low_power_action)
        print('state_1 = ', state_1)
        self.assertTrue(state_1.velocity_x == 0)
        self.assertTrue(state_1.velocity_y == 0)
        self.assertTrue(state_1.wheel_angle == 0)
        self.assertTrue(state_1.timestamp == race.race_info.track_info.time_interval)

if __name__ == '__main__':
    unittest.main()

