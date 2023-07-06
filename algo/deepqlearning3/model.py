import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.src import model, car
from core.src.race import *
from core.test.samples import Factory

INPUT_VECTOR_SIZE = 14

action_step = 1 # number of steps for [0, 1]
action_step_count = 2*action_step+1 # total number of options [-1, 1]

OUTPUT_VECTOR_SIZE = action_step_count*action_step_count


device = "cpu"

DATA_FILE_NAME = "policy_net.pt"

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        HideLayer1Size = 32
        self.layer1 = nn.Linear(INPUT_VECTOR_SIZE, HideLayer1Size)
        HideLayer2Size = 32
        self.layer2 = nn.Linear(HideLayer1Size, HideLayer2Size)

        self.layer3 = nn.Linear(HideLayer2Size, OUTPUT_VECTOR_SIZE)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        #x = F.relu(self.layer3(x))
        return self.layer3(x)


class Wheel:
    def __init__(self, keys:list[float], weights:list[float]):
        self.wheel = []
        total_weights = sum(weights)
        top = 0
        for key, weight in zip(keys, weights):
            f = weight/total_weights
            self.wheel.append((top, top+f, key))
            top += f

    def binary_search(self, min_index:int, max_index:int, number:float) -> float:
        mid_index = (min_index + max_index)//2
        low, high, key = self.wheel[mid_index]
        if low<=number<=high:
            return key
        elif high < number:
            return self.binary_search(mid_index+1, max_index, number)
        else:
            return self.binary_search(min_index, mid_index-1, number)
        
    def spin(self) -> int:
        return self.binary_search(0, len(self.wheel) - 1, random.random())


# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000


steps_done = 0
class Model(model.IModelInference):

    def __init__(self, max_acceleration:float = 1, max_angular_velocity:float = 1, is_train:bool = False):
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.policy_net = DQN().to(device)
        self.is_train = is_train

        self.acceleration_wheel = Wheel([-1, 0, 1], [1, 2, 5])
        self.angular_wheel = Wheel([-1, 0, 1], [1, 2, 3])

    def load(self, folder:str) -> bool:
        loaded = False
        try:
            model_path = os.path.join(folder, DATA_FILE_NAME)
            if os.path.exists(model_path):
                self.policy_net.load_state_dict(torch.load(model_path))
                loaded = True
        except:
            print(f"Failed to load q_learning model from {model_path}")
    
        return loaded

    @classmethod
    def state_tensor(cls, car_state: car.CarState) -> torch.tensor:
        
        input = np.empty((INPUT_VECTOR_SIZE), dtype=np.float32)
        input[0] = car_state.position.x
        input[1] = car_state.position.y
        input[2] = car_state.wheel_angle
        input[3] = car_state.track_state.velocity_forward
        input[4] = car_state.track_state.velocity_right
        input[5:14] = car_state.track_state.rays[0:9]
        input = torch.FloatTensor(input).reshape((1, 14)).to(device)

        return input

    def action_tensor(self, action: car.Action) -> torch.tensor:
        acceleration_index = round(action.forward_acceleration/self.max_acceleration) * action_step + action_step
        angular_velocity_index = round(action.angular_velocity/self.max_angular_velocity) * action_step + action_step
        action_index =  acceleration_index*action_step_count + angular_velocity_index
        action_tensor = torch.tensor([[action_index]], device=device)
        return action_tensor
    
    def select_action(self, state_tensor) ->int:
        global steps_done
        sample = random.random()
        eps_threshold = (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
        steps_done += 1

        if sample > eps_threshold or not self.is_train:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                net_output = self.policy_net(state_tensor)
                max = net_output.max(1)
                top = max[1]
                action_index = int(top[0])
        else:
            acceleration_value = self.acceleration_wheel.spin()
            acceleration_index = acceleration_value * action_step + action_step
            angular_velocity_value = self.angular_wheel.spin()
            angular_velocity_index = angular_velocity_value * action_step + action_step
            action_index = acceleration_index * action_step_count + angular_velocity_index

        return action_index
        
    
    def get_action(self, car_state: car.CarState) -> car.Action:

        input = self.state_tensor(car_state)
        action_index = self.select_action(input)

        acceleration_index = action_index // action_step_count
        angular_velocity_index = action_index % action_step_count
        acceleration = self.max_acceleration*(acceleration_index-action_step)/action_step
        angular_velocity = self.max_angular_velocity*(angular_velocity_index-action_step)/action_step

        car_acion = car.Action(acceleration, angular_velocity)
        # print('A:', acceleration_index-action_step, angular_velocity_index-action_step)
        return car_acion



def load_model(car_config: car.CarConfig):
  
    model = Model(car_config.motion_profile.max_acceleration, 
        car_config.motion_profile.max_angular_velocity)
    loaded = model.load(os.path.dirname(__file__))
    # print('Model load from data=', loaded)
    if not loaded:
        model.init_data()

    model_info = ModelInfo(name='dqn-hc', version='2023.5.27')

    return model, model_info


if __name__ == '__main__':

    race = Factory.sample_race_sshape()
    model, model_info = load_model(race.race_info.car_config)

    race.model = model
    race.race_info.model_info = model_info

    start_state = race.race_info.start_state
    race.track_field.calc_track_state(start_state)
    print('start_state:\n', start_state)

    action = model.get_action(start_state)
    print('action at start:\n', action)

    race.run(debug=False)

    final_state = race.steps[-1].car_state
    print('race_info:\n', race.race_info)
    print('finish:\n', final_state)

    for i in range(len(race.steps)):
        step = race.steps[i]
        if step.action != None:
            print(i
                  , f'action({step.action.forward_acceleration:.2f}, {step.action.angular_velocity:.2f})'
                  , step.car_state.track_state.tile_total_distance, step.car_state.track_state.score
                  , f'(x={step.car_state.position.x:.2f}, y={step.car_state.position.y:.2f})'
                  , f'(head={step.car_state.wheel_angle:.2f}, v_forward={step.car_state.track_state.velocity_forward:.2f}, v_right={step.car_state.track_state.velocity_right:.2f})'
                  )
    
