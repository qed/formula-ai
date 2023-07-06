import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from core.src import model, car
from core.src.race import *
from core.test.samples import Factory

INPUT_VECTOR_SIZE = 14
OUTPUT_VECTOR_SIZE = 2

device = "cpu"

ACTOR_FILE_NAME = "actor.pt"
CRITIC_FILE_NAME = "critic.pt"


class Actor(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(Actor, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, x):
		activation1 = F.relu(self.layer1(x))
		activation2 = F.relu(self.layer2(activation1))
		output = F.tanh(self.layer3(activation2))

		return output

class Critic(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(Critic, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, x):
		activation1 = F.relu(self.layer1(x))
		activation2 = F.relu(self.layer2(activation1))

		return self.layer3(activation2)

class Model(model.IModelInference):

    def __init__(self, max_acceleration:float = 1, max_angular_velocity:float = 1, is_train:bool = False):
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.actor = Actor(INPUT_VECTOR_SIZE, OUTPUT_VECTOR_SIZE).to(device)

    def load(self, folder:str) -> bool:
        loaded = False
        try:
            model_path = os.path.join(folder, ACTOR_FILE_NAME)
            if os.path.exists(model_path):
                self.actor.load_state_dict(torch.load(model_path))
                loaded = True
        except:
            print(f"Failed to load q_learning model from {model_path}")
    
        return loaded

    @classmethod
    def observation(cls, car_state: car.CarState) -> torch.tensor:
        
        input = np.empty((INPUT_VECTOR_SIZE), dtype=np.float32)
        input[0] = car_state.position.x
        input[1] = car_state.position.y
        input[2] = car_state.wheel_angle
        input[3] = car_state.track_state.velocity_forward
        input[4] = car_state.track_state.velocity_right
        input[5:14] = car_state.track_state.rays[0:9]
        observation = torch.tensor(input, dtype=torch.float)

        return observation

    def get_action(self, car_state: car.CarState) -> car.Action:
        observation = self.observation(car_state)

        mean = self.actor(observation)
        action = mean.detach().numpy()
        car_action = car.Action(
            action[0] * self.max_acceleration, 
            action[1] * self.max_angular_velocity)
        
        return car_action



def load_model(car_config: car.CarConfig):
  
    model = Model(car_config.motion_profile.max_acceleration, 
        car_config.motion_profile.max_angular_velocity)
    loaded = model.load(os.path.dirname(__file__))
    print('Model load from data=', loaded)

    model_info = ModelInfo(name='ppo-hc', version='2023.06.24')

    return model, model_info


if __name__ == '__main__':

    race = Factory.sample_race_multi_turn_large()
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
    
