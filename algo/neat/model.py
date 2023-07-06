import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pickle
import neat
import numpy as np
import math

from core.src import model, car
from core.src.race import *
from core.test.samples import Factory

INPUT_VECTOR_SIZE = 14
OUTPUT_VECTOR_SIZE = 2

DATA_FILE_NAME = "neat_genome"
CONFIG_FILE_NAME = "config-feedforward"

class Model(model.IModelInference):

    def __init__(self, max_acceleration:float = 1, max_angular_velocity:float = 1):
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity

    def load(self, folder:str) -> bool:
        loaded = False
        try:
            model_path = os.path.join(folder, DATA_FILE_NAME)
            with open(model_path, "rb") as f:
                winner = pickle.load(f)
            
            config_path = os.path.join(folder, CONFIG_FILE_NAME)
            config = neat.Config(neat.DefaultGenome, 
                neat.DefaultReproduction, 
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation, 
                config_path)

            self.net = neat.nn.FeedForwardNetwork.create(winner, config)
            loaded = True
        except:
            # print(f"Failed to load NEAT model from {folder}")
            loaded = False
    
        return loaded

    def load_genome(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    """
        inference
    """
    def get_action(self, car_state: car.CarState) -> car.Action:
        
        input = np.empty((INPUT_VECTOR_SIZE), dtype=np.float32)
        input[0] = car_state.position.x
        input[1] = car_state.position.y
        input[2] = car_state.wheel_angle
        input[3] = car_state.track_state.velocity_forward
        input[4] = car_state.track_state.velocity_right
        input[5:14] = car_state.track_state.rays[0:9]

        output = self.net.activate(input)

        return car.Action(self.max_acceleration*math.tanh(output[0]), self.max_angular_velocity*math.tanh(output[1]))


def load_model(car_config: car.CarConfig):
  
    model = Model(car_config.motion_profile.max_acceleration, 
        car_config.motion_profile.max_angular_velocity)
    loaded = model.load(os.path.dirname(__file__))
    # print('Model load from data=', loaded)

    model_info = ModelInfo(name='neat-hc', version='2023.5.20')

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