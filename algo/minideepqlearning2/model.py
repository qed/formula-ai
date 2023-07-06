import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_VECTOR_SIZE = 1

OUTPUT_VECTOR_SIZE = 2

device = "cpu"

DATA_FILE_NAME = "policy_net.pt"

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        HiddenLayerSize = 1
        self.layer1 = nn.Linear(INPUT_VECTOR_SIZE, HiddenLayerSize)
        self.layer2 = nn.Linear(HiddenLayerSize, OUTPUT_VECTOR_SIZE)


    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        return self.layer2(x)


# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_START = 0.95
EPS_END = 0.1
EPS_DECAY = 100

class MiniState:
    def __init__(self, position:float = 0, time:int = 0):
        self.position = position
        self.time = time
    
    def __str__(self) -> str:
        return f'MiniState(position={self.position}, time={self.time})'
    
class MiniAction:
    def __init__(self, move:int = 0):
        self.move = move

    def __str__(self) -> str:
        return f'MiniAction(move={self.move})'
    

class MiniActionState:
    def __init__(self, action:MiniAction, state:MiniState):
        self.action = action
        self.state = state

    def __str__(self) -> str:
         return f'MiniActionState(action={self.action}, state={self.state})' 

class MiniRace:

    def __init__(self, max_time_to_finish:int):
        self.start_state = MiniState()
        self.steps = []
        self.max_time_to_finish = max_time_to_finish
        
    def run(self, debug:bool = False) -> None:

        current_state = self.start_state

        self.steps= []
        self.steps.append(MiniActionState(None, current_state))
        
        time = 0
        while (not self.is_out(current_state.position)
               and time < self.max_time_to_finish) :
            
            action = self.model.get_action(current_state)
            next_state = MiniState(current_state.position + action, time+1)
            step_data = MiniActionState(action, next_state)
            self.steps.append(step_data)

            current_state = next_state
            time += 1

        return self.steps

    def get_next_position(self, position:int, action:int) -> int:
        next_position = position + action
        return next_position

    @classmethod
    def is_out(cls, position:int) -> bool:
        return position < -2  or position > 2
    

steps_done:int = 0 

class Model():

    def __init__(self, is_train:bool = False):
        self.policy_net = DQN().to(device)
        self.is_train = is_train


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
    def state_tensor(cls, state:MiniState) -> torch.tensor:     
        input =  torch.tensor([state.position], device=device)

        input = np.empty((INPUT_VECTOR_SIZE), dtype=np.float32)
        input[0] = state.position
        input = torch.FloatTensor(input).reshape((1, 1)).to(device)

        return input

    
    def action_tensor(self, move:int) -> torch.tensor:
        action_index = 0
        if move == 1:
            action_index = 1
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
            action_index = random.randrange(OUTPUT_VECTOR_SIZE)

        return action_index
        
    
    def get_action(self, position: int) -> int:

        input = self.state_tensor(position)
        action_index = self.select_action(input)

        if action_index == 0:
            action = -1
        else:
            action = 1
      
        return action


def create_model_race():
    race = MiniRace(10)

    model = Model()
    loaded = model.load(os.path.dirname(__file__))
    print('Model load from data=', loaded)

    race.model = model
    
    return model, race

if __name__ == '__main__':

    model, race = create_model_race()
    race.run(debug=True)

    final_state = race.steps[-1]
    print(f'Finish at {final_state}')

