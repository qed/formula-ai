
import os

import matplotlib
import matplotlib.pyplot as plt

import random
from collections import namedtuple, deque

import torch.optim as optim
from model import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 10
GAMMA = 0.5

TAU = 1
LR = 1e-2

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ModelTrain():
    def __init__(self, model:Model, race:MiniRace):
        self.model = model
        self.race = race
        self.target_net = DQN().to(device).to(device)

        self.model.is_train = True

        self.optimizer = optim.AdamW(self.model.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def load(self, folder:str) -> bool:
        loaded = self.model.load(folder)
        self.target_net.load_state_dict(self.model.policy_net.state_dict())
                
        return loaded
    

    def save(self, folder:str) -> bool:
        try:
            model_path = os.path.join(folder, DATA_FILE_NAME)
            torch.save(self.target_net.state_dict(), f=model_path)
            return True
        except:
            print(f"Failed to save model into {model_path}")
            return False
        
   
    def train(self, episodes:int) -> float:
        global steps_done

        total_score:float = 0
        max_score:float =  -100
        for episode in range(episodes):

            self.race.run(debug=False)
            final_state = race.steps[-1].state

            total_score += final_state.time
            if final_state.time > max_score:
                max_score = final_state.time
            episode_durations.append(final_state.time)
            plot_durations()

            for i in range(len(race.steps)):
                step = race.steps[i]
                print(f'{step}')

            step_count = len(race.steps)
            state = race.steps[0].state
            state_tensor = Model.state_tensor(state)

            for i in range(1, step_count):
                step = race.steps[i]
                action_tensor = self.model.action_tensor(step.action)
                next_state = step.state
                next_state_tensor = Model.state_tensor(next_state)
            
                reward = 1
                reward_tensor = torch.tensor([reward], device=device)

                if MiniRace.is_out(next_state.position):
                    next_state_tensor = None

                self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
                state = next_state
                state_tensor = next_state_tensor

                #print(i, action)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.model.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

        return total_score/episodes, max_score

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model.policy_net(state_batch).gather(1, action_batch)
        
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        expected = expected_state_action_values.unsqueeze(1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.model.policy_net.parameters(), 100)
        self.optimizer.step()
        

if __name__ == '__main__':

    model, race = create_model_race()

    model_train = ModelTrain(model, race)
    model_train.load(os.path.dirname(__file__))
    
    for epoch in range(10):
        average, max = model_train.train(10)
        print(f"epoch {epoch}: {average, max}")
        model_train.save(os.path.dirname(__file__))


    plot_durations(show_result=True)
    plt.ioff()
    plt.show()