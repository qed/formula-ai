import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.distributions import MultivariateNormal
from core.src import model
from core.src.race import *

from model import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_progress = []


def plot_progress(show_result=False):
    plt.figure(1)
    progress_tensor = torch.tensor(episode_progress, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(progress_tensor.numpy())
    # Take 100 episode averages and plot them too
    if len(progress_tensor) >= 100:
        means = progress_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



class ModelTrain(model.IModelInference):
    def __init__(self, model:Model, race:Race):
        self.model = model
        self.race = race

        self._init_hyperparameters()
        self.critic = Critic(INPUT_VECTOR_SIZE, 1)

        # Create covariance matrix for get_action()
        self.cov_var = torch.full(size=(OUTPUT_VECTOR_SIZE,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.model.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.episodes_per_batch = 10            # episode(race) per batch
        self.gamma = 0.9					   # discount factor
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005
                
    def load(self, folder:str) -> bool:
        loaded = self.model.load(folder)

        try:
            model_path = os.path.join(folder, CRITIC_FILE_NAME)
            if os.path.exists(model_path):
                self.critic.load_state_dict(torch.load(model_path))
        except:
            print(f"Failed to load q_learning model from {model_path}")
            loaded = False
                
        return loaded
    

    def save(self, folder:str) -> bool:
        try:
            torch.save(self.model.actor.state_dict(), os.path.join(folder, ACTOR_FILE_NAME))
            torch.save(self.critic.state_dict(), os.path.join(folder, CRITIC_FILE_NAME))
            return True
        except:
            print(f"Failed to save model into {folder}")
            return False
        
   
    def train(self):

        batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, average_score, max_score = self.rollout()

        # Calculate V_{phi, k}
        V, _ = self.evaluate(batch_obs, batch_acts)

        # Calculate advantage
        A_k = batch_rewards_to_go - V.detach()
        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        for _ in range(self.n_updates_per_iteration):      
            # Calculate pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # Calculate ratios
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * A_k

            clamp = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
            surr2 = clamp * A_k

            min_surr = torch.min(surr1, surr2)
            actor_loss = (-min_surr).mean()
            # Calculate gradients and perform backward propagation for actor 
            # network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            critic_loss = nn.MSELoss()(V, batch_rewards_to_go)
            # Calculate gradients and perform backward propagation for critic network    
            self.critic_optim.zero_grad()    
            critic_loss.backward()    
            self.critic_optim.step()

        return average_score, max_score

    def rollout(self):

        max_score = 0
        total_score = 0

        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []            # batch rewards
        batch_rewards_to_go = []            # batch rewards-to-go

        for episode in range(self.episodes_per_batch):
            # Rewards this episode
            ep_rewards = []

            current_state = self.race.race_info.start_state
            self.race.track_field.calc_track_state(current_state)

            while ((current_state.timestamp < 1000 # let it start
                    or (current_state.velocity_x != 0 or current_state.velocity_y != 0))
               and current_state.round_count < self.race.race_info.round_to_finish
               and current_state.track_state.tile_type != track.TileType.Wall.value
               and current_state.timestamp < self.race.race_info.max_time_to_finish) :
                
                observation, action, log_prob, car_action = self.get_action(current_state)

                batch_obs.append(observation)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                next_state = self.race.track_field.get_next_state(self.race.race_info.car_config, current_state, car_action, debug=False)
                reward = next_state.track_state.tile_total_distance - current_state.track_state.tile_total_distance
                ep_rewards.append(reward)

                current_state = next_state


            batch_rewards.append(ep_rewards) 
            episode_score = current_state.track_state.score
            if max_score < episode_score:
                max_score = episode_score
            total_score += episode_score
            
            episode_progress.append(episode_score)
            plot_progress()

        # ALG STEP #4
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, total_score/self.episodes_per_batch, max_score


    def get_action(self, car_state: car.CarState):

        observation = self.model.observation(car_state)

        mean = self.model.actor(observation)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
		# Sample an action from the distribution and get its log prob
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)

        action = action_tensor.detach().numpy()
        car_action = car.Action(
            action[0] * self.model.max_acceleration, 
            action[1] * self.model.max_angular_velocity)
        
        return observation.detach().numpy(), action, log_prob.detach(), car_action

    def compute_rewards_to_go(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rewards_to_go = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0 # The discounted reward so far
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0, discounted_reward)
    
        # Convert the rewards-to-go into a tensor
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)
        return batch_rewards_to_go
    	
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.model.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

if __name__ == '__main__':

    race = Factory.sample_race_multi_turn_large()
    model, model_info = load_model(race.race_info.car_config)

    race.model = model
    race.race_info.model_info = model_info

    model_train = ModelTrain(model, race)
    model_train.load(os.path.dirname(__file__))
    
    for epoch in range(1000):
        average, max = model_train.train()
        print(f"epoch {epoch}: {average, max}")
        model_train.save(os.path.dirname(__file__))

    plot_progress(show_result=True)
    plt.ioff()
    plt.show()
