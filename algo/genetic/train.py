import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random

import numpy as np
import copy
from core.src import model
from core.test.samples import Factory
from model import *

import torch
from torch import nn
from torch.nn.parameter import Parameter
# because we do not need gradients for GA
torch.set_grad_enabled(False)

from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method("spawn")
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POPULATION_SIZE = 50
TOP_SIZE = 5

CROSS_RATE = 0.65
CROSS_CHANCE = 0.6
MUTATION_FACTOR = 0.1
MUTATION_RATE = 0.15

POPULATION_FILE_NAME = "population.pt"
Parameters = list[Parameter]

class ModelTrain(model.IModelInference):

    def __init__(self):
        self.population = []


    def load(self, folder:str) -> bool:

        loaded = False
        try:
            population_path = os.path.join(folder, POPULATION_FILE_NAME)
            self.population = torch.load(population_path)
            loaded = True
        except:
            print(f"Failed to load population from {population_path}")

        return loaded 
                

    def init_population(self):
        model = Model()
        params = model.get_params()
        shapes = [param.shape for param in params]
        self.population = []
        for _ in range(POPULATION_SIZE):
            entity = []
            for shape in shapes:
                # if fan in and fan out can be calculated (tensor is 2d) then using kaiming uniform initialisation
                # as per nn.Linear
                # otherwise use uniform initialisation between -0.5 and 0.5
                try:
                    rand_tensor = nn.init.kaiming_uniform_(torch.empty(shape)).to(device)
                except ValueError:
                    rand_tensor = nn.init.uniform_(torch.empty(shape), -0.2, 0.2).to(device)
                entity.append((torch.nn.parameter.Parameter(rand_tensor)))
            self.population.append(entity)
        
    def save(self, folder:str) -> bool:

        model_saved:bool = False
        try:
            model = Model()
            model.set_params(self.population[-1])
            model_path = os.path.join(folder, DATA_FILE_NAME)
            torch.save(model.net.state_dict(), model_path)
            model_saved = True
        except:
            print(f"Failed to save model into {model_path}")
            model_saved = False
        
        population_saved:bool = False
        try:
            population_path = os.path.join(folder, POPULATION_FILE_NAME)
            torch.save(self.population, population_path)
            population_saved = True
        except:
            print(f"Failed to save population into {population_path}")
            population_saved = False

        return model_saved and population_saved
    
    def train(self) :

        fitnesses = self.evaluate_population(self.population)
        print(f"fitnesses = {fitnesses}")

        n_fittest = [self.population[x] for x in np.argpartition(fitnesses, -TOP_SIZE)[-TOP_SIZE:]]

        clipped_fitnesses = np.clip(fitnesses, 1, None) 
        wheel = self.make_wheel(self.population, clipped_fitnesses)
        self.population = self.select(wheel, len(self.population) - TOP_SIZE)
        self.population.extend(n_fittest)

        pop2 = list(self.population)
        for index in range(len(self.population) - TOP_SIZE):
            child = self.crossover(self.population[index], pop2)
            child = self.mutate(child)
            self.population[index] = child

    def eval_model(self, params: Parameters) -> float:

        race = Factory.sample_race_sshape()
        model, model_info = load_model(race.race_info.car_config)
        race.model = model
        race.race_info.model_info = model_info

        model.set_params(params)

        race.run(debug=False)
        final_state = race.steps[-1].car_state
        return final_state.track_state.score
    

    def evaluate_population(self, population: list[Parameters]) -> list[float]:
        fitnesses = np.array(list(map(self.eval_model, population)))
        avg_fitness = fitnesses.sum() / len(fitnesses)
        print(f"avg: {avg_fitness:6.2f}, fittest: {fitnesses.max():6.2f} at {fitnesses.argmax()}")
        return fitnesses


    def make_wheel(self, population:list[Parameters], fitness: np.ndarray):
        wheel = []
        total = fitness.sum()
        top = 0
        for p, f in zip(population, fitness):
            f = f/total
            wheel.append((top, top+f, p))
            top += f
        return wheel


    def binary_search(self, wheel, num):
        mid = len(wheel)//2
        low, high, answer = wheel[mid]
        if low<=num<=high:
            return answer
        elif high < num:
            return self.binary_search(wheel[mid+1:], num)
        else:
            return self.binary_search(wheel[:mid], num)


    def select(self, wheel, target:int):
        answer = []
        while len(answer) < target:
            r = random.random()
            answer.append(self.binary_search(wheel, r))
        return answer

    def crossover(self, parent1: Parameters, pop: list[Parameters]) -> Parameters:
        """
        Crossover two individuals and produce a child.

        This is done by randomly splitting the weights and biases at each layer for the parents and then
        combining them to produce a child

        @params
            parent1 (Parameters): A parent that may potentially be crossed over
            pop (List[Parameters]): The population of solutions
        @returns
            Parameters: A child with attributes of both parents or the original parent1
        """
        if np.random.rand() < CROSS_RATE:
            pop_index = np.random.randint(0, len(pop), size=1)[0]
            parent2 = pop[pop_index]
            child = []
            mask = None

            for p1l, p2l in zip(parent1, parent2):
                # splitpoint = int(len(p1l) * split)
                # new_param = nn.parameter.Parameter(
                #     torch.cat([p1l[:splitpoint], p2l[splitpoint:]])
                # )
                # child.append(new_param)
                if len(p1l.shape) == 2:
                    mask = torch.bernoulli(torch.full((p1l.shape[0],), CROSS_CHANCE)).int()
                    tmp = mask.broadcast_to((p1l.shape[1], p1l.shape[0])).transpose(0, 1)
                else:
                    tmp = mask
                reverse_mask = torch.ones(p1l.shape).int() - tmp
                new_param = nn.parameter.Parameter(p1l * reverse_mask + p2l * tmp)
                child.append(new_param)
 
            return child
        else:
            return copy.deepcopy(parent1)


    def gen_mutate(self, shape: torch.Size) -> torch.Tensor:
        """
        Generate a tensor to use for random mutation of a parameter

        @params
            shape (torch.Size): The shape of the tensor to be created
        @returns
            torch.tensor: a random tensor
        """
        drop_rate = 1 - MUTATION_RATE
        dropout = nn.Dropout(drop_rate)(torch.ones(shape))
        randn = torch.randn(shape)
        result1 = dropout * randn 
        result = result1 * MUTATION_FACTOR
        return result


    def mutate(self, child: Parameters) -> Parameters:
        """
        Mutate a child

        @params
            child (Parameters): The original parameters
        @returns
            Parameters: The mutated child
        """
        for i in range(len(child)):
            for j in range(len(child[i])):
                gene = child[i][j]
                mutate = self.gen_mutate(child[i][j].shape)
                gene += mutate

        return child


if __name__ == '__main__':

    model_train = ModelTrain()
    loaded = model_train.load(os.path.dirname(__file__))
    print('loaded=', loaded)

    if not loaded:
        model_train.init_population()
        model_train.save(os.path.dirname(__file__))


    for i in range(1000):
        print(f"Generation {i}")
        model_train.train()

        if i % 20 == 0:
            model_train.save(os.path.dirname(__file__))
