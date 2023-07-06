import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
import math
import numpy as np

from core.src import model
from core.test.samples import Factory
from model import *


class ModelTrain(model.IModelInference):

    def load(self, folder:str) -> bool:
        loaded = False
        try:
            config_path = os.path.join(folder, CONFIG_FILE_NAME)
            self.config = neat.Config(neat.DefaultGenome, 
                neat.DefaultReproduction, 
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation, 
                config_path)
            loaded = True
        except:
            # print(f"Failed to load config from {config_path}")
            loaded = False
                
        return loaded
    

    def save(self, folder:str) -> bool:
        try:
            model_path = os.path.join(folder, DATA_FILE_NAME)
            with open(model_path, "wb") as f:
                pickle.dump(self.winner, f)
            return True
        except:
            print(f"Failed to save model into {model_path}")
            return False
        
   
    def train(self, generation_count:int) :
        # Create the population, which is the top-level object for a NEAT run.
        population = neat.Population(self.config)

        # Add a stdout reporter to show progress in the terminal.
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.Checkpointer(5))

        evaluator = neat.ParallelEvaluator(1, self.eval_model)
        self.winner = population.run(evaluator.evaluate, generation_count)

        print('\nBest genome:\n{!s}'.format(self.winner))
        print('Fitness:', self.winner.fitness)


    def eval_model(self, genome, config) -> float:

        race = Factory.sample_race_sshape()
        model, model_info = load_model(race.race_info.car_config)

        race.model = model
        race.race_info.model_info = model_info   

        model.load_genome(genome, config)
        
        race.run(debug=False)

        final_state = race.steps[-1].car_state
        # print(final_state)
        return final_state.track_state.score
    

if __name__ == '__main__':

    model_train = ModelTrain()
    loaded = model_train.load(os.path.dirname(__file__))
    print('loaded=', loaded)

    if not loaded:
        print('Fail to load NEAT config, exit')
        exit()

    model_train.train(300)
    model_train.save(os.path.dirname(__file__))


