import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from racecar.model import load_model
from core.src.jsoner import *
from core.test.samples import Factory

if __name__ == '__main__':

    #race = Factory.sample_race_1()
    race = Factory.sample_race_multi_turn_large()
    model, model_info = load_model(race.race_info.car_config)
    race.model = model
    race.race_info.model_info = model_info

    
    race.run(debug=False)

    data_root = os.path.join(os.path.dirname(__file__), 'data')
    RaceSaver.save(race, data_root)
    print(f'Race finished, saved at {data_root}\\race\{race.race_data.race_info.id}')
