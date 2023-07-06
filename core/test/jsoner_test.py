import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
import math

from samples import Factory
from src import car
from src.jsoner import *

class JsonTest(unittest.TestCase):

    def test_100_cc(self):
        print('\n===\ntest_100_cc()')

        cc_1= Factory.default_car_config()
        print('cc_1:', cc_1)
        print('type(cc_1):', type(cc_1))

        cc_json = Jsoner.to_json(cc_1)
        print('cc_json:', cc_json)

        cc_2 = json.loads(cc_json)
        print('cc_2:', cc_2)
        print('type(cc_2):', type(cc_2))

        cc_3 = Jsoner.from_json_dict(cc_2)
        print('cc_3:', cc_3)
        print('type(cc_3):', type(cc_3))

        print('cc_1 == cc_3:', cc_1 == cc_3) 


    def test_101_cc(self):
        print('\n===\ntest_101_cc()')

        cc_1= Factory.default_car_config()
        print('cc_1:', cc_1)
        print('type(cc_1):', type(cc_1))

        directory = 'data/carconfig'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = 'data/carconfig/cc_1.json'
        Jsoner.to_json_file(cc_1, file_name)

        cc_2 = Jsoner.dict_from_json_file(file_name)
        print('cc_2:', cc_2)
        print('type(cc_2):', type(cc_2))

        cc_3 = Jsoner.object_from_json_file(file_name)
        print('cc_3:', cc_3)
        print('type(cc_3):', type(cc_3))

        print('cc_1 == cc_3:', cc_1 == cc_3) 

    def test_200_save_tf(self):
        print('\n===\ntest_200_save_tf()')

        tf = Factory.sample_track_field_0()
        print('tf:', tf)
        print('type(tf):', type(tf))

        TrackFieldSaver.save(tf, 'data')

    def test_201_load_tf(self):
        print('\n===\ntest_201_load_tf()')
        tf= TrackFieldSaver.load('data', 'sample_track_field_0')
        print('tf : ', tf)

    def test_300_save_race_data(self):
        print('\n===\ntest_300_save_race_data()')
        race = Factory.sample_race_0()
        race.run(debug=False) 
        race.race_data.race_info.id = 'SampleRace0_20230512_000000'
        RaceDataSaver.save(race.race_data, 'data')
        TrackFieldSaver.save(race.track_field, 'data')

    def test_301_load_race_data(self):
        print('\n===\ntest_301_load_race_data()')
        race_data = RaceDataSaver.load('data', 'SampleRace0_20230512_000000')
        track_field = TrackFieldSaver.load('data', race_data.race_info.track_info.id)

        print('track_field : ', track_field)

        print('race_info : ', race_data.race_info)

        print('steps: ')
        for step in race_data.steps:
            print(step)

    def test_400_save_race(self):
        print('\n===\ntest_400_save_race()')
        race = Factory.sample_race_1()
        race.run(debug=False) 
        race.race_data.race_info.id = 'SampleRace1_20230618_120000'
        RaceSaver.save(race, 'data')


    def test_401_load_race_by_id(self):
        print('\n===\ntest_401_load_race_by_id()')
    
        race_data, track_field = RaceSaver.load('data', 'SampleRace1_20230618_120000')

        print('track_field : ', track_field)

        print('race_info : ', race_data.race_info)

        print('steps: ')
        for step in race_data.steps:
            print(step)

    def test_402_load_race_fullpath(self):
        print('\n===\ntest_402_load_race_fullpath()')
    
        race_data, track_field = RaceSaver.load_folder('data/race/SampleRace1_20230618_120000')

        print('track_field : ', track_field)

        print('race_info : ', race_data.race_info)

        print('steps: ')
        for step in race_data.steps:
            print(step)

if __name__ == '__main__':
    unittest.main()

