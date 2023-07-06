from dataclasses import dataclass
from datetime import datetime
import json
import os
import pickle

from .car import *
from .track import *
from .race import *

class Jsoner:

    type_registry = {}
    type_registry['Point2D'] = Point2D
    type_registry['RotationFriction'] = RotationFriction
    type_registry['MotionProfile'] = MotionProfile
    type_registry['SlideFriction'] = SlideFriction
    type_registry['CarConfig'] = CarConfig
    type_registry['CarInfo'] = CarInfo
    type_registry['CarState'] = CarState
    type_registry['TrackState'] = TrackState

    type_registry['Action'] = Action

    type_registry['MarkLine'] = MarkLine
    type_registry['TrackInfo'] = TrackInfo
    type_registry['ModelInfo'] = ModelInfo
    type_registry['RaceInfo'] = RaceInfo
    type_registry['ActionCarState'] = ActionCarState

    @classmethod
    def to_json(cls, input: object, indent=None) ->str :
        return json.dumps(input, default=lambda o: o.__dict__, indent=indent)
    
    @classmethod
    def to_json_file(cls, input: object, file_path:str, indent=None) ->None :
        with open(file_path, 'w') as f:
            json.dump(input, f, default=lambda o: o.__dict__, indent=indent)
    
    @classmethod
    def from_json_dict(cls, json_dict: dict):
        return cls.json_to_object(json_dict, cls.type_registry)
    
    @classmethod
    def from_json_str(cls, json_str: str):
        json_dict = json.loads(json_str)
        return cls.json_to_object(json_dict, cls.type_registry)

    @classmethod
    def dict_from_json_file(cls, file_path:str) -> dict:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @classmethod
    def object_from_json_file(cls, file_path:str):
        with open(file_path, 'r') as f:
            json_dict = json.load(f)
        return cls.json_to_object(json_dict, cls.type_registry)
    
    @classmethod
    def json_to_object(cls, json_dict: dict, type_registry):
        if isinstance(json_dict, dict):
            obj_type = json_dict['type']
            del json_dict['type']
            ObjType = type_registry[obj_type]
            for key, value in json_dict.items():
                json_dict[key] = cls.json_to_object(value, type_registry)
            return ObjType(**json_dict)
        elif isinstance(json_dict, list):
            return [cls.json_to_object(elem, type_registry) for elem in json_dict]
        else:
            return json_dict
        
class TrackFieldSaver:
    @classmethod
    def save(cls, track_field: TrackField, data_folder: str):
        directory = os.path.join(os.path.join(data_folder, 'trackfield'), track_field.track_info.id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        info_file = 'info.json'
        info_path = os.path.join(directory, info_file)
        with open(info_path, 'w') as infofile:
            info_json = Jsoner.to_json(track_field.track_info, indent=4)
            # print('race_json', info_json)
            infofile.write(info_json)
        
        field_path = os.path.join(directory, 'field')
        with open(field_path, 'wb') as field_file:
            pickle.dump(track_field.field, field_file)
        
    
    @classmethod
    def load(cls, data_folder: str, id: str) -> TrackField:
        directory = os.path.join(os.path.join(data_folder, 'trackfield'), id)
        if not os.path.exists(directory):
            return None, None
        
        info_file = 'info.json'
        info_path = os.path.join(directory, info_file)
        track_info = Jsoner.object_from_json_file(info_path)
        tf = TrackField(track_info)

        field_path = os.path.join(directory, 'field')
        with open(field_path, 'rb') as field_file:
            tf.field = pickle.load(field_file)
    
        return tf
    

class RaceDataSaver:

    @classmethod
    def save(cls, race_data: RaceData, data_folder: str):
        directory = os.path.join(os.path.join(data_folder, 'race'), race_data.race_info.id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        info_file = 'info.json'
        info_path = os.path.join(directory, info_file)
        with open(info_path, 'w') as infofile:
            info_json = Jsoner.to_json(race_data.race_info, indent=4)
            # print('race_json', info_json)
            infofile.write(info_json)

        step_path = os.path.join(directory, 'action_state.log')
        with open(step_path, 'w') as logfile:
            for step in race_data.steps: 
                step_json = Jsoner.to_json(step)
                # print(step_json)
                logfile.write(step_json + '\n')
    
    @classmethod
    def load(cls, data_folder: str, id: str) -> RaceData:

        directory = os.path.join(os.path.join(data_folder, 'race'), id)
        if not os.path.exists(directory):
            return None
        
        info_file = 'info.json'
        info_path = os.path.join(directory, info_file)
        race_info = Jsoner.object_from_json_file(info_path)
        #  print('race_info_read : ', race_info)

        
        steps: list[ActionCarState] = []
        step_path = os.path.join(directory, 'action_state.log')
        with open(step_path, 'r') as logfile:
            Lines = logfile.readlines()
            for line in Lines:
                steps.append(Jsoner.from_json_str(line))

        # print('steps: ', steps)
        return RaceData(race_info, steps)


class RaceSaver:

    @classmethod
    def save(cls, race: Race, data_root: str):
        RaceDataSaver.save(race.race_data, data_root)
        TrackFieldSaver.save(race.track_field, data_root)
    
    @classmethod
    def load(cls, data_root: str, id: str):

        race_data = RaceDataSaver.load(data_root, id)
        if race_data is None:
            return None, None
        elif race_data.race_info is None:
            return None, None
        elif race_data.race_info.track_info is None:
            return None, None

        track_field = TrackFieldSaver.load(data_root, race_data.race_info.track_info.id)
        return race_data, track_field

    @classmethod
    def load_folder(cls, race_folder: str):
        id = os.path.basename(race_folder)
        data_root = os.path.dirname(os.path.dirname(race_folder))
        return cls.load(data_root, id)
