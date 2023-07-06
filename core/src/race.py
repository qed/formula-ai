
from datetime import datetime

from . import model, car, track


class ModelInfo:
    def __init__(self, name:str , version:str):
        self.type = 'ModelInfo'
        self.name = name
        self.version = version

    def __str__(self) -> str:
        return f'ModelInfo(name={self.name}, version={self.version})'


class ActionCarState:

    def __init__(self, action:car.Action, car_state:car.CarState):
        self.type = 'ActionCarState'
        self.action = action
        self.car_state = car_state

    def __str__(self) -> str:
        return f'ActionCarState(action={self.action}, car_state={self.car_state})'
    

class RaceInfo:
    def __init__(self, 
                 name: str,
                 id: str,
                 track_info: track.TrackInfo, 
                 model_info: ModelInfo, 
                 car_info: car.CarInfo,
                 car_config : car.CarConfig,
                 start_state : car.CarState,
                 round_to_finish: int, 
                 max_time_to_finish: int = 300000):

        self.type = 'RaceInfo'
        self.name = name
        self.id = id  
        self.track_info = track_info
        self.round_to_finish = round_to_finish
        self.model_info = model_info
        self.car_info = car_info
        self.car_config = car_config
        self.start_state = start_state
        self.max_time_to_finish = max_time_to_finish

    def __str__(self) -> str:
        return f'RaceInfo(name={self.name}, id={self.id}, track_info={self.track_info}, round_to_finish={self.round_to_finish}, model_info={self.model_info}, car_info={self.car_info}, car_config={self.car_config}, start_state={self.start_state})'
    

class RaceData:
    
    def __init__(self, race_info:RaceInfo, steps: list[ActionCarState]):
        self.race_info = race_info
        self.steps = steps


class Race:
    def __init__(self, 
            race_info: RaceInfo, 
            track_field: track.TrackField, 
            model: model.IModelInference):
        
        self.race_info = race_info
        self.track_field = track_field
        self.model = model
        self.steps = []

    def run(self, debug:bool = False):
    
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.race_info.id = (self.race_info.name + "_" 
            + self.race_info.track_info.id + "_" 
            + self.race_info.model_info.name + "_"
            + start_time)

        current_state = self.race_info.start_state
        self.track_field.calc_track_state(current_state)

        self.steps = []
        if debug:
            print('Race start at time', start_time)
            print(current_state)

        self.steps.append(ActionCarState(None, current_state))
        while (current_state.round_count < self.race_info.round_to_finish
               and current_state.track_state.tile_type != track.TileType.Wall.value
               and current_state.timestamp < self.race_info.max_time_to_finish) :
            
            action = self.model.get_action(current_state)
            next_state = self.track_field.get_next_state(self.race_info.car_config, current_state, action, debug)
            self.steps.append(ActionCarState(action, next_state))
            
            if debug:
                print('\naction:\n', action)
                print('\nnext:\n', next_state)

            current_state = next_state

        if debug:
            print('Race finished at time', datetime.now())
            print(current_state)

        self.race_data = RaceData(self.race_info, self.steps)
        #return self.race_data
        
    