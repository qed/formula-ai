import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import math
from src.track import *
from src.car import *
from src.race import *    

class ModelSpecialNumber(model.IModelInference):

    def load(self, folder:str) -> bool:
        return True

    def get_action(self, car_state: CarState) -> Action:
        if car_state.position.x > 14 and car_state.position.x < 18 and car_state.position.y < 7:
            return Action(2, 0)
        elif car_state.position.x > 18 and car_state.position.y < 14:
            return Action(2, 1.2)
        elif car_state.position.x > 11 and car_state.position.y > 12: 
            return Action(2, 0)
        elif car_state.position.x < 11 and car_state.position.y < 14: 
            return Action(2, 1.75)     
        return Action(2, 0)



class Factory:

    @classmethod
    def sample_track_field_0(cls) -> TrackField:

        track_info = TrackInfo(
            id='trackfield0', 
            row=5, 
            column=8,
            time_interval = 1000)
    
        tf = TrackField(track_info)
        for x in range(track_info.column) :
            tf.field[0, x]['type'] = TileType.Wall.value
            tf.field[1, x]['type'] = TileType.Shoulder.value
            tf.field[2, x]['type'] = TileType.Road.value
            tf.field[2, x]['distance'] = x
            tf.field[3, x]['type'] = TileType.Shoulder.value
            tf.field[4, x]['type'] = TileType.Wall.value
        
        return tf

    @classmethod
    def sample_track_field_1(cls) -> TrackField:
        track_info = TrackInfo(
            id='trackfield1', 
            row=20, 
            column=30, 
            time_interval = 1000)
        tf = TrackField(track_info)

        # inner Wall
        tf.fill_block(range(8, 12), range(8, 22), TileType.Wall.value)

        # inner Shoulder         
        tf.fill_block(range(6, 8), range(6, 24), TileType.Shoulder.value)
        tf.fill_block(range(12, 14), range(6, 24), TileType.Shoulder.value)
        tf.fill_block(range(8, 12), range(6, 8), TileType.Shoulder.value)
        tf.fill_block(range(8, 12), range(22, 24), TileType.Shoulder.value)

        # Road
        tf.fill_block(range(4, 6), range(4, 26), TileType.Road.value)
        tf.fill_block(range(14, 16), range(4, 26), TileType.Road.value)
        tf.fill_block(range(6, 14), range(4, 6), TileType.Road.value)
        tf.fill_block(range(6, 14), range(24, 26), TileType.Road.value)

        # outer Shoulder
        tf.fill_block(range(2, 4), range(2, 28), TileType.Shoulder.value)
        tf.fill_block(range(16, 18), range(2, 28), TileType.Shoulder.value)
        tf.fill_block(range(4, 16), range(2, 4), TileType.Shoulder.value)
        tf.fill_block(range(4, 16), range(26, 28), TileType.Shoulder.value)

        # outer Wall
        tf.fill_block(range(0, 2), range(0, 30), TileType.Wall.value)
        tf.fill_block(range(18, 20), range(0, 20), TileType.Wall.value)
        tf.fill_block(range(2, 18), range(0, 2), TileType.Wall.value)
        tf.fill_block(range(2, 18), range(28, 30), TileType.Wall.value)

        return tf

    @classmethod
    def sample_track_field_2(cls, compute_distance:bool = False, debug:bool = False) -> TrackField:
        track_info = TrackInfo(
            id='trackfield2', 
            row=20, 
            column=30,
            start_line=MarkLine(4, 7, 14, 15),
            finish_line=MarkLine(4, 7, 13, 14),
            time_interval = 100)

        tf = TrackField(track_info)

        tf.fill_block(range(0, 20), range(0, 30), TileType.Wall.value)
        tf.fill_block(range(2, 18), range(2, 28), TileType.Shoulder.value)
        tf.fill_block(range(4, 16), range(4, 26), TileType.Road.value)     
        tf.fill_block(range(7, 13), range(11, 19), TileType.Shoulder.value)
        tf.fill_block(range(9, 11), range(13, 17), TileType.Wall.value)

   
        # block start and finish line by wall, allow only get to them from the road
        tf.fill_block(range(2, 4), range(12, 16), TileType.Wall.value)   # block top by wall
        tf.fill_block(range(7, 9), range(12, 16), TileType.Wall.value)   # block bottom by wall
        
        if compute_distance:
            tf.compute_tile_distance(debug)

        return tf
    

    @classmethod
    def sample_track_field_sshape(cls, compute_distance:bool = False, debug:bool = False) -> TrackField:
        track_info = TrackInfo(
            id='sshape', 
            row=18, 
            column=31,
            start_line=MarkLine(2, 4, 21, 22),
            finish_line=MarkLine(2, 4, 20, 21),
            time_interval = 100)

        tf = TrackField(track_info)

        # outer range
        tf.fill_block(range(0, 18), range(0, 31), TileType.Wall.value)
        tf.fill_block(range(1, 17), range(1, 30), TileType.Shoulder.value)
        tf.fill_block(range(2, 16), range(2, 29), TileType.Road.value)

        # inside part
        tf.fill_block(range(1, 9), range(9, 12), TileType.Shoulder.value)
        tf.fill_block(range(1, 8), range(10, 11), TileType.Wall.value)

        tf.fill_block(range(9, 17), range(19, 22), TileType.Shoulder.value)
        tf.fill_block(range(10, 18), range(20, 21), TileType.Wall.value)

        tf.fill_block(range(4, 14), range(4, 7), TileType.Shoulder.value)
        tf.fill_block(range(11, 14), range(7, 14), TileType.Shoulder.value)
        tf.fill_block(range(4, 14), range(14, 17), TileType.Shoulder.value)
        tf.fill_block(range(4, 7), range(17, 24), TileType.Shoulder.value)
        tf.fill_block(range(4, 14), range(24, 27), TileType.Shoulder.value)

        tf.fill_block(range(5, 13), range(5, 6), TileType.Wall.value)
        tf.fill_block(range(12, 13), range(6, 16), TileType.Wall.value)        
        tf.fill_block(range(5, 13), range(15, 16), TileType.Wall.value)
        tf.fill_block(range(5, 6), range(16, 25), TileType.Wall.value)
        tf.fill_block(range(5, 13), range(25, 26), TileType.Wall.value)

        
        # block start and finish line by wall, allow only get to them from the road
        tf.fill_block(range(1, 2), range(19, 23), TileType.Wall.value)   # block top
        tf.fill_block(range(4, 5), range(19, 23), TileType.Wall.value)   # block bottom
        
        if compute_distance:
            tf.compute_tile_distance(debug)

        return tf

    @classmethod
    def sample_track_field_round_angle(cls, compute_distance:bool = False, debug:bool = False) -> TrackField:
        track_info = TrackInfo(
            id='round_angle', 
            row=30, 
            column=40,
            start_line=MarkLine(1, 10, 20, 21),
            finish_line=MarkLine(1, 10, 19, 20),
            time_interval = 100)

        tf = TrackField(track_info)

        # outer range
        tf.fill_block(range(0, 30), range(0, 40), TileType.Wall.value)
        tf.fill_block(range(10, 20), range(10, 30), TileType.Wall.value)

        tf.fill_block(range(1, 10), range(10, 30), TileType.Road.value)
        tf.fill_block(range(21, 29), range(10, 30), TileType.Road.value)
        tf.fill_block(range(10, 20), range(1, 10), TileType.Road.value)
        tf.fill_block(range(10, 20), range(30, 39), TileType.Road.value)
    
        tf.fill_circle(10, 10, 9, range(0, 10), range(0, 10), TileType.Road.value)
        tf.fill_circle(10, 30, 9, range(0, 10), range(30, 40), TileType.Road.value)
        tf.fill_circle(20, 10, 9, range(20, 30), range(0, 10), TileType.Road.value)
        tf.fill_circle(20, 30, 9, range(20, 30), range(30, 40), TileType.Road.value)
        
        if compute_distance:
            tf.compute_tile_distance(debug)

        return tf


    @classmethod
    def sample_track_field_multi_turn(cls, compute_distance:bool = False, debug:bool = False) -> TrackField:
        track_info = TrackInfo(
            id='multi_turn', 
            row=30, 
            column=50,
            start_line=MarkLine(1, 10, 25, 26),
            finish_line=MarkLine(1, 10, 24, 25),
            time_interval = 100)

        tf = TrackField(track_info)

        # outer range
        tf.fill_block(range(0, 30), range(0, 50), TileType.Wall.value)

        tf.fill_block(range(1, 10), range(10, 40), TileType.Road.value)
        tf.fill_block(range(11, 20), range(20, 30), TileType.Road.value)
        tf.fill_block(range(10, 20), range(1, 10), TileType.Road.value)
        tf.fill_block(range(10, 20), range(40, 49), TileType.Road.value)
    
        tf.fill_circle(10, 10, 9, range(0, 10), range(0, 10), TileType.Road.value)
        tf.fill_circle(10, 40, 9, range(0, 10), range(40, 50), TileType.Road.value)
        tf.fill_circle(20, 10, 9, range(20, 30), range(0, 10), TileType.Road.value)
        tf.fill_circle(20, 40, 9, range(20, 30), range(40, 50), TileType.Road.value)

        tf.fill_circle(20, 20, 9, range(10, 20), range(10, 20), TileType.Road.value)
        tf.fill_circle(20, 30, 9, range(10, 20), range(30, 40), TileType.Road.value)
        tf.fill_circle(20, 10, 9, range(20, 30), range(10, 20), TileType.Road.value)
        tf.fill_circle(20, 40, 9, range(20, 30), range(30, 40), TileType.Road.value)
        
        if compute_distance:
            tf.compute_tile_distance(debug)

        return tf
    
    @classmethod
    def sample_track_field_multi_turn_large(cls, compute_distance:bool = False, debug:bool = False) -> TrackField:
        track_info = TrackInfo(
            id='multi_turn_large', 
            row=70, 
            column=120,
            start_line=MarkLine(1, 10, 60, 61),
            finish_line=MarkLine(1, 10, 59, 60),
            time_interval = 100)

        tf = TrackField(track_info)

        # outer range
        tf.fill_block(range(0, 70), range(0, 120), TileType.Wall.value)

        #horizontal
        tf.fill_block(range(1, 10), range(10, 110), TileType.Road.value)
        tf.fill_block(range(20, 30), range(10, 40), TileType.Road.value)
        tf.fill_block(range(20, 30), range(70, 90), TileType.Road.value)
        tf.fill_block(range(40, 50), range(10, 40), TileType.Road.value)
        tf.fill_block(range(60, 69), range(10, 70), TileType.Road.value)
        tf.fill_block(range(60, 69), range(100, 110), TileType.Road.value)
        #vertical
        tf.fill_block(range(10, 20), range(1, 10), TileType.Road.value)
        tf.fill_block(range(50, 60), range(1, 10), TileType.Road.value)
        tf.fill_block(range(10, 60), range(110, 119), TileType.Road.value)
        tf.fill_block(range(30, 40), range(40, 50), TileType.Road.value)
        tf.fill_block(range(30, 40), range(60, 70), TileType.Road.value)
        tf.fill_block(range(50, 60), range(70, 80), TileType.Road.value)
        tf.fill_block(range(30, 60), range(90, 100), TileType.Road.value)

        #corner
        tf.fill_circle(10, 10, 9, range(0, 10), range(0, 10), TileType.Road.value)
        tf.fill_circle(20, 10, 9, range(20, 30), range(0, 10), TileType.Road.value)
        tf.fill_circle(30, 40, 9, range(20, 30), range(40, 50), TileType.Road.value)
        tf.fill_circle(40, 40, 9, range(40, 50), range(40, 50), TileType.Road.value)
        
        tf.fill_circle(50, 10, 9, range(40, 50), range(0, 10), TileType.Road.value)
        tf.fill_circle(60, 10, 9, range(60, 70), range(0, 10), TileType.Road.value)
        tf.fill_circle(60, 70, 9, range(60, 70), range(70, 80), TileType.Road.value)
        tf.fill_circle(50, 70, 9, range(40, 50), range(70, 80), TileType.Road.value)
        tf.fill_circle(40, 70, 9, range(40, 50), range(60, 70), TileType.Road.value)

        tf.fill_circle(30, 70, 9, range(20, 30), range(60, 70), TileType.Road.value)
        tf.fill_circle(30, 70, 9, range(20, 30), range(60, 70), TileType.Road.value)
        tf.fill_circle(30, 90, 9, range(20, 30), range(90, 100), TileType.Road.value)
        tf.fill_circle(60, 100, 9, range(60,70), range(90, 100), TileType.Road.value)
        tf.fill_circle(60, 110, 9, range(60, 70), range(110, 120), TileType.Road.value)
        tf.fill_circle(10, 110, 9, range(0, 10), range(110, 120), TileType.Road.value)
        
        if compute_distance:
            tf.compute_tile_distance(debug)

        return tf

    @classmethod
    def default_car_config(cls) -> CarConfig:

        return CarConfig(
            rotation_friction = RotationFriction(min_accel_start = 1, friction = 0.5),
            slide_friction = SlideFriction(min_velocity_start = 4, friction = 2),
            motion_profile = MotionProfile(max_acceleration = 5, max_velocity = 10, max_angular_velocity = math.pi/2))


    @classmethod
    def sample_race_0(cls) -> Race:
        
        track_field = cls.sample_track_field_2(True)
 
        model = ModelSpecialNumber()
        model_info = ModelInfo(name='simplefixedrightturn', version='0.0.21')
        
        car_info = CarInfo(id = 1024, team = 'kirin')

        race_info = RaceInfo(
            name = 'Race0',
            id = 'NotStarted',
            track_info = track_field.track_info, 
            round_to_finish = 1, 
            model_info = model_info,
            car_info = car_info,
            car_config= cls.default_car_config(),
            start_state = CarState(
                position = Point2D(y = 5.5, x = 14.5), 
                last_road_position = Point2D(y = 5.5, x = 14.5))
            )

        return Race(race_info = race_info, track_field = track_field, model = model)


    @classmethod
    def sample_race_1(cls) -> Race:
        
        track_field = cls.sample_track_field_2(True)

        model = ModelSpecialNumber()
        model_info = ModelInfo(name='simplefixedrightturn', version='0.0.21')
        car_info = CarInfo(id = 1024, team = 'kirin')

        race_info = RaceInfo(
            name = 'Race1',
            id = 'NotStarted',
            track_info = track_field.track_info, 
            model_info = model_info, 
            car_info = car_info,
            car_config= cls.default_car_config(),
            start_state = CarState(
                position = Point2D(y = 5.5, x = 14.5), 
                last_road_position = Point2D(y = 5.5, x = 14.5)),
            round_to_finish = 1,
            max_time_to_finish = 10000)

        return Race(race_info = race_info, track_field = track_field, model = model)
    

    @classmethod
    def sample_race_sshape(cls) -> Race:
        
        track_field = cls.sample_track_field_sshape(True)

        model_info = ModelInfo(name='unknown', version='0.0.21')
        car_info = CarInfo(id = 123, team = 'halo')

        race_info = RaceInfo(
            name = 'Race2',
            id = 'NotStarted',
            track_info = track_field.track_info, 
            model_info = model_info, 
            car_info = car_info,
            car_config= cls.default_car_config(),
            start_state = CarState(
                position = Point2D(y = 3, x = 21), 
                last_road_position = Point2D(y = 3, x = 21)),
            round_to_finish = 1,
            max_time_to_finish = 100000)

        return Race(race_info = race_info, track_field = track_field, model = None)
    

    @classmethod
    def sample_race_round_angle(cls) -> Race:
        
        track_field = cls.sample_track_field_round_angle(True)

        model_info = ModelInfo(name='unknown', version='0.0.21')
        car_info = CarInfo(id = 123, team = 'halo')

        race_info = RaceInfo(
            name = 'Race3',
            id = 'NotStarted',
            track_info = track_field.track_info, 
            model_info = model_info, 
            car_info = car_info,
            car_config= cls.default_car_config(),
            start_state = CarState(
                position = Point2D(y = 5, x = 20), 
                last_road_position = Point2D(y = 5, x = 20)),
            round_to_finish = 1,
            max_time_to_finish = 100000)

        return Race(race_info = race_info, track_field = track_field, model = None)
    


    @classmethod
    def sample_race_multi_turn(cls) -> Race:
        
        track_field = cls.sample_track_field_multi_turn(True)

        model_info = ModelInfo(name='unknown', version='0.0.21')
        car_info = CarInfo(id = 123, team = 'halo')

        race_info = RaceInfo(
            name = 'Race4',
            id = 'NotStarted',
            track_info = track_field.track_info, 
            model_info = model_info, 
            car_info = car_info,
            car_config= cls.default_car_config(),
            start_state = CarState(
                position = Point2D(y = 5, x = 25), 
                last_road_position = Point2D(y = 5, x = 25)),
            round_to_finish = 1,
            max_time_to_finish = 100000)

        return Race(race_info = race_info, track_field = track_field, model = None)
    

    @classmethod
    def sample_race_multi_turn_large(cls) -> Race:
        
        track_field = cls.sample_track_field_multi_turn_large(True)

        model_info = ModelInfo(name='unknown', version='0.0.21')
        car_info = CarInfo(id = 123, team = 'halo')

        race_info = RaceInfo(
            name = 'Race5',
            id = 'NotStarted',
            track_info = track_field.track_info, 
            model_info = model_info, 
            car_info = car_info,
            car_config= cls.default_car_config(),
            start_state = CarState(
                position = Point2D(y = 5, x = 60), 
                last_road_position = Point2D(y = 5, x = 60)),
            round_to_finish = 1,
            max_time_to_finish = 100000)

        return Race(race_info = race_info, track_field = track_field, model = None)