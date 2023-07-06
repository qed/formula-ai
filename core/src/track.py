import numpy as np
from enum import Enum
import math

from . import car

"""
Track system acts as physics engine.

A track field consists of equally sized tiles, with upper left corner as (0, 0).

A [500, 2000] track field has, upper right corner at (0, 1999),
lower left corner at (499, 0), and lower right corner at (499, 1999)
"""

# Use value as friction ratio => Shoulder surface friction is 3 * Road
class TileType(Enum):
    Road = 1
    Shoulder = 3
    Wall = 1024

 
class TileCell :  
    def __init__(self, row:int = 0, col:int = 0):
        self.type = 'TileCell'
        self.row = row
        self.col = col

    def __str__(self) -> str:
        return f'TileCell(row={self.row}, col={self.col})'

class MarkLine:
    def __init__(self, 
            y_start:int = 0, 
            y_end:int = 0, 
            x_start:int = 0, 
            x_end:int = 0):
        self.type = 'MarkLine'
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end


    def __str__(self) -> str:
        return f'MarkLine(y:({self.y_start}, {self.y_end}), x:({self.x_start}, {self.x_end}))'


class TrackInfo:    
    def __init__(self, 
            id:str, 
            round_distance:int = 0, 
            row:int= 1, 
            column:int = 1,
            start_line:MarkLine = MarkLine(), 
            finish_line:MarkLine = MarkLine(), 
            time_interval:int = 100):
        
        self.type = 'TrackInfo'
        self.id = id
        self.round_distance = round_distance
        self.row = row
        self.column = column
        self.start_line = start_line
        self.finish_line = finish_line
        self.time_interval = time_interval      # milliseconds


    def __str__(self) -> str:
        return (
            f'TrackInfo(id={self.id}'
            + f', round_distance={self.round_distance}'
            + f', row={self.row}'
            + f', column={self.column}'
            + f', start_line:{self.start_line}'
            + f', finish_line:{self.finish_line}'
            + f', time_interval={self.time_interval})'
        )
    


class TrackField:

    def __init__(self, track_info: TrackInfo):
        self.track_info = track_info
        self.field = np.zeros((track_info.row, track_info.column), dtype=np.dtype([('type', 'H'), ('distance', 'H')]))


    def fill_block(self, y_range: range, x_range: range , type: int) :
        for y in y_range :
            for x in x_range :
                self.field[y, x]['type'] = type
    
    def fill_circle(self, 
        center_y:float, 
        center_x:float,
        radius:float,
        y_range: range, 
        x_range: range, 
        type: int) :
        for y in y_range :
            for x in x_range :
                cell_y = y+.5
                cell_x = x+.5
                distance = math.sqrt((cell_y - center_y)**2 + (cell_x - center_x)**2)
                if distance <= radius :
                    self.field[y, x]['type'] = type

    def is_start(self, cell: TileCell) -> bool:
        return (self.track_info.start_line.y_start <= cell.row 
            and cell.row < self.track_info.start_line.y_end 
            and self.track_info.start_line.x_start <= cell.col 
            and cell.col < self.track_info.start_line.x_end)
        
    def is_finish(self, cell: TileCell) -> bool:
        return (self.track_info.finish_line.y_start <= cell.row 
            and cell.row < self.track_info.finish_line.y_end 
            and self.track_info.finish_line.x_start <= cell.col 
            and cell.col < self.track_info.finish_line.x_end)
    

    def compute_tile_distance(self, debug:bool = False) -> None:

        MAX_Distance = 65535

        # Create a queue for BFS
        queue = []

        for y in range(self.track_info.row) :
            for x in range(self.track_info.column) :
                if self.field[y, x]['type'] == TileType.Wall.value:
                    self.field[y, x]['distance'] = 0 
                if self.field[y, x]['type'] == TileType.Shoulder.value:
                    self.field[y, x]['distance'] = 0
                if self.field[y, x]['type'] == TileType.Road.value:
                    self.field[y, x]['distance'] = MAX_Distance
                
		# Add the start line
        for y in range(self.track_info.start_line.y_start, self.track_info.start_line.y_end) :
            for x in range(self.track_info.start_line.x_start, self.track_info.start_line.x_end) :
                self.field[y, x]['distance'] = 0
                queue.append(TileCell(y, x))
        
        while queue:
            center = queue.pop(0)
            if debug:    
                print ('\ncenter', center)
            center_distance = int(self.field[center.row, center.col]['distance'])
            
            for y in [-1,0,1] :
                for x in [-1,0,1] :
                    if y == 0 and x == 0:
                        continue    # center
                    
                    target = TileCell(center.row + y, center.col + x)
                    if target.row < 0 or target.row >= self.field.shape[0] :
                        continue # row out of bound
                    if target.col < 0 or target.col >= self.field.shape[1] :
                        continue # col out of bound
                    if self.field[target.row, target.col]['type'] != TileType.Road.value :
                        continue # only deal with bound
                    if (self.is_start(center) and self.is_finish(target)):
                        # not process finish line from start line,
                        continue 
                    if (self.is_start(target)):
                        continue 
                    if (self.field[target.row, target.col]['distance'] == MAX_Distance):
                        queue.append(target)
                        self.field[target.row, target.col]['distance'] = center_distance + 1
                        if debug:
                            print('append queue', target, self.field[target.row, target.col]['distance'])
 
        if debug:
            for y in range(self.track_info.row) :
                for x in range(self.track_info.column) :
                    print(self.field[y, x]['distance'], end=' ')
                print()


        # Reverse direction From finish line to start line, update distance based on the distance to finish line
        
        # find the minimum distance to finish line
        self.track_info.round_distance = MAX_Distance
        for y in range(self.track_info.finish_line.y_start, self.track_info.finish_line.y_end) :
            for x in range(self.track_info.finish_line.x_start, self.track_info.finish_line.x_end) :
                queue.append(TileCell(y, x))
                if self.field[y, x]['distance'] < self.track_info.round_distance:   
                    self.track_info.round_distance = int(self.field[y, x]['distance'])
        if debug:
            print("Minimum self.track_info.round_distance:", self.track_info.round_distance)

        # start all road tile use distance of finish line
        for y in range(self.track_info.row) :
            for x in range(self.track_info.column) :
                if self.field[y, x]['type'] == TileType.Road.value:
                    self.field[y, x]['distance'] = self.track_info.round_distance
        if debug:
            print("Start Reverse")
            for y in range(self.track_info.row) :
                for x in range(self.track_info.column) :
                    print(self.field[y, x]['distance'], end=' ')
                print()


        while queue:
            center = queue.pop(0)
            center_distance = int(self.field[center.row, center.col]['distance'])
            if debug:
                print ('\ncenter', center, ':', center_distance)

            for y in [-1,0,1] :
                for x in [-1,0,1] :
                    if y == 0 and x == 0:
                        continue    # center
                    
                    target = TileCell(center.row + y, center.col + x)
                    if target.row < 0 or target.row >= self.field.shape[0] :
                        continue # row out of bound
                    if target.col < 0 or target.col >= self.field.shape[1] :
                        continue # col out of bound
                    if self.field[target.row, target.col]['type'] != TileType.Road.value :
                        continue # only deal with bound
                    if (self.is_finish(target)):
                        continue 
                    if (self.is_finish(center) and self.is_start(target)):
                        # not process start line from finish line,
                        continue 
                    if (self.field[target.row, target.col]['distance'] == self.track_info.round_distance):
                        queue.append(target)
                        self.field[target.row, target.col]['distance'] = center_distance - 1
                        if debug:
                            print('append queue', target, self.field[target.row, target.col]['distance']) 

        self.track_info.round_distance += 1 # add 1 from finish line to start line
        if debug:
            print("After Reverse")
            for y in range(self.track_info.row) :
                for x in range(self.track_info.column) :
                    print(self.field[y, x]['distance'], end=' ')
                print()
            print("self.track_info.round_distance:", self.track_info.round_distance)

    def calc_track_state(self, car_state:car.CarState) -> None:

        if car_state.wheel_angle > math.pi :
            car_state.wheel_angle -= math.pi*2
        elif car_state.wheel_angle < -math.pi :
            car_state.wheel_angle += math.pi*2

        track_state = car.TrackState()
        track_state.velocity_forward = (car_state.velocity_x * math.cos(0 - car_state.wheel_angle) 
            + car_state.velocity_y * math.cos(math.pi / 2 - car_state.wheel_angle))
        track_state.velocity_right = (car_state.velocity_y * math.sin(math.pi / 2 - car_state.wheel_angle) 
            + car_state.velocity_x * math.sin(0 - car_state.wheel_angle))
    
        cell = TileCell(int(car_state.position.y), int(car_state.position.x))
        track_state.tile_type = int(self.field[cell.row, cell.col]['type'])
        track_state.tile_distance = int(self.field[cell.row, cell.col]['distance'])
        if track_state.tile_type == TileType.Road.value :
            track_state.tile_total_distance = self.track_info.round_distance * car_state.round_count + track_state.tile_distance  
        else:
            track_state.tile_total_distance = 0
        
        if None != car_state.last_road_position:
            last_road_cell = TileCell(int(car_state.last_road_position.y), int(car_state.last_road_position.x))
            track_state.last_road_tile_distance = int(self.field[last_road_cell.row, last_road_cell.col]['distance'])
            track_state.last_road_tile_total_distance = self.track_info.round_distance * car_state.round_count + track_state.last_road_tile_distance  

        track_state.score = (
            #car_state.round_count*100 + 
            #track_state.tile_total_distance + 
            track_state.last_road_tile_total_distance -
            car_state.timestamp / 1000)

        angles = [0, -math.pi/2, math.pi/2, -math.pi/4, math.pi/4, -math.pi*1/8, math.pi*1/8, -math.pi*3/8, math.pi*3/8]
        track_state.rays = []
        for angle in angles:
            track_state.rays.append(self.get_ray(car_state.position.x, car_state.position.y, car_state.wheel_angle, angle))  

        car_state.track_state = track_state
       
    def get_next_state(self, 
            car_config: car.CarConfig, 
            car_state: car.CarState, 
            action: car.Action, 
            debug: bool = False) -> car.CarState :
        
        if debug: 
            print('\nget_next_state() >>>')

        # Limit action by motion profile
        action_forward_acceleration = action.forward_acceleration
        if abs(action.forward_acceleration) > car_config.motion_profile.max_acceleration :
            action_forward_acceleration = (car_config.motion_profile.max_acceleration 
                * action.forward_acceleration / abs(action.forward_acceleration))
        if debug:
            print('action_forward_acceleration = ', action_forward_acceleration)

        angular_velocity = action.angular_velocity
        if abs(action.angular_velocity) > car_config.motion_profile.max_angular_velocity :
            action_forward_acceleration = (car_config.motion_profile.max_angular_velocity 
                * action.angular_velocity / abs(action.angular_velocity))
        if debug:
            print('angular_velocity = ', angular_velocity)

        # next position
        time_sec:float = 0.001 * self.track_info.time_interval
        next_position = car.Point2D(
            x = car_state.position.x + car_state.velocity_x * time_sec, 
            y = car_state.position.y + car_state.velocity_y * time_sec)
        
        if (next_position.x >= self.field.shape[1]) :
            next_position.x = self.field.shape[1] - .5
        if (next_position.x < 0) :
            next_position.x = .5
        if (next_position.y >= self.field.shape[0]) :    
            next_position.y = self.field.shape[0] - .5      
        if (next_position.y < 0) :
            next_position.y = .5
        if debug:
            print('next_position = ', next_position)

        next_cell = TileCell(int(next_position.y), int(next_position.x))
        next_cell_type = int(self.field[next_cell.row, next_cell.col]['type'])
        next_tile_distance = int(self.field[next_cell.row, next_cell.col]['distance'])
        if next_cell_type == TileType.Road.value :
            last_road_position = next_position
        else:
            last_road_position = car_state.last_road_position

        next_state = car.CarState(
            timestamp = car_state.timestamp + self.track_info.time_interval,
            wheel_angle = car_state.wheel_angle + angular_velocity * time_sec,
            position = next_position,
            last_road_position = last_road_position,
            round_count = car_state.round_count)

        # next velocity
        velocity_forward: float = (car_state.velocity_x * math.cos(0 - car_state.wheel_angle) 
            + car_state.velocity_y * math.cos(math.pi / 2 - car_state.wheel_angle))
        if debug: 
            print('velocity_forward = ', velocity_forward)

        velocity_right: float = (car_state.velocity_y * math.sin(math.pi / 2 - car_state.wheel_angle) 
            + car_state.velocity_x * math.sin(0 - car_state.wheel_angle))
        if abs(velocity_right) <= car_config.slide_friction.min_velocity_start :
            velocity_right = 0
        if debug: 
            print('velocity_right = ', velocity_right)
    
        cell = TileCell(int(car_state.position.y), int(car_state.position.x))
        cell_type = self.field[cell.row, cell.col]['type']

        tile_distance = int(self.field[cell.row, cell.col]['distance'])

        if (cell_type == TileType.Road.value 
            and next_cell_type == TileType.Road.value):
                if (self.is_start(cell) and self.is_finish(next_cell)) :
                    next_state.round_count = car_state.round_count - 1        # start to finish backward, decrease a round
                if (self.is_finish(cell) and self.is_start(next_cell)) :
                    next_state.round_count = car_state.round_count + 1        # finish to start, complete a round
        if debug: 
            print('next cell_type = ', next_cell_type, 'next_tile_distance = ', next_tile_distance)
            
            if next_state.round_count != car_state.round_count: 
                print('from cell', cell , 'Tile', self.field[cell.row, cell.col], 
                    'round_count', car_state.round_count,
                    'to cell', next_cell, 'Tile', self.field[next_cell.row, next_cell.col], 
                    'round_count', next_state.round_count)

        friction_ratio = cell_type
        if debug: 
            print('cell', cell, 'cell_type', cell_type, 'friction_ratio = ', friction_ratio)

        acceleration_forward: float = 0
        if velocity_forward != 0:
            acceleration_forward = (action_forward_acceleration 
                - car_config.rotation_friction.friction * friction_ratio)
        elif action_forward_acceleration >= car_config.rotation_friction.min_accel_start :
            acceleration_forward = (action_forward_acceleration 
                - car_config.rotation_friction.friction * friction_ratio)
        if debug: 
            print('acceleration_forward = ', acceleration_forward)

        acceleration_right:float = 0
        if abs(velocity_right) > car_config.slide_friction.min_velocity_start :
            if velocity_right > 0 :
                acceleration_right = -1 * car_config.slide_friction.friction * friction_ratio
            else :
                acceleration_right = car_config.slide_friction.friction * friction_ratio
        if debug: 
            print('acceleration_right = ', acceleration_right)
    
        next_velocity_forward = velocity_forward + acceleration_forward * time_sec
        # never rotate backward
        if next_velocity_forward < 0 :
            next_velocity_forward = 0

        if debug:
            print('before limit, next_velocity_forward = ', next_velocity_forward)
        if next_velocity_forward > car_config.motion_profile.max_velocity:
            next_velocity_forward = car_config.motion_profile.max_velocity 
        if debug: 
            print('after limit, next_velocity_forward = ', next_velocity_forward)

        next_velocity_right = velocity_right + acceleration_right * time_sec
        if (next_velocity_right * velocity_right < 0) :
            next_velocity_right = 0
        if debug: 
            print('next_velocity_right = ', next_velocity_right)

        next_state.velocity_x = (next_velocity_forward * math.cos(car_state.wheel_angle)
            + next_velocity_right * math.cos(car_state.wheel_angle + math.pi / 2))
        next_state.velocity_y = (next_velocity_forward * math.sin(car_state.wheel_angle)
            + next_velocity_right * math.sin(car_state.wheel_angle + math.pi / 2))

        if debug: 
            print('get_next_state() <<<\n')

        self.calc_track_state(next_state)
        return next_state

    
    def get_ray(self, position_x:float, position_y:float, wheel_angle:float, ray_angle:float, debug=False) -> float:

        target_angle = wheel_angle + ray_angle

        use_x = abs(math.cos(target_angle)) >= abs(math.sin(target_angle))
        if debug:
            print('position_x = ', position_x
                , ', position_y = ', position_y
                , ', target_angle = ', target_angle
                , ', use_x = ', use_x)
            
        if use_x:
            step_x = abs(math.cos(target_angle))/math.cos(target_angle)
            step_y = math.tan(target_angle)
            if debug:
                print('step_x = ', step_x, 'step_y = ', step_y)
                
            for step in range(self.track_info.column):
                x = position_x + step * step_x
                y = position_y + step * step_x * step_y

                cell = TileCell(int(y), int(x))
                if debug:
                    print('x = ', x, ', y = ', y, ', cell = ', cell)
            
                if cell.row < 0 or cell.row >= self.track_info.row or cell.col < 0 or cell.col >= self.track_info.column:
                    return 0

                tile_type = self.field[cell.row, cell.col]['type']
                if debug:
                    print('tile_type = ', tile_type)
                if tile_type == TileType.Wall.value:
                    if position_x < x :
                        x_edge = int(x)
                    else:
                        x_edge = int(x) + 1

                    y_edge = ( x_edge - position_x) /step_x * step_y + position_y
                    if int(y) <= y_edge and y_edge <= int(y) + 1:
                        if debug:
                            print('vertial: x_edge = ', x_edge, ', y_edge = ', y_edge)
                        return math.sqrt((x_edge - position_x)**2 + (y_edge - position_y)**2)
                    
                    if position_y < y :
                        y_edge = int(y)
                    else:
                        y_edge = int(y) + 1
                    x_edge = (y_edge - position_y) / step_y * step_x + position_x
                    if debug:
                        print('horizontal: x_edge = ', x_edge, ', y_edge = ', y_edge)
                    return math.sqrt((x_edge - position_x)**2 + (y_edge - position_y)**2)
        else:
            step_y = abs(math.sin(target_angle))/math.sin(target_angle)
            step_x = math.cos(target_angle) / math.sin(target_angle)

            if debug:
                print('step_x = ', step_x, 'step_y = ', step_y)
                
            for step in range(self.track_info.row):
                y = position_y + step * step_y
                x = position_x + step * step_y * step_x
            
                cell = TileCell(int(y), int(x))
                if debug:
                    print('x = ', x, ', y = ', y, ', cell = ', cell)
            
                if cell.row < 0 or cell.row >= self.track_info.row or cell.col < 0 or cell.col >= self.track_info.column:
                    return 0

                tile_type = self.field[cell.row, cell.col]['type']
                if debug:
                    print('tile_type = ', tile_type)
                if tile_type == TileType.Wall.value:
                    if position_y < y :
                        y_edge = int(y)
                    else:
                        y_edge = int(y) + 1

                    x_edge = (y_edge - position_y) * step_x + position_x
                    if debug:
                        print('x_edge = ', x_edge, ', y_edge = ', y_edge)
                    if int(x) <= x_edge and x_edge <= int(x) + 1:
                        if debug:
                            print('horizontal, use x_edge = ', x_edge, ', y_edge = ', y_edge)
                        return math.sqrt((x_edge - position_x)**2 + (y_edge - position_y)**2)
                    
                    if position_x < x :
                        x_edge = int(x)
                    else:
                        x_edge = int(x) + 1
                    y_edge = (x_edge - position_x) / step_x * step_y + position_y
                    if debug:
                        print('vertial: x_edge = ', x_edge, ', y_edge = ', y_edge)
                    return math.sqrt((x_edge - position_x)**2 + (y_edge - position_y)**2)

                        

