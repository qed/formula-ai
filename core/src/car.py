

class Point2D :  
    def __init__(self, x:float = 0, y:float = 0):
        self.type = 'Point2D'
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return f'Point2D(x={self.x}, y={self.y})'


class RotationFriction:
    def __init__(self, min_accel_start:float = 0, friction:float = 0):
        self.type = 'RotationFriction'
        self.min_accel_start = min_accel_start  # The minimum acceleration needed to start the car
        self.friction = friction                # The friction coefficient of the car
    
    def __str__(self) -> str:
        return f'RotationFriction(min_accel_start={self.min_accel_start}, friction={self.friction})'


class SlideFriction:
    def __init__(self, min_velocity_start=0, friction=0):
        self.type = 'SlideFriction' 
        self.min_velocity_start = min_velocity_start    # Minimum velocity need to start slide sideway
        self.friction = friction                    # Friction coefficient when sliding sideway

    def __str__(self) -> str:
        return f'SlideFriction(min_velocity_start={self.min_velocity_start}, friction={self.friction})'
    

class MotionProfile:
    def __init__(self, 
            max_acceleration = 0, 
            max_velocity = 0,
            max_angular_velocity = 0):
        self.type = 'MotionProfile'
        self.max_acceleration = max_acceleration    # max power produced acceleration in wheel forward direction m/s/s
        self.max_velocity = max_velocity            # wheel forward direction maximum velocity m/s
        self.max_angular_velocity = max_angular_velocity    # radian/sec

    def __str__(self) -> str:
        return f'MotionProfile(max_acceleration={self.max_acceleration}, max_velocity={self.max_velocity}, max_angular_velocity={self.max_angular_velocity})'



class CarConfig:
    def __init__(self, 
            rotation_friction = RotationFriction(), 
            slide_friction = SlideFriction(),
            motion_profile = MotionProfile()):
        self.type = 'CarConfig'
        self.rotation_friction = rotation_friction  # Rotational friction parameters
        self.slide_friction = slide_friction        # Slide friction parameters
        self.motion_profile = motion_profile        # Motion profile parameters

    def __str__(self) -> str:
        return f'CarConfig(rotation_friction={self.rotation_friction}, slide_friction={self.slide_friction}, motion_profile={self.motion_profile})'
    

class CarInfo:
    def __init__(self, id = 0, team = '', city = '', state = '', region = ''):
        self.type = 'CarInfo'
        self.id = id
        self.team = team
        self.city = city
        self.state = state
        self.region = region

    def __str__(self) -> str:
        return f'CarInfo(id={self.id}, team={self.team}, city={self.city}, state={self.state}, region={self.region})'
    


class TrackState:
    def __init__(self, 
            velocity_forward: float = 0,    
            velocity_right: float = 0,       
            rays:list[float] = [],           
            tile_type:int = 0,
            tile_distance:int = 0,
            tile_total_distance:int = 0,
            last_road_tile_distance:int = 0,
            last_road_tile_total_distance:int = 0,
            score:float = 0):

        self.type = 'TrackState'
        self.velocity_forward = velocity_forward  # velocity in the forward direction of the car
        self.velocity_right = velocity_right # velocity in the right direction of the car
        self.rays = rays                # list of ray distance
        self.tile_type = tile_type      # tile type of the Tile it is on
        self.tile_distance = tile_distance     # distance from the start of the track, in unit of tile count
        self.tile_total_distance = tile_total_distance   # round_count * TrackInfo.total + tile_distance, 0 if tile is not road
        self.last_road_tile_distance = last_road_tile_distance  # tile_distance of the last road tile
        self.last_road_tile_total_distance = last_road_tile_total_distance  # tile_total_distance of the last road tile
        self.score = score              # score of the track state


    def __str__(self) -> str:
        return f'TrackState(velocity_forward={self.velocity_forward}, velocity_right={self.velocity_right}, rays={self.rays}, tile_type={self.tile_type}, tile_distance={self.tile_distance}, tile_total_distance={self.tile_total_distance}, last_road_tile_distance={self.last_road_tile_distance}, last_road_tile_total_distance={self.last_road_tile_total_distance}, score={self.score})'
    

class CarState:
    def __init__(self, 
            timestamp:int = 0, 
            wheel_angle:float = 0, 
            velocity_x:float = 0, 
            velocity_y :float = 0,        
            position = Point2D(),
            round_count:int = 0,
            last_road_position = None,
            track_state: TrackState = None):

        self.type = 'CarState'
        self.timestamp = timestamp  # Milliseconds since race start

        self.wheel_angle = wheel_angle  # whee angle, radian
        self.velocity_x = velocity_x    # m/s
        self.velocity_y = velocity_y    # m/s

        self.position = position    # (x,y)
        self.last_road_position = last_road_position # last road position before off Road tile, as last progress
        self.round_count = round_count  # full track round completed

        self.track_state = track_state  # car state calculated track data
    
    def __str__(self) -> str:
        return f'CarState(timestamp={self.timestamp}, wheel_angle={self.wheel_angle}, velocity_x={self.velocity_x}, velocity_y={self.velocity_y}, position={self.position}, round_count={self.round_count}, last_road_position={self.last_road_position}, track_state={self.track_state})'



class Action:

    def __init__(self, forward_acceleration=0, angular_velocity=0):
        self.type = 'Action'
        self.forward_acceleration = forward_acceleration # wheel forward acceleration
        self.angular_velocity = angular_velocity         # wheel angle change rate

    def __str__(self) -> str:
        return f'Action(forward_acceleration={self.forward_acceleration}, angular_velocity={self.angular_velocity})'
