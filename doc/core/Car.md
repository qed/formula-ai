# Car

Car define the characteristics of a car, and all related data used in running.

## CarConfig
CarConfig class capture car's characteristic properties not change over time. Same CarConfig value maybe used by multiple cars.

Car is modeled as unicycle, with mass on a single wheel. Only one point of its wheel touches ground. 

If we want to consider car collision, we can model each car as the cirle with 1 meter radius. It will be easy to detact collision, and calculate after collision CarState based on reservation of moment.

### Mass
At beginning we do not specify mass value. We can assume all cars has same mass. Other configuration values are specified as ratio, or provided as acceleration. Therefore we will not need consider mass in calculating car movement. 

Later we may model a more complex car with different mass, wheel's performance, fuel capacity etc. With different combination, may require change tire or refuel to complete a race.

### Wheel

The wheel rotate around axel fixed on car body. Wheel direction can be moved to any direction. 
A wheel movement can be:
-   Rotate along wheel angle direction. 

    Need a start acceleration to overcome static friction. Once start rotation, the rotational friction act as a negative acceleration. Car output power minuse rotational friction create acceleration in this direction. Car velocity along this direction has an upper limit. Once reached, any more power will cause tire to slide, not increase velocity. 

    `forward_velocity` measure velocity in wheel heading direction.

-   Slide at perpendicular to wheel angle direction

    Velocity projected on this direction determine whether it stay static or slide. 
    
    When velocity projected is less than the threshold, there is no sliding. Any velocity projected  become 0 instantly, generate no movement in this direction. 

    When velocity projected is more than the threshold, wheel slides. Friction generates a negative acceleration to reduce velocity projected value.

    `slide_velocity` measure sliding velocity in right-hand direction.

On track field, the friction caused acceleration is calculated as multiplication of tile's FrictionRatio and the wheel friction. 

A wheel can only rotate forward only, never backward. 

A wheel can have negative acceleration like applying brake to slow down. 

To move backward, wheel can steer by `math.pi` then rotate forward.

```
RotationFriction
    float minimumAccelerationToStart
    float friction
    float maxVelocity


SlideFriction
    float minimumVelocityToStart
    float friction

CarConfig
    RotationFriction rotationFriction
    SlideFriction slideFriction

```

## CarInfo
CarInfo class include a specific car's related info. 

```
CarInfo
    UInt16 CarId
    String Team
    String City
    String State
    String Region
```

## CarState
CarState class capture car's runtime data at a specific moment.

```
CarState
    Int32 timeStamp 
    ; miliseconds elapsed since race started

    float forward_velocity  # m/s
    float slide_velocity    # m/s

    Point position          # Point(X,Y)
    float angle             # radian
    
    Int32 trackDistance; // same value as the tile the Car is on.    
 
```
