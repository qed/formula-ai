# Track

Track system define the characteristics of a racing track. It act as the physical engine. 

TrackField capture a complete racing track, includes path, shoulder, wall, etc. 

A track field is defined as a rectangular area consist of equal sized tiles. Each tile has its own property.


## Design Goal

The track field be able to handle any possible scenario for cars racing in it.

- A car should never run out of the TrackField, even they want and try to.
- No matter where the car is, it can always move, possible to go back on track.
- TrackField data should be reasonable, though not always accurate, for a realistic racing track.
- It is possible to increase its resolution, to be closer to real world.

## Coordinate

When showing on screen, the upper left corner is at (0, 0). X direct from left to right. Y direct from up to down. 

A position in field is defined as a Point2D with positive float value. 

A field of size 2000 meter wide and 500 meter height has
- upper left corner (0, 0)
- upper right corner (0, 2000)
- lower left corner (500, 0)
- lower right corner (500, 2000).


Angle is meansured in radian, use right direction as 0, increase in clockwise. 
- right direction is 0, 2*PI
- down direction is PI/2
- left direction is PI
- up direction is 3/2*PI


## Field Tile

Each tile has its own tile type, which define its physical characteristics such as the fraction ratio. Initial set of tiles include:

| Type | TypeId | FrictionRatio |
| --- | ---: | ---: |
| Road | 0 | 1 | 
| Shoulder | 1 | 3 | 
| Wall | 2 | 100 | 

A car can run well on road, slower on shoulder, and quickly stop on wall. 

The effective friction coefficient is:

```
FrictionCoefficient =  FrictionRatio * CarConfig.rotationFriction.friction 
```

## TrackDistance

TrackField may contain complex winding route with twists and turns. We want to assign a number for each FieldTile, to indicate its effective distance from starting line.  Using this number, a car can know how much effective distance it has covered. It can also help to choose which direction to move next.  
- Moving into a tile with a larger number, it is getting closer toward the target line. 
- Moving into a tile with a smaller number, it is moving backward, getting further from the target line.
- Moving into a tile with equal number, it is moving sideways, stay same distance from starting line and toward the target line.

Each track has a starting line. Each tile on the starting line have TrackDistance value 0. When race started, a tile is not on starting line and next to a starting line tile in the forward direction have TrackDistance value 1, a tile is not on starting line and next to a starting line tile in the backward direction have TrackDistance value -1.

We define a tile's TrackDistance as the minimum number of tiles a car have to move into from starting line, toward the target direction. Moving in forward direction, TrackDistance of traveled tiles should not decrease. When completed a full route and reach a tile on the starting line, its TrackDistance should equal the RoundDistance, defined as the minimum number of tiles to cover to complete a closed route. Keep moving forward will increase TrackDistance. Each track has a sequence of tiles, that each tile overlap with the next tile by an edge or corner point, complete the whole route, and have the smallest number of tiles. We call this set of tiles the shortest path. Its number of tiles equal to the track's RoundDistance. 

For each FieldTile of type Road, We can calculate it TrackDistance as:
- Initialize all FieldTile's value as -1, indicate unknown.
- Set all Road tiles on starting line's value as 0, set all Road tiles next to starting line tiles on the forward direction's value as 1
- Set Key = 1
- Loop
    From each tile with value = Key, check each next tile it touches (have overlapped edge or corner point). 
        If next tile's value == Key - 1, do nothing
        If next tile's value == -1, set its value to Key+1

    Key++


```
Tile
    TypeId  typeId

    UInt32  trackDistance    
```

## TrackField

TrackField class
```
TrackField:

    Tile Field[Width, Height];

```
Row and column index start with 0. 

A small TrackField of 6 by 11 meter consist of tile size of 1 by 1 meter, each tile is represent as {TypeId, TrackDistance} can be:

```
{
    "Field" : 
    {
        {{2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}},
        {{1,0}, {1,0}, {1,0}, {1,0}, {1,0}, {1,0}, {1,0}, {0,7}, {1,0}, {1,0}, {1,0}},
        {{0,0}, {0,1}, {0,2}, {0,3}, {0,4}, {0,5}, {0,6}, {0,7}, {0,8}, {0,9}, {0,10}},
        {{0,0}, {0,1}, {0,2}, {1,0}, {0,4}, {0,5}, {0,6}, {0,7}, {0,8}, {0,9}, {0,10}},
        {{1,0}, {1,0}, {1,0}, {1,0}, {1,0}, {1,0}, {1,0}, {0,7}, {1,0}, {1,0}, {1,0}},
        {{2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}, {2,0}},
    }
}
```

In this track, top and bottom rows are wall. Logical route go through the 2 row, start at ```[2,0]```, end at ```[2,10]```. Track distance is 10. there is a Shoulder tile at ```[3,3]```. Tile ```[3.4]``` have TrackDistance value 4, because a car can run through ```[3,0], [3,1], [3,2], [2,3], [3,4]``` cover distance of 4. 


This track only have a straight track, does not support multiple round race. 


# CarView
Car has a limited view of the track field. Depending the car's position, only a subset of the track field around the car is visible. 

```
CarView:
    Point UpperLeft;
    Tile Field[] ; a sub section of the track field 

```

We can define how far the can can see. Assume we allow a car to see two tiles from current tile on all 4 direction. When the car is on tile ```[2,3]```, the CarView is:

```
   {
    "UpperLeft" : 
        {
            X : 0,
            Y : 1,
        }
    "Field" : 
    {
        {{2,0}, {2,0}, {2,0}, {2,0}, {2,0}, },
        {{1,0}, {1,0}, {1,0}, {1,0}, {1,0}, },
        {{0,1}, {0,2}, {0,3}, {0,4}, {0,5}, },
        {{0,1}, {0,2}, {1,0}, {0,4}, {0,5}, },
        {{1,0}, {1,0}, {1,0}, {1,0}, {1,0}, },
    }
} 
```

