# Arena

Arena combine Car and Track, act as the physics engine.

```
Arena

public :
    CarView GetCarView(CarState) 
    ; return sub section of TraveField visible to car at position of CarState.

    CarState GetNextState(CarState, Action, timeInterval)
    ; return the CarState using Action after timeInterval.

    const Shape FieldShape
    ; return (Width, Height) of whole TrackField. 
    ; could help Model if it want to build the whole TrackField.
    
private :
    ; physical engine function, should be stable. 
    CarState GetNextState(CarState, Action, timeInterval, CarConfig, TileConfig)

    TrackField trackField;
```


# Future expansion

## Noisy phyical world

We can model imperfection of a track field by adding a noise to individual tile. The noise is defined an error on a tile's fraction ratio. Noise distribution should follow standard distribution, center at 0. Most tiles do not differ much from its ideal case. 

Combination of TrackField and NoiseSet identify a race setting. 

Including noice into environment will make each race unique. It help to test AI model's robustness, as well as add some unpredictability.

A multiple round competition can be more interesting, even using same TrackField, with different NoiseSet. 

## Mutli-car interaction 
To support multiple car interaction in a race, we need to expand

- CarView

    Add visible cars' CarState

- GetNextState

    Physical engine need to consider collision between cars to determine next state.

