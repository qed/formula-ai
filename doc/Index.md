# Overview

Define key components in car racing system that will support interaction between Car, Model, Track. 

## Core

### [Car](core\car.md) 
Represent data unique to a car. At beginning, all cars can be same except its identity.
- Static data - like size, friction ratio(static, sliding, rotation), maximum (linear, angular) velocity and acceleration. 
- Runtime data 
    - Field location : (X, Y), velocity (V_x, V_y), Angle, etc.
    - Route location : number of rounds (used to track progress), time stamp.
- Action
    - Forward acceleration from power output
    - Angular velocity from steering 

### [Track](core\track.md) 
Define the characteristics of a track field.
    - size, coordinate
    - characteristic of field

### [Arena](core\arena.md) 
Combina Car and Track together to provide the physics engine. Given a car's current state, it will provide:
- CarView: what a car can see of the track environment.
- Given an Action, what is the car's state after delta time. 

### [Model](core\model.md) 
Represents the AI function. Taking in CarState, CarView, it generates an Action to drive the car.  

### [Race](race.md)
Drive interaction between Car, Model and Track, generate a race dataset.

## Mid layer

### [UI](mid\ui.md) 
Visualize a race

- Offline mode: based on saved race dataset. It can support pause, fast forward, at different speed, frame by frame etc.
- online mode: interact with car, model and track in realtime, collect data at each interaction, and update UI. Support to save a race dataset, enable to playback later in offline mode.

### [Training](mid\training.md)
Drive model training.

- Online: call model to adjust at every interaction between Car, Model and Track.
- Offline: call model to adjust with a completed race dataset.

## Up layer

### [Competition](up\competition.md)
Manage a complex competition events involve multiple cars, multiple rounds of races.

- A game is defined a race with specific combination of multiple car (with its model), and a track. 
- A game data contains race data for each car, generate result such as each car's time, position, score, ranking point etc.
- Select which set of cars will play at each game.
- decide which car to advance and final standing.


