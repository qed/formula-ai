# Race

Race system drive interaction between Track system and one or multiple cars, generate a race dataset.

A car data include car id, car config, and model config it used. 


A race can be interactive. The client will start it, and advance turn by turn, get update data at each turn until it finishes.

A race can also be automatic. Once started, it will finish and output all turn data.

## Data types

### TurnData 

TurnData capture detail data for single round of interaction between a car, its car model and the Track.

``` 
TurnData
    UInt32 timeStamp
    CarState carState
    Action  action

```

During each turn, CarView data is generated and used. We do not capture it in TurnData to save space. CarView can be easily reproduced using Track system with CarState for the turn.

### CarTrace

CarTrace capture sequece of TurnData for a car in a run. It capture all turns in a whole run. First TurnData state should be the car stopped at the starting point.  Last TurnData should be the car stopped, and model output is ```(0,0)```.  .

CarTrace also capture all car related data. 

``` 
CarTrace
    UInt32 carId
    CarConfig   config
    Model       model ; AI model used
    Vector<TurnData>    carTrace
```

### RaceData

RaceData record all related information for a complete race. It should capture all data necessary to replay. It support render the progress in UI, without rerun. Multiple cars may race together. 

``` 
RaceData
    DateTime startTime; local data and time when run start

    TrackConfig trackConfig;

    Vector<CarTrace> 
   

```




## RaceSystem

RaceSystem setup a race with necessary config data, carry out the interaction between track and each car, and record race related runtime data.

We can use 20 milisecond as default turn time interval. We can use other value for debugging or training.

``` 
RaceSystem

    bool SetTrack(TrackSystem trackSystem);
    bool SetTurnInterval(UInt16 timeIntervalMsec)

    bool AddCar(carId, CarConfig, Model)

    void Prepare();
    // Finish all initialization operations. 

    // interactive
    bool Step(); // return whether run finished
    TurnData GetLastTurnData(UInt32 carId);

    bool RunToFinish(); auto run till all cars finish

    TurnData GetTurnData(UInt32 carId, UInt32 timeStamp); 
    // return latest turn data up to timeStamp

    bool SaveData(string folderPath); 
    // save race data into a folder as a collection of files.
    // Track System should have its own file. 
    // Each car have its own folder, named by carId.

```
