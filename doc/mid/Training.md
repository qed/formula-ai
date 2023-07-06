# Training

Training system drive model training use race data. It call model's IModelTrain API to update model.

As each race TurnData include time stamp and distance, model optimize to maximize distance at each turn, or minimize time stamp for reaching the target distance.

## Online 

Online training allow model to update itself at each turn of interaction

Training system will drive Race system in interactive mode, obtain TurnData and feed into model for updating. 


## Offline mode

Offline training feed model with complete race dataset for updating. Model internal can decide how to use the dataset, using micro-batch updating, or only 1 update using summaries data. 


## TrainingSystem


```
Training
    bool SetModel()

    Load Model()

    bool SetupRace()

    bool Train()

    bool SaveModel(string folderPath)
```

Training system may run multiple epoch. 

Optional function:
- streaming data to allow observation of progress, such as how long it take the model to finish a round.
 