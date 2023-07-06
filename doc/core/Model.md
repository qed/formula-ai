# Model

Model represents the AI function. Input CarState, CarView, output Action to drive a car.

Model class capture the data and function of AI system. Each model have its own code, and training generated model parameter dataset. 

During a race, a model is used in inference mode. It loads from saved model data, generate `Action` in responding to `CarState` and `CarView`.

In training mode, a model adjusts internal parameter dataset, based on `CarState` and `CarView`. After trainging, a model saves paramter dataset for later use in inference or training.



## Inference 


``` 
    Action get_action(self, CarState, CarView)
```


## Train

### Online
```
    bool Update(self, start_car_state, action, result_car_state)
```

### Offline
```
    bool Update(self, race_dataset)
```

## Implementation

Internal data and code logic across different models may be very different. A model may use only part of data it sees, while ignore other part.

Each model should contains a function to convert the input data into the format of internal usage. For example, it may construct a tensor, each field value extract from a input data field.

Model data saving should consider
- Easy to debug

    Write a text file, easy to read in text editor.

- Easy to load/save by other tools.
    
    Use standard format like csv, json, xml etc.

- Make it simple

    If model data structure is complex, save different part into its own file, like:
    
    - Save a matrix into csv file, can use Excel to view it.
    - Save complex data using json file.

- Robust and compatible

    Model data structure may change overtime. When design and change model data, shoule make new data not break old code, and new code not break on old data. 
    
    For row based data:
        - Do not delete an old column. Can make the old data column empty, or minimum sized valid value.
        - Add new column at end. 

    For data with named fields:
        - Add new field with new name.
        - Do not delete a field required by old code.
