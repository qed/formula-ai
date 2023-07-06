# Environment Framework

Testing the performance of AI on a racetrack environment, using various
different techniques including reinforcement learning and genetic algorithms.

## Prerequisites:

Install Python: https://www.python.org/downloads/

Install Anaconda: https://www.anaconda.com/download/

## Setup

It's recommended you use a Python venv for this project. Run the following
commands (assuming Anaconda):

```
conda create -n car-test python=3.9
conda activate car-test
pip install -r requirements.txt
```

to create and activate the conda environment and install all requirements. Note
`python=3.9` is needed because some packages do not work with `python>=3.11`.


In vscode, install "Python" extension from Microsoft.


## Notes on the environment

folder structure:

- core:

    framework, unit test, and sample track field, race setup

- algo: 
    
    sample AI algorithms. model.py for inference, train.py for training
    
- runner: 
    
    3rd party model testing folder. To verify your model, 

    - Copy your algo model.py and data files into racecar sub folder
    - run `runner.py`, you will see output like

        ```
        Race finished, saved at c:\git\Environment-Framework\runner\data\race\Race5_multi_turn_large_ppo-hc_20230706_123522
        ```

    - run `raceviewer.py`, select above data folder `..\Race5_multi_turn_large_ppo-hc_20230706_123522`, click `ok` button



