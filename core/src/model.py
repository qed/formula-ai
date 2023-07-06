from abc import ABC, abstractmethod
from . import car
from . import track


class IModelLoad(ABC):
    @abstractmethod
    def load(self, folder:str) -> bool:
        """
        Loads the model from a folder
        """
        raise NotImplementedError
    

class IModelSave(ABC):
    @abstractmethod
    def save(self, folder:str) -> bool:
        """
        Saves the model into a folder
        """
        raise NotImplementedError


class IModelInference(IModelLoad):
    pass


class IModelTrain(IModelLoad, IModelSave):
    pass

    
"""
    # update model after each step
class IModelTrainOnline(IModelLoad, IModelSave):
    def update_online(self, start: car.CarState, action: car.Action, end: car.CarState) -> bool:
        raise NotImplementedError

    # update model after complete race
class IModelTrainOffline(IModelLoad, IModelSave):
    pass


    def update_offline(self, dataset: RaceDataset) -> bool:

        For offline training, tun with dataset from a complete race

        raise NotImplementedError
"""
