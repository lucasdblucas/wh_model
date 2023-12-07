from abc import ABC, abstractmethod
from configs.config import Config
from torch.utils.data import DataLoader

class ExperimentInterface(ABC):
    
    @abstractmethod
    def define_data(config: Config):
        pass
    
    @abstractmethod
    def define_model_and_infer(config: Config, infe_loader: DataLoader):
        pass
    
    @abstractmethod
    def define_model_and_train(config: Config, infe_loader: DataLoader):
        pass