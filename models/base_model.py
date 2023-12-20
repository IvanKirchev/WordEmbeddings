from abc import ABC, abstractmethod
import yaml
import os

class Config:
    def __init__(self, dataset_path, learning_rate, epochs, window_size, embedding_dim, num_ns, batch_size=None, vocab_size=None, sequence_length=None):
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.num_ns = num_ns
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

class BaseModel(ABC):
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):

        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        config_obj = Config(
            dataset_path=config_data.get('dataset_path'),
            learning_rate=config_data.get('learning_rate'),
            epochs=config_data.get('epochs'),
            window_size=config_data.get('window_size'),
            embedding_dim=config_data.get('embedding_dim'),
            num_ns=config_data.get('num_ns'),
            batch_size=config_data.get('batch_size'),
            vocab_size=config_data.get('vocab_size'),
            sequence_length=config_data.get('sequence_length')
        )
        return config_obj
    
    @classmethod
    @abstractmethod
    def load_data():
        pass

    @classmethod
    @abstractmethod
    def train():
        pass

    @classmethod
    @abstractmethod
    def eval():
        pass