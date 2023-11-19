from abc import ABC, abstractmethod

class ConfigObject(ABC):

    def __init__(self, config):
        self.config = config
        self.__to_object()
        self.validate()

    def __to_object(self):
        """
            Convert the config to an object.
        """
        for key, value in self.config.items():
            setattr(self, key, value)

    @abstractmethod
    def validate(self):
        pass

    def add(self, key, value):
        """
            Add a new key-value pair to the config.
        """
        if key not in self.config:
            self.config[key] = value
            setattr(self, key, value)
        else:
            raise Exception('Key already exists.')