from abc import ABC, abstractmethod

class ConfigObject(ABC):

    def __init__(self, config):
        self.config = config
        self.__to_object()
        self.validate()

    def __to_object(self):
        for key, value in self.config.items():
            setattr(self, key, value)

    @abstractmethod
    def validate(self):
        pass