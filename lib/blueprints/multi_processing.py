from abc import ABC, abstractmethod
import multiprocessing as mp
from typing import Callable as function
import pathlib

class MultiProcessing(ABC):

    def __init__(self, config) -> None:
        self.config_class = config
        self.config = self.config_class.config
        self.manager = mp.Manager()
        self.pool = mp.Pool(3)
        self.jobs = []
        self.q = self.manager.Queue()
        self.path = None
        self.watcher = None
        self.__debug = self.config['debug'] if 'debug' in self.config else False
        self.func = None
        self.start()

    """
        Set the path to write to.
    """
    @abstractmethod
    def set_path(self) -> None:
        pass

    """
        Write the data to the path.

        Parameters:
            path (pathlib.Path): The path to write to.
            q (mp.Queue): The queue to get data from.
    """
    @staticmethod
    @abstractmethod
    def write(path: pathlib.Path, q: mp.Queue) -> None:
        pass

    """
        Clear the path.
    """
    @abstractmethod
    def clear(self) -> None:
        pass
    
    """
        Append the data to the queue.

        Parameters:
            data (any): The data to append.
    """
    @abstractmethod
    def append(self, data) -> None:
        pass

    """
        Invoke when done
    """
    @abstractmethod
    def done(self) -> None:
        pass

    """
        Set the watcher.
    """
    def set_watcher(self) -> None:
        self.watcher = self.pool.apply_async(self.write, (self.func, self.path, self.q,))

    """
        Start the multi-processing.
    """
    def start(self):
        self.set_path()
        self.clear()
        self.set_watcher()

    """
        Dispatch a job.

        Parameters:
            func (function): The function to dispatch.
            data (tuple): The data to dispatch.
    """
    def dispatch(self, func: function, data: tuple) -> None:
        data, callback, callback_args = data
        if self.__debug:
            func(
                data, 
                callback,
                callback_args,
                self.q
            )
        else:
            job = self.pool.apply_async(func=func, args=(
                data, 
                callback,
                callback_args,
                self.q
            ))
            self.jobs.append(job)
        return
    