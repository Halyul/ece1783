import pathlib
import multiprocessing as mp
import numpy as np
from typing import Callable as function
from lib.block_processing import calc_motion_vector_helper

# TODO: better mp solution

class MultiProcessingNew:
    def __init__(self, config) -> None:
        self.config_class = config
        self.config = self.config_class.config
        self.manager = mp.Manager()
        self.pool = mp.Pool(mp.cpu_count() + 2)
        self.jobs = []
        
        self.__debug = self.config['debug'] if 'debug' in self.config else False

        self.write_frame_q = self.manager.Queue()
        self.write_frame_watcher = self.pool.apply_async(write_data_dispatcher, (self.write_frame_q,))

        self.og_frame_q = self.manager.Queue()
        self.reconstructed_frame_q = self.manager.Queue()
        self.block_processing_dispatcher_process = self.pool.apply_async(block_processing_dispatcher, (self.og_frame_q, self.reconstructed_frame_q, self.write_frame_q, self.config,))


    """
        Dispatch a job.

        Parameters:
            func (function): The function to dispatch.
            data (tuple): The data to dispatch.
    """
    def dispatch(self, func: function, data: tuple) -> None:
        data, q = data
        if self.__debug:
            func(
                data, 
                q,
            )
        else:
            job = self.pool.apply_async(func=func, args=(
                data, 
                q,
            ))
            self.jobs.append(job)
        return
    
    """
        
    """
    def done(self):
        for job in self.jobs: 
            job.get()

        self.block_processing_dispatcher_process.get()
        self.og_frame_q.put('kill')
        self.reconstructed_frame_q.put('kill')
        self.write_frame_q.put('kill')
        self.pool.close()
        self.pool.join()


def block_processing_dispatcher(og_frame_q, reconstructed_frame_q, write_data_q, config):
    params_i = config['params']['i']
    params_r = config['params']['r']
    while True:
        data = og_frame_q.get()
        if data == 'kill':
            break
        frame, frame_index = data
        height, width, _ = frame.shape
        if frame_index == 0:
            reconstructed_frame = np.full(height*width, 128).reshape(height, width)
        else:
            prev_index, reconstructed_frame = reconstructed_frame_q.get()
        calc_motion_vector_helper(frame, frame_index, reconstructed_frame, params_i, params_r, reconstructed_frame_q, write_data_q)

def write_data_dispatcher(q):
    while True:
        data = q.get()
        if data == 'kill':
            break
        frame_index, mv_dump, residual_frame, current_reconstructed_frame, og_y_component = data
        with pathlib.Path.cwd().joinpath('output', 'mv{}.txt'.format(frame_index)).open("w") as f:
            f.write(str(mv_dump))
        with pathlib.Path.cwd().joinpath('output', 'residual{}'.format(frame_index)).open("wb") as f:
            residual_frame = np.array(residual_frame, dtype=np.int8)
            f.write(residual_frame)
        with pathlib.Path.cwd().joinpath('output', 'reconstructed{}'.format(frame_index)).open("wb") as f:
            current_reconstructed_frame = np.array(current_reconstructed_frame, dtype=np.uint8)
            f.write(current_reconstructed_frame)
        with pathlib.Path.cwd().joinpath('output', 'original{}'.format(frame_index)).open("wb") as f:
            og_y_component = np.array(og_y_component, dtype=np.uint8)
            f.write(og_y_component)