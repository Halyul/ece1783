import multiprocessing as mp
import numpy as np
import time
from typing import Callable as function
from lib.block_processing import calc_motion_vector_parallel_helper
from lib.config.config import Config
from lib.components.frame import Frame
from lib.components.qtc import quantization_matrix

class MultiProcessingNew:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.manager = mp.Manager()
        self.pool = mp.Pool(mp.cpu_count())
        self.jobs = []
        self.__debug = self.config.debug
        self.signal_q = self.manager.Queue()
        self.start()

    """
        Start watcher processes.
    """
    def start(self) -> None:
        self.block_processing_dispatcher_process = mp.Process(target=block_processing_dispatcher, args=(self.signal_q, self.config,))
        self.block_processing_dispatcher_process.start()

    """
        Dispatch a job.

        Parameters:
            func (function): The function to dispatch.
            data (tuple): The data to dispatch.
    """
    def dispatch(self, func: function, data: tuple) -> None:
        data, config = data
        if self.__debug:
            func(
                data, 
                config,
            )
        else:
            job = self.pool.apply_async(func=func, args=(
                data, 
                config,
            ))
            self.jobs.append(job)
        return
    
    """
        Wait for all jobs to finish. 
    """
    def done(self) -> None:
        for job in self.jobs: 
            job.get()

        self.pool.close()
        self.pool.join()
        self.signal_q.put((0, 0))
        self.block_processing_dispatcher_process.join()

"""
    Dispatch block processing jobs.
    read from output/reconstructed to get the reconstructed frame
    read from output/original to get the original frame
    so the execution order is guaranteed
    no parallelism here, becase we need the reconstructed frame to be written first

    Parameters:
        signal_q (mp.Queue): The queue to get signal from.
        config (Config): The config object.
"""
def block_processing_dispatcher(signal_q: mp.Queue, config: Config) -> None:
    pool = mp.Pool(mp.cpu_count())

    q_matrix = quantization_matrix(config.params.i, config.params.qp)
    counter = 0
    run_flag = True
    meta_file = config.output_path.meta_file
    height, width = signal_q.get()
    if height == 0 or width == 0:
        pool.close()
        pool.join()
        return
    jobs = []
    while run_flag:
        file = config.output_path.original_folder.joinpath(str(counter))
        while not file.exists():
            print("Waiting for original file {} to be written".format(counter))
            time.sleep(1)
            continue
        frame = Frame(counter, height, width, params_i=config.params.i, is_intraframe=counter % config.params.i_period == 0)
        frame.read_from_file(file)
        frame.convert_type(np.int16)
        reconstructed_path = config.output_path.reconstructed_folder
        if frame.index != 0:
            prev_index = frame.index - 1
            prev_file = reconstructed_path.joinpath(str(prev_index))
            while not prev_file.exists():
                print("Waiting for reconstructed file {} to be written".format(prev_index))
                time.sleep(1)
                continue
            frame.read_prev_from_file(prev_file, prev_index)
            frame.prev.convert_type(np.int16)
        index, mv_dump, qtc_block_dump = calc_motion_vector_parallel_helper(frame, config.params.r, q_matrix, reconstructed_path, pool)
        job = pool.apply_async(func=write_data_dispatcher, args=((index, mv_dump, qtc_block_dump), config,))
        jobs.append(job)
        counter += 1
        if meta_file.exists():
            l = meta_file.read_text().split(',')
            last = int(l[0])
            if counter == last:
                run_flag = False

    for job in jobs: 
        job.get()

    pool.close()
    pool.join()

"""
    Write data to disk.

    Parameters:
        data (tuple): The data to write.
        config (Config): The config object.
"""
def write_data_dispatcher(data: tuple, config: Config) -> None:
    frame_index, mv_frame, qtc_frame = data
    print("Dumping", frame_index)

    config.output_path.mv_folder.joinpath('{}'.format(frame_index)).write_bytes(mv_frame.tobytes())

    config.output_path.residual_folder.joinpath('{}'.format(frame_index)).write_bytes(qtc_frame.tobytes())

    with config.output_path.mae_file.open('a') as f:
        f.write("{} {}\n".format(frame_index, mv_frame.average_mae()))
