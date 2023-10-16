import pathlib
import multiprocessing as mp
import shutil
import numpy as np
import time
from typing import Callable as function
from lib.block_processing import calc_motion_vector_parallel_helper
from lib.utils.config import Config

class MultiProcessingNew:
    def __init__(self, config) -> None:
        self.config_class = config
        self.config = self.config_class.config
        self.manager = mp.Manager()
        self.pool = mp.Pool(mp.cpu_count())
        self.jobs = []
        
        self.__debug = self.config['debug'] if 'debug' in self.config else False

        self.write_frame_q = self.manager.Queue()

        self.signal_q = self.manager.Queue()
        self.clean_up()
        self.start()

    """
        Start watcher processes.
    """
    def start(self) -> None:
        self.block_processing_dispatcher_process = mp.Process(target=block_processing_dispatcher, args=(self.signal_q, self.write_frame_q, self.config_class,))
        self.write_frame_watcher = mp.Process(target=write_data_dispatcher, args=(self.write_frame_q, self.config_class,))
        self.block_processing_dispatcher_process.start()
        self.write_frame_watcher.start()

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
        self.block_processing_dispatcher_process.join()
        self.write_frame_q.put('kill')
        self.write_frame_watcher.join()

    """
        Clean up the output folder.
    """
    def clean_up(self) -> None:
        output_path = pathlib.Path.cwd().joinpath(self.config_class.get_output_path('main_folder'))
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for key, value in self.config_class.config['output_path'].items():
            if key == 'main_folder':
                continue
            elif key.endswith('_folder'):
                output_path.joinpath(value).mkdir(parents=True, exist_ok=True)
        return

"""
    Dispatch block processing jobs.
    read from output/reconstructed to get the reconstructed frame
    read from output/original to get the original frame
    so the execution order is guaranteed
    no parallelism here, becase we need the reconstructed frame to be written first

    Parameters:
        signal_q (mp.Queue): The queue to get signal from.
        write_data_q (mp.Queue): The queue to write to.
        config_class (Config): The config object.
"""
def block_processing_dispatcher(signal_q: mp.Queue, write_data_q: mp.Queue, config_class: Config) -> None:
    pool = mp.Pool(mp.cpu_count())

    params_i = config_class.config['params']['i']
    params_r = config_class.config['params']['r']
    counter = 0
    run_flag = True
    meta_file = pathlib.Path.cwd().joinpath(config_class.get_output_path('main_folder'), config_class.get_output_path('meta_file'))
    height, width = signal_q.get()
    while run_flag:
        file = pathlib.Path.cwd().joinpath(config_class.get_output_path('main_folder'), config_class.get_output_path('original_folder'), str(counter))
        while not file.exists():
            print("waiting for original file {} to be written".format(counter))
            time.sleep(1)
            continue
        file_bytes = file.read_bytes()
        frame_uint8 = np.frombuffer(file_bytes, dtype=np.uint8).reshape(height, width)
        frame = np.array(frame_uint8, dtype=np.int16)
        frame_index = counter
        reconstructed_path = pathlib.Path.cwd().joinpath(config_class.get_output_path('main_folder'), config_class.get_output_path('reconstructed_folder'))
        if frame_index == 0:
            prev_index = -1
            prev_frame = np.full(height*width, 128).reshape(height, width)
        else:
            prev_index = frame_index - 1
            prev_file = reconstructed_path.joinpath(str(prev_index))
            while not prev_file.exists():
                print("waiting for reconstructed file {} to be written".format(prev_index))
                time.sleep(1)
                continue
            prev_file_bytes = prev_file.read_bytes()
            prev_frame_uint8 = np.frombuffer(prev_file_bytes, dtype=np.uint8).reshape(height, width)
            prev_frame = np.array(prev_frame_uint8, dtype=np.int16)
        calc_motion_vector_parallel_helper(frame, frame_index, prev_frame, prev_index, params_i, params_r, write_data_q, reconstructed_path, pool)
        counter += 1
        if meta_file.exists():
            l = meta_file.read_text().split(',')
            last = int(l[0])
            if counter == last:
                run_flag = False
    pool.close()
    pool.join()

"""
    Write data to disk.

    Parameters:
        q (mp.Queue): The queue to read from.
        config_class (Config): The config object.
"""
def write_data_dispatcher(q: mp.Queue, config_class: Config) -> None:
    while True:
        data = q.get()
        if data == 'kill':
            break
        frame_index, mv_dump, residual_frame, average_mae = data

        mv_dump_text = ''
        for i in range(len(mv_dump)):
            for j in range(len(mv_dump[i])):
                min_motion_vector = mv_dump[i][j]
                mv_dump_text += '{} {}\n'.format(min_motion_vector[0], min_motion_vector[1])

        pathlib.Path.cwd().joinpath(config_class.get_output_path('main_folder'), config_class.get_output_path('mv_folder'), '{}'.format(frame_index)).write_text(str(mv_dump_text))

        residual_frame = np.array(residual_frame, dtype=np.int16)
        pathlib.Path.cwd().joinpath(config_class.get_output_path('main_folder'), config_class.get_output_path('residual_folder'), '{}'.format(frame_index)).write_bytes(residual_frame)

        with pathlib.Path.cwd().joinpath(config_class.get_output_path('main_folder'), config_class.get_output_path('mae_file')).open('a') as f:
            f.write("{} {}\n".format(frame_index, average_mae))
