import multiprocessing as mp
import numpy as np
import time
from typing import Callable as function
from lib.block_processing import calc_motion_vector_parallel_helper
from lib.config.config import Config
from lib.utils.quantization import quantization_matrix
from lib.utils.entropy import reording_encoding, rle_encoding, exp_golomb_encoding
from lib.utils.enums import TypeMarker
from lib.utils.misc import binstr_to_bytes

class MultiProcessingNew:
    def __init__(self, config) -> None:
        self.config = config
        self.manager = mp.Manager()
        self.pool = mp.Pool(mp.cpu_count())
        self.jobs = []
        
        self.__debug = self.config.debug

        self.write_frame_q = self.manager.Queue()

        self.signal_q = self.manager.Queue()
        self.start()

    """
        Start watcher processes.
    """
    def start(self) -> None:
        self.block_processing_dispatcher_process = mp.Process(target=block_processing_dispatcher, args=(self.signal_q, self.write_frame_q, self.config,))
        self.write_frame_watcher = mp.Process(target=write_data_dispatcher, args=(self.write_frame_q, self.config,))
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
    Dispatch block processing jobs.
    read from output/reconstructed to get the reconstructed frame
    read from output/original to get the original frame
    so the execution order is guaranteed
    no parallelism here, becase we need the reconstructed frame to be written first

    Parameters:
        signal_q (mp.Queue): The queue to get signal from.
        write_data_q (mp.Queue): The queue to write to.
        config (Config): The config object.
"""
def block_processing_dispatcher(signal_q: mp.Queue, write_data_q: mp.Queue, config: Config) -> None:
    pool = mp.Pool(mp.cpu_count())

    q_matrix = quantization_matrix(config.params.i, config.params.qp)
    counter = 0
    run_flag = True
    meta_file = config.output_path.meta_file
    height, width = signal_q.get()
    while run_flag:
        file = config.output_path.original_folder.joinpath(str(counter))
        while not file.exists():
            print("waiting for original file {} to be written".format(counter))
            time.sleep(1)
            continue
        file_bytes = file.read_bytes()
        frame_uint8 = np.frombuffer(file_bytes, dtype=np.uint8).reshape(height, width)
        frame = np.array(frame_uint8, dtype=np.int16)
        frame_index = counter
        reconstructed_path = config.output_path.reconstructed_folder
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
        calc_motion_vector_parallel_helper(frame, frame_index, prev_frame, prev_index, config.params, q_matrix, write_data_q, reconstructed_path, pool)
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
        config (Config): The config object.
"""
def write_data_dispatcher(q: mp.Queue, config: Config) -> None:
    while True:
        data = q.get()
        if data == 'kill':
            break
        frame_index, mv_dump, qtc_dump, average_mae = data

        mv_dump_text = ''
        is_intraframe = mv_dump[0]
        if is_intraframe:
            mv_dump_text += '{}'.format(TypeMarker.I_FRAME.value)
        else:
            mv_dump_text += '{}'.format(TypeMarker.P_FRAME.value)
        for object in mv_dump[1]:
            for item in object:
                min_motion_vector = item
                if is_intraframe:
                    mv_dump_text += '{}'.format(exp_golomb_encoding(min_motion_vector))
                else:
                    mv_dump_text += '{}{}'.format(exp_golomb_encoding(min_motion_vector[0]), exp_golomb_encoding(min_motion_vector[1]))

        qtc_dump_text = ''
        for object in qtc_dump:
            for item in object:
                qtc_dump_text += ''.join(exp_golomb_encoding(x) for x in rle_encoding(reording_encoding(item)))

        mv_dump_bytes = binstr_to_bytes(mv_dump_text)
        qtc_dump_bytes = binstr_to_bytes(qtc_dump_text)

        config.output_path.mv_folder.joinpath('{}'.format(frame_index)).write_bytes(mv_dump_bytes)

        config.output_path.residual_folder.joinpath('{}'.format(frame_index)).write_bytes(qtc_dump_bytes)

        with config.output_path.mae_file.open('a') as f:
            f.write("{} {}\n".format(frame_index, average_mae))
