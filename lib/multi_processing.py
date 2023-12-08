import multiprocessing as mp
import numpy as np
import time
from typing import Callable as function
from lib.block_processing import processing, processing_mode3
from lib.config.config import Config
from lib.components.frame import Frame
from lib.components.qtc import quantization_matrix
from qp_bitcount import CIF_bitcount_perRow_i, QCIF_bitcount_perRow_i

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
    counter_i = 0
    counter_p = 0
    run_flag = True
    meta_file = config.output_path.meta_file
    stop_at = config.params.stop_at
    height, width = signal_q.get()
    if height == 0 or width == 0:
        pool.close()
        pool.join()
        return
    jobs = []
    prev_frame = Frame(-1, height, width, params_i=config.params.i, data=np.full(height*width, 128).reshape(height, width))
    split_counters = []
    total_bitcount_per_row_i = 0
    total_bitcount_per_row_p = 0
    if height == 288 and width == 352:
        table = CIF_bitcount_perRow_i
    elif height == 144 and width == 176:
        table = QCIF_bitcount_perRow_i
    if config.params.ParallelMode == 3:
        data_queues = []
        while run_flag:
            file = config.output_path.original_folder.joinpath(str(counter))
            while not file.exists():
                print("Waiting for original file {} to be written".format(counter))
                time.sleep(1)
                continue
            is_intraframe=counter % config.params.i_period == 0
            frame = Frame(counter, height, width, params_i=config.params.i, is_intraframe=is_intraframe)
            frame.read_from_file(file)
            frame.convert_type(np.int16)
            reconstructed_path = config.output_path.reconstructed_folder
            if not frame.is_intraframe:
                prev_data_queue = data_queues[-1]
            else:
                prev_data_queue = None
            if (counter + 1) % config.params.i_period == 0 or counter == stop_at - 1:
                next_data_queue = None
            else:
                next_data_queue = mp.Queue()
            job = mp.Process(target=processing_mode3, args=(
                    frame,
                    config,
                    q_matrix,
                    reconstructed_path,
                    prev_data_queue,
                    next_data_queue,
                    write_data_dispatcher,
            ))
            data_queues.append(next_data_queue)
            job.start()
            jobs.append(job)
            counter += 1
            if meta_file.exists():
                l = meta_file.read_text().split(',')
                stop_at = last = int(l[0])
                if counter == last:
                    run_flag = False

        for job_index in range(len(jobs)): 
            jobs[job_index].join()
            print("Job {} finished".format(job_index))
    else:
        while run_flag:
            file = config.output_path.original_folder.joinpath(str(counter))
            while not file.exists():
                print("Waiting for original file {} to be written".format(counter))
                time.sleep(1)
                continue
            if config.params.i_period != -1:
                is_intraframe=counter % config.params.i_period == 0
            else:
                is_intraframe=False
            frame = Frame(counter, height, width, params_i=config.params.i, is_intraframe=is_intraframe)
            frame.read_from_file(file)
            frame.convert_type(np.int16)
            reconstructed_path = config.output_path.reconstructed_folder
            if not frame.is_intraframe:
                frame.prev = prev_frame
            per_block_row_bit_count = []
            RCflag_store = config.params.RCflag
            if RCflag_store == 2:
                config.params.RCflag = 0
            prev_frame, mv_dump, qtc_block_dump, split_counter, bitcount_per_row, bit_count_per_frame, per_block_row_bit_count = processing(frame, config.params, q_matrix, reconstructed_path, pool, 1, per_block_row_bit_count, 1)
            config.params.RCflag = RCflag_store

            bit_count_threshold = 8331 * config.params.qp ** 2 - 135000 * config.params.qp + 560000
            if config.params.RCflag == 2:
                if bit_count_per_frame > bit_count_threshold:
                    frame.is_intraframe = True
                    frame.prev = None
                scale_factor = bitcount_per_row / table[config.params.qp]
                prev_frame, mv_dump, qtc_block_dump, split_counter, bitcount_per_row, bit_count_per_frame, _ = processing(frame, config.params, q_matrix, reconstructed_path, pool, 2, per_block_row_bit_count, scale_factor)
                config.params.qp = qtc_block_dump.get_average_qp()
                q_matrix = quantization_matrix(config.params.i, config.params.qp)

            split_counters.append(dict(
                index=counter,
                counter=split_counter,
            ))

            if not frame.is_intraframe:
                nRefFrames = config.params.nRefFrames
                prev_pointer = prev_frame
                prev_counter = 0
                while prev_pointer.prev is not None:
                    if prev_counter == nRefFrames - 1:
                        prev_pointer.prev = None
                        break
                    else:
                        prev_counter += 1
                    prev_pointer = prev_pointer.prev

            job = pool.apply_async(func=write_data_dispatcher, args=((frame.index, mv_dump, qtc_block_dump), config,))
            jobs.append(job)
            if frame.is_intraframe:
                total_bitcount_per_row_i += bitcount_per_row
                counter_i += 1
            else: 
                total_bitcount_per_row_p += bitcount_per_row
                counter_p += 1
            counter += 1
            if meta_file.exists():
                l = meta_file.read_text().split(',')
                last = int(l[0])
                if counter == last:
                    run_flag = False
        if config.params.ParallelMode != 1:
            averge_bitcount_per_row_i = total_bitcount_per_row_i/counter_i
            print(f'the avg_bitcount_per_row_for_i_block = {int(averge_bitcount_per_row_i)},number of i_frame: {counter_i}')
        if  config.params.i_period != 1:
            averge_bitcount_per_row_p = total_bitcount_per_row_p/counter_p
            print(f'the avg_bitcount_per_row_p_for_p_block = {int(averge_bitcount_per_row_p)},number of p_frame: {counter_p}')
        with config.output_path.split_counter_file.open('a') as f:
            total_blocks = (height // config.params.i) * (width // config.params.i)
            f.write("{} {}\n".format(-1, sum(split_counter['counter'] for split_counter in split_counters) /
                    (total_blocks * len(split_counters)) * 100))
            for split_counter in split_counters:
                f.write("{} {}\n".format(split_counter['index'], split_counter['counter'] / total_blocks * 100))

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
