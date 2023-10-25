import numpy as np
from multiprocessing import Pool, Queue
from pathlib import Path
from lib.utils.misc import extend_block
from lib.utils.enums import Intraframe
from lib.utils.quantization import get_qtc_and_reconstructed_block
from lib.utils.differential import frame_differential_encoding
from lib.components.frame import Frame

"""
    Calculate the motion vector for a block from the search window.

    Parameters:
        block (np.ndarray): The block.
        block_coor (tuple): The top left coordinates of block.
        search_window (np.ndarray): The search window.
        search_window_coor (tuple): The top left coordinates of search window.
        params_i (int): The block size.
    
    Returns:
        min_mae (int): The minimum mean absolute error.
        min_motion_vector (tuple): The motion vector (y, x).
        min_block (np.ndarray): The block from the search window.
"""
def calc_motion_vector(block: np.ndarray, block_coor: tuple, search_window: np.ndarray, search_window_coor: tuple, params_i: int) -> tuple:
    min_mae = -1
    min_motion_vector = None
    min_yx = None
    block_reshaped = block.reshape(params_i * params_i)
    min_block = None
    for y in range(0, search_window.shape[0] - params_i + 1):
        actual_y = y + search_window_coor[0]
        for x in range(0, search_window.shape[1] - params_i + 1):
            actual_x = x + search_window_coor[1]
            a = search_window[y:params_i + y, x:params_i + x]
            b = a.reshape(params_i * params_i)
            mae = np.abs(b - block_reshaped).mean().astype(int)
            motion_vector = (actual_y - block_coor[0], actual_x - block_coor[1])
            if min_mae == -1:
                min_mae = mae
                min_motion_vector = motion_vector
                min_yx = (actual_y, actual_x)
                min_block = a
            elif mae < min_mae:
                min_mae = mae
                min_motion_vector = motion_vector
                min_yx = (actual_y, actual_x)
                min_block = a
            elif mae == min_mae:
                current_min_l1_norm = (abs(min_motion_vector[0]) + abs(min_motion_vector[1]))
                new_min_l1_norm = (abs(motion_vector[0]) + abs(motion_vector[1]))
                if new_min_l1_norm < current_min_l1_norm:
                    min_mae = mae
                    min_motion_vector = motion_vector
                    min_yx = (actual_y, actual_x)
                    min_block = a
                elif new_min_l1_norm == current_min_l1_norm:
                    if actual_y < min_yx[0]:
                        min_mae = mae
                        min_motion_vector = motion_vector
                        min_yx = (actual_y, actual_x)
                        min_block = a
                    elif actual_y == min_yx[0]:
                        if actual_x < min_yx[1]:
                            min_mae = mae
                            min_motion_vector = motion_vector
                            min_yx = (actual_y, actual_x)
                            min_block = a

    return min_mae, min_motion_vector, min_block

"""
    Calculate intra-frame prediction.
    No parallisim due to block dependency.

    Parameters:
        frame (np.ndarray): The current frame.
        params_i (int): The block size.
        q_matrix (np.ndarray): The quantization matrix.

    Returns:
        qtc_block_dump (list): The quantized transformed coefficients.
        predictor_dump (list): The predictor blocks.
        mae_dump (list): The mean absolute errors.
        reconstructed_block_dump (np.ndarray): The reconstructed blocks.
"""
def intraframe_prediction(frame, q_matrix: np.ndarray) -> tuple:
    height, width = frame.shape
    block_frame = frame.pixel_to_block().astype(int)
    # reconstructed_block_dump = np.empty(frame.shape, dtype=int)
    reconstructed_block_dump = Frame(frame.index, height, width, params_i=frame.params_i, data=np.empty(frame.shape, dtype=int))
    predictor_dump = []
    qtc_block_dump = []
    mae_dump = []
    y_counter = 0
    x_counter = 0
    for y in range(0, height, frame.params_i):
        predictor_dump.append([])
        qtc_block_dump.append([])
        mae_dump.append([])
        for x in range(0, width, frame.params_i):
            current_coor = (y, x)
            current_block = block_frame[y_counter, x_counter]
            hor_top_left, _ = extend_block(current_coor, frame.params_i, (0, 0, 0, 1), (height, width))
            ver_top_left, _ = extend_block(current_coor, frame.params_i, (1, 0, 0, 0), (height, width))
            
            if hor_top_left[Intraframe.HORIZONTAL.value] == current_coor[Intraframe.HORIZONTAL.value]:
                hor_block = np.full((frame.params_i, 1), 128)
            else:
                hor_block = reconstructed_block_dump.raw[hor_top_left[0]:hor_top_left[0] + frame.params_i, hor_top_left[1]:hor_top_left[1] + 1]
            hor_block = hor_block.repeat(frame.params_i, Intraframe.HORIZONTAL.value)
            hor_mae = np.abs(hor_block - current_block).mean().astype(int)

            if ver_top_left[Intraframe.VERTICAL.value] == current_coor[Intraframe.VERTICAL.value]:
                ver_block = np.full((1, frame.params_i), 128)
            else:
                ver_block = reconstructed_block_dump.raw[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + frame.params_i]
            ver_block = ver_block.repeat(frame.params_i, Intraframe.VERTICAL.value)
            ver_mae = np.abs(ver_block - current_block).mean().astype(int)

            predictor_block = None
            if ver_mae < hor_mae:
                predictor_dump[y_counter].append(Intraframe.VERTICAL.value)
                predictor_block = ver_block
                mae_dump[y_counter].append(ver_mae)
            else:
                predictor_dump[y_counter].append(Intraframe.HORIZONTAL.value)
                predictor_block = hor_block
                mae_dump[y_counter].append(hor_mae)

            qtc_dump, reconstructed_block = get_qtc_and_reconstructed_block(current_block, predictor_block, q_matrix)
            qtc_block_dump[y_counter].append(qtc_dump)
            reconstructed_block_dump.raw[y_counter * frame.params_i:y_counter * frame.params_i + frame.params_i, x_counter * frame.params_i:x_counter * frame.params_i + frame.params_i] = reconstructed_block
            x_counter += 1
        y_counter += 1
        x_counter = 0
    return (qtc_block_dump, predictor_dump, mae_dump, reconstructed_block_dump)

"""
    Helper function to calculate the motion vector, residual blocks, and mae values.

    Parameters:
        index (int): The index of the frame.
        frame (np.ndarray): The current frame.
        params_i (int): The block size.
        params_r (int): The search window size.
        q_matrix (np.ndarray): The quantization matrix.
        prev_frame (np.ndarray): The previous frame.
        y (int): The y coordinate of the block.

    Returns:
        index (int): The index of the frame.
        qtc_block_dump (np.ndarray): The quantized transformed coefficients.
        mv_dump (list): The motion vectors.
        mae_dump (list): The mean absolute errors.
        reconstructed_block_dump (np.ndarray): The reconstructed blocks.
"""
def mv_parallel_helper(index: int, frame, params_r: int, q_matrix: np.ndarray, y: int) -> tuple:
    qtc_block_dump = []
    mv_dump = []
    mae_dump = []
    reconstructed_block_dump = []
    for x in range(0, frame.width, frame.params_i):
        centered_top_left = (y, x)
        centered_block = frame.raw[centered_top_left[0]:centered_top_left[0] + frame.params_i, centered_top_left[1]:centered_top_left[1] + frame.params_i]

        top_left, bottom_right = extend_block(centered_top_left, frame.params_i, (params_r, params_r, params_r, params_r), frame.shape)
        
        search_window = frame.prev.raw[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        min_mae, min_motion_vector, min_block = calc_motion_vector(centered_block, centered_top_left, search_window, top_left, frame.params_i)

        qtc_dump, reconstructed_block = get_qtc_and_reconstructed_block(centered_block, min_block, q_matrix)
        qtc_block_dump.append(qtc_dump)
        reconstructed_block_dump.append(reconstructed_block)

        mv_dump.append(min_motion_vector)
        mae_dump.append(min_mae)
    return (index, qtc_block_dump, mv_dump, mae_dump, reconstructed_block_dump)

"""
    Calculate the motion vector for a block from the search window in parallel.

    Parameters:
        frame (np.ndarray): The current frame.
        frame_index (int): The current frame index.
        prev_frame (np.ndarray): The previous frame.
        prev_index (int): The previous frame index.
        params_i (int): The block size.
        params_r (int): The search window size.
        is_intraframe (bool): Whether the current frame is an intraframe.
        q_matrix (np.ndarray): The quantization matrix.
        write_data_q (Queue): The queue to write data to.
        reconstructed_path (Path): The path to write the reconstructed frame to.
        pool (Pool): The pool of processes.
"""
def calc_motion_vector_parallel_helper(frame, params_r, q_matrix: np.ndarray, write_data_q: Queue, reconstructed_path: Path, pool: Pool) -> None:
    print("Dispatched", frame.index)
    if frame.prev.index + 1 != frame.index:
        raise Exception('Frame index mismatch. Current: {}, Previous: {}'.format(frame.index, frame.prev.index))
    
    if frame.is_intraframe:
        qtc_block_dump, mv_dump, mae_dump, current_reconstructed_frame = intraframe_prediction(frame, q_matrix)
    else:
        jobs = []
        results = []
        counter = 0
        for y in range(0, frame.shape[0], frame.params_i):
            job = pool.apply_async(func=mv_parallel_helper, args=(
                counter,
                frame, 
                params_r, 
                q_matrix,
                y,
            ))
            jobs.append(job)
            counter += 1

        for job in jobs:
            results.append(job.get())
        
        results.sort(key=lambda x: x[0])
        qtc_block_dump = [None] * len(results)
        mv_dump = []
        mae_dump = [None] * len(results)
        reconstructed_block_dump = [None] * len(results)
        for result in results:
            index = result[0]
            qtc_block_dump[index] = result[1]
            mv_dump.append(result[2])
            mae_dump[index] = result[3]
            reconstructed_block_dump[index] = result[4]
        
        current_reconstructed_frame = Frame(index, frame.height, frame.width, params_i=frame.params_i)
        current_reconstructed_frame.block_to_pixel(np.array(reconstructed_block_dump))
    
    average_mae = np.array(mae_dump).mean().astype(int)
    mv_dump = frame_differential_encoding(mv_dump, frame.is_intraframe)
    current_reconstructed_frame.convert_within_range()

    current_reconstructed_frame.dump(reconstructed_path.joinpath('{}'.format(frame.index)))
    
    write_data_q.put((frame.index, (frame.is_intraframe, mv_dump), qtc_block_dump, average_mae))
    print('Frame {} done'.format(frame.index))
