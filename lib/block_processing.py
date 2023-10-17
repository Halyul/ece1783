import numpy as np
from multiprocessing import Pool, Queue
from pathlib import Path
from lib.utils.misc import *

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
    Helper function to calculate the motion vector, residual blocks, and mae values.

    Parameters:
        index (int): The index of the frame.
        frame (np.ndarray): The current frame.
        params_i (int): The block size.
        params_r (int): The search window size.
        q_matrix (np.ndarray): The quantization matrix.
        prev_frame (np.ndarray): The previous frame.
        y (int): The y coordinate of the block.
"""
def parallel_helper(index: int, frame: np.ndarray, params_i: int, params_r: int, q_matrix: np.ndarray, prev_frame: np.ndarray, y: int) -> tuple:
    residual_block_dump = []
    mv_dump = []
    mae_dump = []
    for x in range(0, frame.shape[1], params_i):
        top_left = centered_top_left = (y, x)
        centered_block = frame[top_left[0]:top_left[0] + params_i, top_left[1]:top_left[1] + params_i]
        bottom_right = (top_left[0] + params_i, top_left[1] + params_i)

        y_offset = top_left[0] - params_r
        if y_offset >= 0:
            top_left = (y_offset, top_left[1])
            bottom_right = (bottom_right[0] + params_r, bottom_right[1])
        else:
            top_left = (0, top_left[1])
            bottom_right = (bottom_right[0] + params_r, bottom_right[1])

        # set the bottom right corner of the search window
        x_offset = top_left[1] - params_r
        if x_offset >= 0:
            top_left = (top_left[0], x_offset)
            bottom_right = (bottom_right[0], bottom_right[1] + params_r)
        else:
            top_left = (top_left[0], 0)
            bottom_right = (bottom_right[0], bottom_right[1] + params_r)
        
        search_window = prev_frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        min_mae, min_motion_vector, min_block = calc_motion_vector(centered_block, centered_top_left, search_window, top_left, params_i)
        mv_dump.append(min_motion_vector)
        mae_dump.append(min_mae)

        residual_block = centered_block - min_block
        residual_block_transformed = dct2(residual_block).astype(int)
        qtc_dump = tc_to_qtc(residual_block_transformed, q_matrix)
        residual_block_dump.append(qtc_dump)
    return (index, residual_block_dump, mv_dump, mae_dump)

"""
    Calculate the motion vector for a block from the search window in parallel.

    Parameters:
        frame (np.ndarray): The current frame.
        frame_index (int): The current frame index.
        prev_frame (np.ndarray): The previous frame.
        prev_index (int): The previous frame index.
        params_i (int): The block size.
        params_r (int): The search window size.
        q_matrix (np.ndarray): The quantization matrix.
        write_data_q (Queue): The queue to write data to.
        reconstructed_path (Path): The path to write the reconstructed frame to.
        pool (Pool): The pool of processes.
"""
def calc_motion_vector_parallel_helper(frame: np.ndarray, frame_index: int, prev_frame: np.ndarray, prev_index: int, params_i: int, params_r: int, q_matrix: np.ndarray, write_data_q: Queue, reconstructed_path: Path, pool: Pool) -> None:
    print("Dispatched", frame_index)
    if prev_index + 1 != frame_index:
        raise Exception('Frame index mismatch. Current: {}, Previous: {}'.format(frame_index, prev_index))
    
    jobs = []
    results = []
    counter = 0
    for y in range(0, frame.shape[0], params_i):
        job = pool.apply_async(func=parallel_helper, args=(
            counter,
            frame, 
            params_i, 
            params_r, 
            q_matrix,
            prev_frame,
            y,
        ))
        jobs.append(job)
        counter += 1

    for job in jobs:
        results.append(job.get())
    
    mv_dump = [None] * len(results)
    qtc_block_dump = [None] * len(results) 
    mae_dump = [None] * len(results)
    for result in results:
        index = result[0]
        qtc_block_dump[index] = result[1]
        mv_dump[index] = result[2]
        mae_dump[index] = result[3]
    
    qtc_block_dump = np.array(qtc_block_dump)
    residual_block_dump = frame_qtc_to_tc(qtc_block_dump, q_matrix)
    predicted_frame = construct_predicted_frame(mv_dump, prev_frame, params_i)
    residual_frame = residual_coefficients_to_residual_frame(residual_block_dump, params_i, frame.shape)
    average_mae = np.array(mae_dump).mean().astype(int)

    current_reconstructed_frame = predicted_frame + residual_frame
    current_reconstructed_frame = convert_within_range(current_reconstructed_frame)
    
    reconstructed_path.joinpath('{}'.format(frame_index)).write_bytes(current_reconstructed_frame)
    
    write_data_q.put((frame_index, mv_dump, pixel_create(qtc_block_dump, frame.shape, params_i), average_mae))
    print('Frame {} done'.format(frame_index))