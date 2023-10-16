import numpy as np
import multiprocessing as mp
from lib.utils.misc import rounding, convert_within_range, pixel_create, reconstruct_frame

def calc_motion_vector(block, block_coor, search_window, search_window_coor, params_i):
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

    return min_motion_vector, min_mae, min_yx, min_block

def calc_motion_vector_helper(frame, frame_index, prev_frame, prev_index, params_i, params_r, write_data_q, reconstructed_path):
    print("Dispatched", frame_index)
    if prev_index + 1 != frame_index:
        raise Exception('Frame index mismatch. Current: {}, Previous: {}'.format(frame_index, prev_index))
    mv_dump = []
    residual_block_dump = []
    mae_dump = []
    counter = 0
    for y in range(0, frame.shape[0], params_i):
        residual_block_dump.append([])
        mv_dump.append([])
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

            min_motion_vector, min_mae, min_yx, min_block = calc_motion_vector(centered_block, centered_top_left, search_window, top_left, params_i)
            mv_dump[counter].append((min_motion_vector, min_yx))
            mae_dump.append(min_mae)

            residual_block = centered_block - min_block
            residual_block_rounded = np.array([rounding(x, 2) for x in residual_block.reshape(params_i * params_i)])
            residual_block_dump[counter].append(residual_block_rounded)
        counter += 1
    residual_frame = pixel_create(np.array(residual_block_dump), frame.shape, params_i)
    average_mae = np.array(mae_dump).mean().astype(int)

    current_reconstructed_frame = reconstruct_frame(mv_dump, prev_frame, residual_frame, params_i, frame.shape[0], frame.shape[1])
    current_reconstructed_frame = convert_within_range(current_reconstructed_frame)
    
    reconstructed_path.joinpath('{}'.format(frame_index)).write_bytes(current_reconstructed_frame)
    
    write_data_q.put((frame_index, mv_dump, residual_frame, average_mae))
    print('Frame {} done'.format(frame_index))

def parallel_helper(index, frame, params_i, params_r, prev_frame, y):
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

        min_motion_vector, min_mae, min_yx, min_block = calc_motion_vector(centered_block, centered_top_left, search_window, top_left, params_i)
        mv_dump.append((min_motion_vector, min_yx))
        mae_dump.append(min_mae)

        residual_block = centered_block - min_block
        residual_block_rounded = np.array([rounding(x, 2) for x in residual_block.reshape(params_i * params_i)])
        residual_block_dump.append(residual_block_rounded)
    return (index, residual_block_dump, mv_dump, mae_dump)

def calc_motion_vector_parallel_helper(frame, frame_index, prev_frame, prev_index, params_i, params_r, write_data_q, reconstructed_path, pool):
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
            prev_frame,
            y,
        ))
        jobs.append(job)
        counter += 1

    for job in jobs:
        results.append(job.get())

    mv_dump = [None] * len(results)
    residual_block_dump = [None] * len(results) 
    mae_dump = [None] * len(results) 
    for result in results:
        index = result[0]
        residual_block_dump[index] = result[1]
        mv_dump[index] = result[2]
        mae_dump[index] = result[3]
    
    residual_frame = pixel_create(np.array(residual_block_dump), frame.shape, params_i)
    average_mae = np.array(mae_dump).mean().astype(int)

    current_reconstructed_frame = reconstruct_frame(mv_dump, prev_frame, residual_frame, params_i, frame.shape[0], frame.shape[1])
    current_reconstructed_frame = convert_within_range(current_reconstructed_frame)
    
    reconstructed_path.joinpath('{}'.format(frame_index)).write_bytes(current_reconstructed_frame)
    
    write_data_q.put((frame_index, mv_dump, residual_frame, average_mae))
    print('Frame {} done'.format(frame_index))