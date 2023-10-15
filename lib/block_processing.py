import numpy as np
from lib.utils.misc import get_padding, rounding, convert_within_range

"""
    Transform the pixel-based frame into an i x i sized block-based frame.
    Y component only.

    Parameters:
        np_array (np.ndarray): The pixel-based frame.
        params_i (int): The block size.

    Returns:
        (np_block_array, offset, block_size, np_array_padded) (tuple): The block-based frame, the offset, the block size, and the padded pixel-based frame.
"""
def block_create(np_array: np.ndarray, params_i: int) -> tuple:
    width = np_array.shape[1]
    height = np_array.shape[0]
    paded_width, paded_height = get_padding(width, height, params_i)
    pad_width = paded_width - width
    pad_height = paded_height - height
    np_array_padded = np.pad(np_array, ((0, pad_height), (0, pad_width)), 'constant', constant_values=128)

    offset = np_array_padded.shape[1] // params_i
    block_size = params_i ** 2
    # combine i rows into 1 row
    # group into i x 1, with <x> channels
    a = np_array_padded.reshape(params_i, -1, params_i, 1) # (i, -1, i, <x>)
    b = []
    # for loop width // i
    # select every i-th column
    # group into one array, size = i**2 * 3
    # = one block in a row, raster order
    for i in range(offset):
        b.append(a[:, i::offset].reshape(-1, block_size * 1)) # [all rows, start::step = width // i] (-1, i**2 * <x>))

    # combine into 1 array
    # group into i x i, with <x> channels
    # each row has width // i blocks
    np_block_array = np.block(b).reshape(-1, offset, block_size) # (-1,  width // i, i**2)
    return (np_block_array, offset, block_size, np_array_padded)

"""
    Transform the i x i blocked-based frame into a pixel-based frame.
    Y component only.

    Parameters:
        np_array (np.ndarray): The block-based frame.
        shape (height, width) (tuple): The shape of the pixel-based frame.
        params_i (int): The block size.
    
    Returns:
        np_pixel_array (np.ndarray): The pixel-based frame.
"""
def pixel_create(np_array: np.ndarray, shape: tuple, params_i: int) -> np.ndarray:
    offset = shape[1] // params_i

    d = np_array.reshape(-1, offset, params_i, params_i)
    e = []
    for i in range(offset):
        e.append(d[:, i].reshape(-1, params_i))
    
    np_pixel_array = np.block(e)
    if not shape == np_pixel_array.shape:
        raise Exception('Shape mismatch.')
    return np_pixel_array


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

"""
    TODO: block-level parallelism
"""
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

    reconstructed_frame = []
    reconstructed_counter = 0
    for mv_row in mv_dump:
        reconstructed_frame.append([])
        for mv in mv_row:
            motion_vector = mv[0]
            yx = mv[1]
            block_coor = (yx[0] - motion_vector[0], yx[1] - motion_vector[1])
            block_in_prev_frame = prev_frame[yx[0]:yx[0] + params_i, yx[1]:yx[1] + params_i]
            residual_block = residual_frame[block_coor[0]:block_coor[0] + params_i, block_coor[1]:block_coor[1] + params_i]
            reconstructed_block = block_in_prev_frame + residual_block
            reconstructed_frame[reconstructed_counter].append(reconstructed_block)
        reconstructed_counter += 1
    current_reconstructed_frame = pixel_create(np.array(reconstructed_frame), frame.shape, params_i)

    # reconstructed_q.put((frame_index, current_reconstructed_frame))
    current_reconstructed_frame = convert_within_range(current_reconstructed_frame)
    reconstructed_path.joinpath('{}'.format(frame_index)).write_bytes(current_reconstructed_frame)
    
    write_data_q.put((frame_index, mv_dump, residual_frame, average_mae))
    print('Frame {} done'.format(frame_index))
