import numpy as np
from multiprocessing import Pool
from pathlib import Path
from lib.config.config import Params
from lib.enums import Intraframe
from lib.components.frame import Frame, extend_block
from lib.components.qtc import QTCBlock, QTCFrame, quantization_matrix
from lib.components.mv import MotionVector, MotionVectorFrame
from lib.enums import VBSMarker

def calculate_mv_dispatcher(params: Params, *args, **kwargs):
    if params.FastME:
        return calc_fast_motion_vector(*args, **kwargs)
    else:
        return calc_full_range_motion_vector(*args, **kwargs)

def rdo(original_block: np.ndarray, reconstructed_block: np.ndarray, qtc_block: QTCBlock, mv: MotionVector, params_qp: int, is_intraframe=False):
    lambda_value = 0.5 ** ((params_qp - 12) / 3) * 0.85
    sad_value = np.abs(original_block - reconstructed_block).sum()
    r_vaule = len(qtc_block.to_str()) + len(mv.to_str(is_intraframe))
    return sad_value + lambda_value * r_vaule

def interframe_vbs(coor_offset: tuple, original_block: np.ndarray, original_search_windows: list, reconstructed_block: np.ndarray, qtc_block: QTCBlock, diff_mv: MotionVector, params: Params):
    """
        Implementation verification pending
    """
    block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_mv, params.qp, is_intraframe=False)
    subblock_params_i = params.i // 2
    q_matrix = quantization_matrix(subblock_params_i, params.qp - 1 if params.qp > 0 else 0)
    top_lefts = [(y, x) for y in range(0, original_block.shape[0], subblock_params_i) for x in range(0, original_block.shape[1], subblock_params_i)]
    top_lefts_in_search_window = [(y + coor_offset[0], x + coor_offset[1]) for y, x in top_lefts]
    subblock_rdo_cost = 0
    prev_motion_vector = diff_mv
    qtc_subblocks = []
    reconstructed_subblocks = []
    mv_subblocks = []
    residual_subblocks = []
    for centered_top_left_index in range(len(top_lefts)):
        centered_top_left = top_lefts[centered_top_left_index]
        top_left_in_search_window = top_lefts_in_search_window[centered_top_left_index]
        centered_subblock = original_block[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
        top_left, bottom_right = extend_block(top_left_in_search_window, subblock_params_i, (params.r, params.r, params.r, params.r), original_block.shape)
        search_windows = []
        for original_search_window in original_search_windows:
            search_window = original_search_window[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            search_windows.append(search_window)
        min_motion_vector, min_block = calculate_mv_dispatcher(params, centered_subblock, top_left_in_search_window, search_windows, top_left, subblock_params_i)
        qtc_subblock = QTCBlock(block=centered_subblock - min_block, q_matrix=q_matrix)
        qtc_subblock.block_to_qtc()
        residual_subblocks.append(qtc_subblock.block)
        reconstructed_subblock = qtc_subblock.block + min_block
        diff_submv = min_motion_vector - prev_motion_vector if prev_motion_vector is not None else min_motion_vector
        prev_motion_vector = min_motion_vector
        subblock_rdo_cost += rdo(centered_subblock, reconstructed_subblock, qtc_subblock, diff_submv, params.qp, is_intraframe=False)
        qtc_subblocks.append(qtc_subblock.qtc_block)
        reconstructed_subblocks.append(reconstructed_subblock)
        mv_subblocks.append(min_motion_vector)

    if subblock_rdo_cost < block_rdo_cost:
        qtc_stack = np.concatenate((np.concatenate((qtc_subblocks[0], qtc_subblocks[1]), axis=1), np.concatenate((qtc_subblocks[2], qtc_subblocks[3]), axis=1)), axis=0)
        temp_stack = np.concatenate((np.concatenate((residual_subblocks[0], residual_subblocks[1]), axis=1), np.concatenate((residual_subblocks[2], residual_subblocks[3]), axis=1)), axis=0)
        reconstructed_stack = np.concatenate((np.concatenate((reconstructed_subblocks[0], reconstructed_subblocks[1]), axis=1), np.concatenate((reconstructed_subblocks[2], reconstructed_subblocks[3]), axis=1)), axis=0)
        qtc_block = QTCBlock(qtc_block=qtc_stack, block=temp_stack)
        return qtc_block, reconstructed_stack, mv_subblocks
    else:
        return qtc_block, reconstructed_block, None

def intraframe_vbs(current_coor:tuple, original_block: np.ndarray, reconstructed_block: np.ndarray, reconstructed_block_dump, qtc_block: QTCBlock, diff_predictor: int, params: Params):
    """
        Implementation verification pending
    """
    block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_predictor, params.qp, is_intraframe=True)
    subblock_params_i = params.i // 2
    q_matrix = quantization_matrix(subblock_params_i, params.qp - 1 if params.qp > 0 else 0)
    top_lefts = [(y, x) for y in range(0, original_block.shape[0], subblock_params_i) for x in range(0, original_block.shape[1], subblock_params_i)]
    subblock_rdo_cost = 0
    prev_predictor = diff_predictor
    subpredictor_dump = []
    qtc_subblocks = []
    reconstructed_subblocks = np.empty(original_block.shape, dtype=int)
    residual_subblocks = []
    for centered_top_left_index in range(len(top_lefts)):
        centered_top_left = top_lefts[centered_top_left_index]
        current_block = original_block[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
        hor_top_left, _ = extend_block(centered_top_left, subblock_params_i, (0, 0, 0, 1), original_block.shape)
        ver_top_left, _ = extend_block(centered_top_left, subblock_params_i, (1, 0, 0, 0), original_block.shape)
        
        # select vertical edge
        if hor_top_left[Intraframe.HORIZONTAL.value] == centered_top_left[Intraframe.HORIZONTAL.value] + current_coor[1]:
            hor_block = np.full((subblock_params_i, 1), 128)
        else:
            if centered_top_left_index == 0 or centered_top_left_index == 2:
                hor_block = reconstructed_block_dump.raw[current_coor[0] + centered_top_left[0]:current_coor[0] + centered_top_left[0] + subblock_params_i, current_coor[1] + centered_top_left[1] - 1:current_coor[1] + centered_top_left[1]]
            else:
                hor_block = reconstructed_subblocks[hor_top_left[0]:hor_top_left[0] + subblock_params_i, hor_top_left[1]:hor_top_left[1] + 1]
        hor_block = hor_block.repeat(subblock_params_i, Intraframe.HORIZONTAL.value)
        hor_mae = np.abs(hor_block - current_block).mean().astype(int)

        # select horizontal edge
        if ver_top_left[Intraframe.VERTICAL.value] == centered_top_left[Intraframe.VERTICAL.value] + current_coor[0]:
            ver_block = np.full((1, subblock_params_i), 128)
        else:
            if centered_top_left_index == 0 or centered_top_left_index == 1:
                ver_block = reconstructed_block_dump.raw[current_coor[0] + centered_top_left[0] - 1:current_coor[0] + centered_top_left[0], current_coor[1] + centered_top_left[1]:current_coor[1] + centered_top_left[1] + subblock_params_i]
            else:
                ver_block = reconstructed_subblocks[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + subblock_params_i]
        ver_block = ver_block.repeat(subblock_params_i, Intraframe.VERTICAL.value)
        ver_mae = np.abs(ver_block - current_block).mean().astype(int)
        predictor_block = None
        if ver_mae < hor_mae:
            current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
            predictor_block = ver_block
        else:
            current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
            predictor_block = hor_block
        
        qtc_subblock = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix)
        qtc_subblock.block_to_qtc()
        residual_subblocks.append(qtc_subblock.block)
        reconstructed_subblock = qtc_subblock.block + predictor_block
        diff_subpredictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor
        prev_predictor = current_predictor
        subblock_rdo_cost += rdo(current_block, reconstructed_subblock, qtc_subblock, diff_subpredictor, params.qp, is_intraframe=True)
        qtc_subblocks.append(qtc_subblock.qtc_block)
        reconstructed_subblocks[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i] = reconstructed_subblock
        subpredictor_dump.append(current_predictor)

    if subblock_rdo_cost < block_rdo_cost:
        qtc_stack = np.concatenate((np.concatenate((qtc_subblocks[0], qtc_subblocks[1]), axis=1), np.concatenate((qtc_subblocks[2], qtc_subblocks[3]), axis=1)), axis=0)
        temp_stack = np.concatenate((np.concatenate((residual_subblocks[0], residual_subblocks[1]), axis=1), np.concatenate((residual_subblocks[2], residual_subblocks[3]), axis=1)), axis=0)
        qtc_block = QTCBlock(qtc_block=qtc_stack, block=temp_stack)
        return qtc_block, reconstructed_subblocks, subpredictor_dump
    else:
        return qtc_block, reconstructed_block, None

"""
    Nearest Neighbors search is a requirement, with MVP being the MV of the latest encoded block (MVP =(0,0) for first block in every row of (𝑖 × 𝑖) blocks). Note: any candidate block that partially or fully exists outside of the frame is not searched. Lecture 6

    Parameters:
        block (np.ndarray): The block.
        block_coor (tuple): The top left coordinates of block.
        search_windows (list): A list of search windows.
        search_window_coor (tuple): The top left coordinates of search window.
        params_i (int): The block size.
    
    Returns:
        min_motion_vector (MotionVector): The motion vector object.
        min_block (np.ndarray): The block from the search window.
"""
def calc_fast_motion_vector(block: np.ndarray, block_coor: tuple, search_windows: list, search_window_coor: tuple, params_i: int) -> tuple:
    pass

"""
    Calculate the motion vector for a block from the search window.

    Parameters:
        block (np.ndarray): The block.
        block_coor (tuple): The top left coordinates of block.
        search_windows (list): A list of search windows.
        search_window_coor (tuple): The top left coordinates of search window.
        params_i (int): The block size.
    
    Returns:
        min_motion_vector (MotionVector): The motion vector object.
        min_block (np.ndarray): The block from the search window.
"""
def calc_full_range_motion_vector(block: np.ndarray, block_coor: tuple, search_windows: list, search_window_coor: tuple, params_i: int) -> tuple:
    min_motion_vector = None
    min_yx = None
    block_reshaped = block.reshape(params_i * params_i)
    min_block = None
    for search_windows_index in range(len(search_windows)):
        search_window = search_windows[search_windows_index]
        for y in range(0, search_window.shape[0] - params_i + 1):
            actual_y = y + search_window_coor[0]
            for x in range(0, search_window.shape[1] - params_i + 1):
                actual_x = x + search_window_coor[1]
                a = search_window[y:params_i + y, x:params_i + x]
                b = a.reshape(params_i * params_i)
                motion_vector = MotionVector(actual_y - block_coor[0], actual_x - block_coor[1], ref_offset=search_windows_index, mae=np.abs(b - block_reshaped).mean())
                if min_motion_vector == None:
                    min_motion_vector = motion_vector
                    min_yx = (actual_y, actual_x)
                    min_block = a
                elif motion_vector.mae < min_motion_vector.mae:
                    min_motion_vector = motion_vector
                    min_yx = (actual_y, actual_x)
                    min_block = a
                elif motion_vector.mae == min_motion_vector.mae:
                    current_min_l1_norm = min_motion_vector.l1_norm()
                    new_min_l1_norm = motion_vector.l1_norm()
                    if new_min_l1_norm < current_min_l1_norm:
                        min_motion_vector = motion_vector
                        min_yx = (actual_y, actual_x)
                        min_block = a
                    elif new_min_l1_norm == current_min_l1_norm:
                        if actual_y < min_yx[0]:
                            min_motion_vector = motion_vector
                            min_yx = (actual_y, actual_x)
                            min_block = a
                        elif actual_y == min_yx[0]:
                            if actual_x < min_yx[1]:
                                min_motion_vector = motion_vector
                                min_yx = (actual_y, actual_x)
                                min_block = a

    return min_motion_vector, min_block

"""
    Calculate intra-frame prediction.
    No parallisim due to block dependency.

    Parameters:
        frame (Frame): The current frame.
        q_matrix (np.ndarray): The quantization matrix.

    Returns:
        qtc_block_dump (QTCFrame): The quantized transformed coefficients.
        predictor_dump (MotionVectorFrame): The predictor blocks.
        reconstructed_block_dump (Frame): The reconstructed blocks.
"""
def intraframe_prediction(frame: Frame, q_matrix: np.ndarray, params: Params) -> tuple:
    height, width = frame.shape
    block_frame = frame.pixel_to_block().astype(int)
    reconstructed_block_dump = Frame(frame.index, height, width, params_i=frame.params_i, data=np.empty(frame.shape, dtype=int))
    predictor_dump = MotionVectorFrame(is_intraframe=True, vbs_enable=params.VBSEnable)
    qtc_block_dump = QTCFrame(params_i=frame.params_i, vbs_enable=params.VBSEnable)
    y_counter = 0
    x_counter = 0
    prev_predictor = None
    for y in range(0, height, frame.params_i):
        predictor_dump.new_row()
        qtc_block_dump.new_row()
        for x in range(0, width, frame.params_i):
            current_coor = (y, x)
            current_block = block_frame[y_counter, x_counter]
            hor_top_left, _ = extend_block(current_coor, frame.params_i, (0, 0, 0, 1), (height, width))
            ver_top_left, _ = extend_block(current_coor, frame.params_i, (1, 0, 0, 0), (height, width))
            
            # select vertical edge
            if hor_top_left[Intraframe.HORIZONTAL.value] == current_coor[Intraframe.HORIZONTAL.value]:
                hor_block = np.full((frame.params_i, 1), 128)
            else:
                hor_block = reconstructed_block_dump.raw[hor_top_left[0]:hor_top_left[0] + frame.params_i, hor_top_left[1]:hor_top_left[1] + 1]
            hor_block = hor_block.repeat(frame.params_i, Intraframe.HORIZONTAL.value)
            hor_mae = np.abs(hor_block - current_block).mean().astype(int)

            # select horizontal edge
            if ver_top_left[Intraframe.VERTICAL.value] == current_coor[Intraframe.VERTICAL.value]:
                ver_block = np.full((1, frame.params_i), 128)
            else:
                ver_block = reconstructed_block_dump.raw[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + frame.params_i]
            ver_block = ver_block.repeat(frame.params_i, Intraframe.VERTICAL.value)
            ver_mae = np.abs(ver_block - current_block).mean().astype(int)

            predictor_block = None
            if ver_mae < hor_mae:
                current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
                predictor_block = ver_block
            else:
                current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
                predictor_block = hor_block

            qtc_block = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix)
            qtc_block.block_to_qtc()
            reconstructed_block = qtc_block.block + predictor_block
            diff_predictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor
            prev_predictor = current_predictor

            if params.VBSEnable:
                vbs_qtc_block, vbs_reconstructed_block, vbs_predictor = intraframe_vbs(current_coor, current_block, reconstructed_block, reconstructed_block_dump, qtc_block, diff_predictor, params)
                reconstructed_block = vbs_reconstructed_block
                if vbs_predictor is not None:
                    qtc_block = dict(
                        vbs=VBSMarker.SPLIT,
                        qtc_block=vbs_qtc_block,
                    )
                    current_predictor = dict(
                        vbs=VBSMarker.SPLIT,
                        predictor=vbs_predictor,
                    )
                    prev_predictor = vbs_predictor[-1]
                    # print('vbs used in Frame', frame.index, current_coor)
                else:
                    qtc_block = dict(
                        vbs=VBSMarker.UNSPLIT,
                        qtc_block=qtc_block,
                    )
                    current_predictor = dict(
                        vbs=VBSMarker.UNSPLIT,
                        predictor=current_predictor,
                    )

            predictor_dump.append(current_predictor)
            qtc_block_dump.append(qtc_block)
            reconstructed_block_dump.raw[y_counter * frame.params_i:y_counter * frame.params_i + frame.params_i, x_counter * frame.params_i:x_counter * frame.params_i + frame.params_i] = reconstructed_block
            x_counter += 1
        y_counter += 1
        x_counter = 0
    return (qtc_block_dump, predictor_dump, reconstructed_block_dump)

"""
    Helper function to calculate the motion vector, residual blocks, and mae values.

    Parameters:
        index (int): The index of the frame.
        frame (Frame): The current frame.
        params_r (int): The search window size.
        q_matrix (np.ndarray): The quantization matrix.
        y (int): The y coordinate of the block.

    Returns:
        index (int): The index of the frame.
        qtc_block_dump (np.ndarray): The quantized transformed coefficients.
        mv_dump (list): The motion vectors.
        mae_dump (list): The mean absolute errors.
        reconstructed_block_dump (np.ndarray): The reconstructed blocks.
"""
def mv_parallel_helper(index: int, frame: Frame, params: Params, q_matrix: np.ndarray, y: int) -> tuple:
    qtc_block_dump = []
    mv_dump = []
    reconstructed_block_dump = []
    prev_motion_vector = None
    for x in range(0, frame.width, frame.params_i):
        centered_top_left = (y, x)
        centered_block = frame.raw[centered_top_left[0]:centered_top_left[0] + frame.params_i, centered_top_left[1]:centered_top_left[1] + frame.params_i]

        top_left, bottom_right = extend_block(centered_top_left, frame.params_i, (params.r, params.r, params.r, params.r), frame.shape)
        
        current_frame = frame
        search_windows = []
        while current_frame.prev is not None:
            search_window = current_frame.prev.raw[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            search_windows.append(search_window)
            current_frame = current_frame.prev
        min_motion_vector, min_block = calculate_mv_dispatcher(params, centered_block, centered_top_left, search_windows, top_left, frame.params_i)

        qtc_block = QTCBlock(block=centered_block - min_block, q_matrix=q_matrix)
        qtc_block.block_to_qtc()
        reconstructed_block = qtc_block.block + min_block
        diff_mv = min_motion_vector - prev_motion_vector if prev_motion_vector is not None else min_motion_vector
        prev_motion_vector = min_motion_vector

        if params.VBSEnable:
            coor_offset = (centered_top_left[0] - top_left[0], centered_top_left[1] - top_left[1])
            vbs_qtc_block, vbs_reconstructed_block, vbs_mv = interframe_vbs(coor_offset, centered_block, search_windows, reconstructed_block, qtc_block, diff_mv, params)
            reconstructed_block = vbs_reconstructed_block
            if vbs_mv is not None:
                qtc_block = dict(
                    vbs=VBSMarker.SPLIT,
                    qtc_block=vbs_qtc_block,
                )
                min_motion_vector = dict(
                    vbs=VBSMarker.SPLIT,
                    predictor=vbs_mv,
                )
                prev_motion_vector = vbs_mv[-1]
                # print('vbs used in Frame', frame.index, centered_top_left)
            else:
                qtc_block = dict(
                    vbs=VBSMarker.UNSPLIT,
                    qtc_block=qtc_block,
                )
                min_motion_vector = dict(
                    vbs=VBSMarker.UNSPLIT,
                    predictor=min_motion_vector,
                )

        qtc_block_dump.append(qtc_block)
        reconstructed_block_dump.append(reconstructed_block)
        mv_dump.append(min_motion_vector)
    return (index, qtc_block_dump, mv_dump, reconstructed_block_dump)

"""
    Calculate the motion vector for a block from the search window in parallel.

    Parameters:
        frame (Frame): The current frame.
        params_r (int): The search window size.
        q_matrix (np.ndarray): The quantization matrix.
        reconstructed_path (Path): The path to write the reconstructed frame to.
        pool (Pool): The pool of processes.

    Returns:
        frame (Frame): The current reconstructed frame.
        mv_dump (MotionVectorFrame): The motion vectors.
        qtc_block_dump (QTCFrame): The quantized transformed coefficients.
"""
def calc_motion_vector_parallel_helper(frame: Frame, params: Params, q_matrix: np.ndarray, reconstructed_path: Path, pool: Pool) -> tuple:
    print("Processing", frame.index)
    if (frame.is_intraframe is False and frame.prev is None) or ((frame.prev is not None) and (frame.prev.index + 1 != frame.index)):
        raise Exception('Frame index mismatch. Current: {}, Previous: {}'.format(frame.index, frame.prev.index))
    
    if frame.is_intraframe:
        qtc_block_dump, mv_dump, current_reconstructed_frame = intraframe_prediction(frame, q_matrix, params)
    else:
        jobs = []
        results = []
        counter = 0
        for y in range(0, frame.shape[0], frame.params_i):
            job = pool.apply_async(func=mv_parallel_helper, args=(
                counter,
                frame, 
                params, 
                q_matrix,
                y,
            ))
            jobs.append(job)
            counter += 1

        for job in jobs:
            results.append(job.get())
        
        results.sort(key=lambda x: x[0])
        qtc_block_dump = QTCFrame(length=len(results), vbs_enable=params.VBSEnable)
        mv_dump = MotionVectorFrame(length=len(results), vbs_enable=params.VBSEnable)
        reconstructed_block_dump = [None] * len(results)
        for result in results:
            index = result[0]
            qtc_block_dump.append_list(index, result[1])
            mv_dump.append_list(index, result[2])
            reconstructed_block_dump[index] = result[3]
        
        current_reconstructed_frame = Frame(frame=frame)
        current_reconstructed_frame.block_to_pixel(reconstructed_block_dump)

    current_reconstructed_frame.convert_within_range()

    current_reconstructed_frame.dump(reconstructed_path.joinpath('{}'.format(frame.index)))
    
    return current_reconstructed_frame, mv_dump, qtc_block_dump
